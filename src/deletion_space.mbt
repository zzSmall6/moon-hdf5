///|
fn Hdf5File::deletion_raw_sections(
  self : Hdf5File,
  header : ObjectHeader,
  metadata : DatasetMetadata,
) -> Array[Hdf5FreeSection] raise Hdf5Error {
  if datatype_contains_vlen(metadata.datatype) ||
    header.first(msg_external_data_files) is Some(_) {
    return []
  }
  let absent = btree_v1_undefined_address(self.superblock.offset_size)
  match metadata.layout.kind() {
    Contiguous =>
      if metadata.layout.data_address == absent ||
        metadata.layout.data_size == 0UL {
        []
      } else {
        [
          {
            address: metadata.layout.data_address,
            size: metadata.layout.data_size,
            section_type: 0,
          },
        ]
      }
    Chunked => {
      let records = match metadata.layout.chunk_index {
        Implicit => return []
        Single(_, _) =>
          if metadata.layout.data_address == absent {
            []
          } else {
            [
              single_chunk_slot_record(
                header,
                metadata,
                chunk_byte_size(
                  metadata.layout.chunk_dimensions,
                  metadata.datatype.size,
                ),
                self.superblock.length_size,
              ),
            ]
          }
        _ => {
          let filters = match header.first(msg_filter_pipeline) {
            Some(message) =>
              parse_filters(message, custom_filters=self.custom_filters)
            None => []
          }
          collect_raw_chunk_records(
            self.bytes,
            metadata.layout,
            metadata.dataspace,
            metadata.datatype,
            filters,
            self.superblock.offset_size,
            self.superblock.length_size,
            self.superblock.base_address,
          )
        }
      }
      records
      .filter(record => record.address != absent)
      .map(record => {
        address: record.address,
        size: record.size,
        section_type: 0,
      })
    }
    _ => []
  }
}

///|
fn Hdf5File::prepare_deleted_storage(
  self : Hdf5File,
  target : Hdf5Node,
) -> ChunkSpacePatch? raise Hdf5Error {
  if target.kind != Dataset {
    return None
  }
  let info = match self.file_space_info() {
    Some(info) if info.strategy == FsmAggregate && info.persistent => info
    _ => return None
  }
  let managers = self.free_space_managers()
  let mut found : Hdf5FreeSpaceManager? = None
  for manager in managers {
    if manager.address == info.manager_addresses[2] {
      found = Some(manager)
    }
  }
  let manager = match found {
    Some(manager) if manager.ghost_sections == 0UL &&
      manager.sections.all(section => section.section_type == 0) => manager
    _ => return None
  }
  let header = self.object_header_at(target.address)
  let metadata = read_dataset_metadata(
    header,
    self.superblock.offset_size,
    self.superblock.length_size,
  )
  let released = self.deletion_raw_sections(header, metadata)
  if released.is_empty() {
    return None
  }
  let nodes = self.walk()
  let paths = ["/"]
  let addresses = [self.root_address()]
  for node in nodes {
    if node.kind == Unknown {
      return None
    }
    if !addresses.contains(node.address) {
      addresses.push(node.address)
      paths.push(node.path)
    }
  }
  for path in paths {
    for attribute in self.attributes(path) {
      if datatype_contains_reference(attribute.datatype) {
        return None
      }
    }
    let node = self.resolve(path)
    if node.kind == Dataset {
      let other_header = self.object_header_at(node.address)
      let other = self.read_dataset_metadata(path)
      if datatype_contains_reference(other.datatype) {
        return None
      }
      if node.address != target.address {
        if datatype_contains_vlen(other.datatype) ||
          other.layout.chunk_index is Implicit ||
          other.layout.kind() == Virtual {
          return None
        }
        let live = self.deletion_raw_sections(other_header, other)
        guard !live.any(region => {
          released.any(section => {
            section.address < region.address + region.size &&
            region.address < section.address + section.size
          })
        }) else {
          raise InvalidField("deleted dataset storage overlaps another dataset")
        }
      }
    }
  }
  let start = address_to_offset(manager.address, self.superblock.base_address)
  let maximum = read_uint_at(
    self.bytes,
    start + 14 + 4 * self.superblock.length_size,
    self.superblock.length_size,
    true,
  )
  space_coalesce(released, maximum) |> ignore
  for section in released {
    guard section.size <= 0x7FFFFFFFUL else {
      raise InvalidField("deleted dataset storage is too large")
    }
    self.space_read(section.address, section.size.to_int()) |> ignore
    guard !managers.any(item => {
      item.sections.any(free => {
        section.address < free.address + free.size &&
        free.address < section.address + section.size
      })
    }) else {
      raise InvalidField("deleted dataset storage overlaps recorded free space")
    }
  }
  Some({
    manager,
    threshold: info.threshold,
    maximum,
    remaining: manager.sections.copy(),
    released,
    changed: true,
  })
}
