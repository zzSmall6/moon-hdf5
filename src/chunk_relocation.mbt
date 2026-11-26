///|
fn Hdf5File::relocate_chunk_storage(
  self : Hdf5File,
  header : ObjectHeader,
  metadata : DatasetMetadata,
  alignment : UInt64,
) -> Bytes raise Hdf5Error {
  guard header.first(msg_external_data_files) is None &&
    !(metadata.layout.chunk_index is Implicit) else {
    raise UnsupportedFeature(
      "chunk relocation requires internal explicit-index storage",
    )
  }
  let info = match self.file_space_info() {
    Some(info) if info.strategy == FsmAggregate && info.persistent => info
    _ =>
      raise UnsupportedFeature(
        "chunk relocation requires persistent aggregate free space",
      )
  }
  let managers = self.free_space_managers()
  let mut raw_manager : Hdf5FreeSpaceManager? = None
  for manager in managers {
    if manager.address == info.manager_addresses[2] {
      raw_manager = Some(manager)
    }
  }
  let manager = match raw_manager {
    Some(manager) => manager
    None =>
      raise UnsupportedFeature("chunk relocation requires a raw-data manager")
  }
  guard manager.ghost_sections == 0UL &&
    manager.sections.all(section => section.section_type == 0) else {
    raise UnsupportedFeature(
      "chunk relocation requires simple sections without ghosts",
    )
  }
  let filters = match header.first(msg_filter_pipeline) {
    Some(message) => parse_filters(message, custom_filters=self.custom_filters)
    None => []
  }
  let records = match metadata.layout.chunk_index {
    Single(_, _) => {
      let absent = btree_v1_undefined_address(self.superblock.offset_size)
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
    }
    _ =>
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
  let absent = btree_v1_undefined_address(self.superblock.offset_size)
  let allocated = records.filter(record => record.address != absent)
  if allocated.is_empty() {
    return self.bytes
  }
  let live = allocated.map(record => {
    address: record.address,
    size: record.size,
    section_type: 0,
  })
  let manager_start = address_to_offset(
    manager.address,
    self.superblock.base_address,
  )
  let maximum = read_uint_at(
    self.bytes,
    manager_start + 14 + 4 * self.superblock.length_size,
    self.superblock.length_size,
    true,
  )
  space_coalesce(live, maximum) |> ignore
  for record in allocated {
    guard record.address_offset >= 0 &&
      record.address_width > 0 &&
      record.size > 0UL &&
      record.size <= 0x7FFFFFFFUL else {
      raise UnsupportedFeature(
        "chunk index does not expose a writable allocated record",
      )
    }
    self.space_read(record.address, record.size.to_int()) |> ignore
    guard !managers.any(item => {
      item.sections.any(section => {
        section.address < record.address + record.size &&
        record.address < section.address + section.size
      })
    }) else {
      raise InvalidField("live chunk storage overlaps recorded free space")
    }
  }
  let mut remaining = manager.sections.copy()
  let destinations : Array[UInt64] = []
  for record in allocated {
    let (address, rest) = match
      space_allocate(remaining, record.size, alignment) {
      Some(result) => result
      None =>
        raise UnsupportedFeature(
          "insufficient persisted free space to relocate all chunks",
        )
    }
    destinations.push(address)
    remaining = rest
  }
  for section in live {
    remaining.push(section)
  }
  let sections = space_coalesce(remaining, maximum).filter(section => {
    section.size >= info.threshold
  })
  let output = self.bytes.to_fixedarray()
  let tail = @buffer.Buffer::Buffer()
  self.write_space_sections(manager, sections, output, tail)
  let checksums : Array[(Int32, Int32)] = []
  for index in 0..<allocated.length() {
    let record = allocated[index]
    let payload = self.space_read(record.address, record.size.to_int())
    let destination = address_to_offset(
      destinations[index],
      self.superblock.base_address,
    )
    output.blit_from_bytes(destination, payload, 0, payload.length())
    patch_chunk_address_field(output, record, destinations[index])
    if record.metadata_start >= 0 && record.metadata_checksum_offset >= 0 {
      let region = (record.metadata_start, record.metadata_checksum_offset)
      if !checksums.contains(region) {
        checksums.push(region)
      }
    }
  }
  for (start, checksum) in checksums {
    patch_metadata_checksum_in_place(output, start, checksum)
  }
  self.finish_appended_image(output, tail)
}
