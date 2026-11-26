///|
priv struct ChunkSpacePatch {
  manager : Hdf5FreeSpaceManager
  threshold : UInt64
  maximum : UInt64
  mut remaining : Array[Hdf5FreeSection]
  released : Array[Hdf5FreeSection]
  mut changed : Bool
}

///|
fn Hdf5File::chunk_space_patch(
  self : Hdf5File,
  header : ObjectHeader,
  metadata : DatasetMetadata,
) -> ChunkSpacePatch? raise Hdf5Error {
  if metadata.layout.chunk_index is Implicit {
    return None
  }
  let info = match self.file_space_info() {
    Some(info) if info.strategy == FsmAggregate && info.persistent => info
    _ => return None
  }
  let managers = self.free_space_managers()
  let mut selected : Hdf5FreeSpaceManager? = None
  for manager in managers {
    if manager.address == info.manager_addresses[2] {
      selected = Some(manager)
    }
  }
  let manager = match selected {
    Some(manager) if manager.ghost_sections == 0UL &&
      manager.sections.all(section => section.section_type == 0) => manager
    _ => return None
  }
  let start = address_to_offset(manager.address, self.superblock.base_address)
  let maximum = read_uint_at(
    self.bytes,
    start + 14 + 4 * self.superblock.length_size,
    self.superblock.length_size,
    true,
  )
  let filters = match header.first(msg_filter_pipeline) {
    Some(message) => parse_filters(message, custom_filters=self.custom_filters)
    None => []
  }
  let absent = btree_v1_undefined_address(self.superblock.offset_size)
  let records = match metadata.layout.chunk_index {
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
  let live = records
    .filter(record => record.address != absent)
    .map(record => {
      address: record.address,
      size: record.size,
      section_type: 0,
    })
  space_coalesce(live, maximum) |> ignore
  for section in live {
    self.space_read(section.address, section.size.to_int()) |> ignore
    guard !managers.any(item => {
      item.sections.any(free => {
        section.address < free.address + free.size &&
        free.address < section.address + section.size
      })
    }) else {
      raise InvalidField("live chunk storage overlaps recorded free space")
    }
  }
  Some({
    manager,
    threshold: info.threshold,
    maximum,
    remaining: manager.sections.copy(),
    released: [],
    changed: false,
  })
}

///|
fn Hdf5File::allocate_patch_chunk(
  self : Hdf5File,
  space : ChunkSpacePatch?,
  output : FixedArray[Byte],
  tail : @buffer.Buffer,
  tail_base : Int32,
  payload : Bytes,
) -> UInt64 raise Hdf5Error {
  if space is Some(state) {
    if space_allocate(state.remaining, payload.length().to_uint64(), 8UL)
      is Some((address, remaining)) {
      state.remaining = remaining
      state.changed = true
      let absolute = address_to_offset(address, self.superblock.base_address)
      output.blit_from_bytes(absolute, payload, 0, payload.length())
      return absolute.to_uint64()
    }
  }
  append_aligned_chunk_to_tail(tail, tail_base, payload)
}

///|
fn release_patch_chunk(
  space : ChunkSpacePatch?,
  address : UInt64,
  size : UInt64,
) -> Unit {
  if space is Some(state) && size > 0UL {
    state.released.push({ address, size, section_type: 0, })
    state.changed = true
  }
}

///|
fn finish_chunk_space_patch(
  bytes : Bytes,
  space : ChunkSpacePatch?,
) -> Bytes raise Hdf5Error {
  let state = match space {
    Some(state) if state.changed => state
    _ => return bytes
  }
  let sections = state.remaining.copy()
  for section in state.released {
    sections.push(section)
  }
  let normalized = space_coalesce(sections, state.maximum).filter(section => {
    section.size >= state.threshold
  })
  let file = Hdf5File::parse(bytes)
  let output = bytes.to_fixedarray()
  let tail = @buffer.Buffer::Buffer()
  file.write_space_sections(state.manager, normalized, output, tail)
  file.finish_appended_image(output, tail)
}
