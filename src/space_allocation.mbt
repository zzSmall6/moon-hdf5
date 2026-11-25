///|
fn space_allocate(
  sections : Array[Hdf5FreeSection],
  size : UInt64,
  alignment : UInt64,
) -> (UInt64, Array[Hdf5FreeSection])? raise Hdf5Error {
  guard size > 0UL && alignment > 0UL && (alignment & (alignment - 1UL)) == 0UL else {
    raise InvalidArgument(
      "space allocation requires positive size and power-of-two alignment",
    )
  }
  let mut selected = -1
  let mut address = 0UL
  for index in 0..<sections.length() {
    let section = sections[index]
    guard section.section_type == 0 &&
      section.size > 0UL &&
      section.address <= 0xFFFFFFFFFFFFFFFFUL - section.size else {
      raise InvalidField("invalid simple free-space section")
    }
    if section.address <= 0xFFFFFFFFFFFFFFFFUL - (alignment - 1UL) {
      let start = (section.address + alignment - 1UL) & (0UL - alignment)
      let padding = start - section.address
      if padding <= section.size &&
        size <= section.size - padding &&
        (
          selected < 0 ||
          section.size < sections[selected].size ||
          (section.size == sections[selected].size && start < address)
        ) {
        selected = index
        address = start
      }
    }
  }
  if selected < 0 {
    return None
  }
  let remaining : Array[Hdf5FreeSection] = []
  for index in 0..<sections.length() {
    let section = sections[index]
    if index != selected {
      remaining.push(section)
    } else {
      if address > section.address {
        remaining.push({
          address: section.address,
          size: address - section.address,
          section_type: 0,
        })
      }
      let end = address + size
      if end < section.address + section.size {
        remaining.push({
          address: end,
          size: section.address + section.size - end,
          section_type: 0,
        })
      }
    }
  }
  Some((address, remaining))
}

///|
fn space_coalesce(
  sections : Array[Hdf5FreeSection],
  maximum_size : UInt64,
) -> Array[Hdf5FreeSection] raise Hdf5Error {
  let sorted = sections.copy()
  sorted.sort_by(fn(left, right) { left.address.compare(right.address) })
  let result : Array[Hdf5FreeSection] = []
  for section in sorted {
    guard section.section_type == 0 &&
      section.size > 0UL &&
      section.size <= maximum_size &&
      section.address <= 0xFFFFFFFFFFFFFFFFUL - section.size else {
      raise InvalidField("invalid released free-space section")
    }
    if result.length() > 0 {
      let previous = result[result.length() - 1]
      let end = previous.address + previous.size
      guard section.address >= end else {
        raise InvalidField("released storage overlaps a free-space section")
      }
      if section.address == end && previous.size <= maximum_size - section.size {
        result[result.length() - 1] = {
          address: previous.address,
          size: previous.size + section.size,
          section_type: 0,
        }
        continue
      }
    }
    result.push(section)
  }
  result
}

///|
fn Hdf5File::write_space_sections(
  self : Hdf5File,
  manager : Hdf5FreeSpaceManager,
  sections : Array[Hdf5FreeSection],
  output : FixedArray[Byte],
  tail : @buffer.Buffer,
) -> Unit raise Hdf5Error {
  let offsets = self.superblock.offset_size
  let lengths = self.superblock.length_size
  let start = address_to_offset(manager.address, self.superblock.base_address)
  let header_size = 18 + 7 * lengths + offsets
  let bits = read_uint_at(self.bytes, start + 12 + 4 * lengths, 2, true).to_int()
  let maximum = read_uint_at(
    self.bytes,
    start + 14 + 4 * lengths,
    lengths,
    true,
  )
  let old_address = read_uint_at(
    self.bytes,
    start + 14 + 5 * lengths,
    offsets,
    true,
  )
  let old_capacity = read_uint_at(
    self.bytes,
    start + 14 + 6 * lengths + offsets,
    lengths,
    true,
  )
  let sorted = sections.copy()
  sorted.sort_by(fn(left, right) {
    let size_order = left.size.compare(right.size)
    if size_order == 0 {
      left.address.compare(right.address)
    } else {
      size_order
    }
  })
  let encoded = @buffer.Buffer::Buffer()
  encoded.write_bytes(b"FSSE\x00")
  write_uint_width(encoded, manager.address, offsets)
  let count_width = space_encoding_width(sorted.length().to_uint64())
  let size_width = space_encoding_width(maximum)
  let mut index = 0
  let mut total = 0UL
  while index < sorted.length() {
    let mut end = index + 1
    while end < sorted.length() && sorted[end].size == sorted[index].size {
      end = end + 1
    }
    write_uint_width(encoded, (end - index).to_uint64(), count_width)
    write_uint_width(encoded, sorted[index].size, size_width)
    while index < end {
      let section = sorted[index]
      write_uint_width(encoded, section.address, (bits + 7) / 8)
      encoded.write_byte(b'\x00')
      guard total <= 0xFFFFFFFFFFFFFFFFUL - section.size else {
        raise InvalidField("free-space total overflow")
      }
      total = total + section.size
      index = index + 1
    }
  }
  if !sorted.is_empty() && (encoded.length() + 4).to_uint64() <= old_capacity {
    writer_write_zeros(encoded, old_capacity.to_int() - encoded.length() - 4)
  }
  encoded.write_uint_le(
    metadata_checksum(encoded.to_bytes(), 0, encoded.length()),
  )
  let image = if sorted.is_empty() { b"" } else { encoded.to_bytes() }
  let (address, capacity) = if image.is_empty() {
    (btree_v1_undefined_address(offsets), 0UL)
  } else if image.length().to_uint64() <= old_capacity {
    let offset = address_to_offset(old_address, self.superblock.base_address)
    output.blit_from_bytes(offset, image, 0, image.length())
    (old_address, old_capacity)
  } else {
    let offset = append_aligned_bytes_to_tail(
      tail,
      self.bytes.length(),
      image,
      "rewrite free-space sections",
    )
    (
      file_offset_to_stored_address(offset, self.superblock.base_address),
      image.length().to_uint64(),
    )
  }
  for
    (field, width, value) in [
      (6, lengths, total),
      (6 + lengths, lengths, sorted.length().to_uint64()),
      (6 + 2 * lengths, lengths, sorted.length().to_uint64()),
      (6 + 3 * lengths, lengths, 0UL),
      (14 + 5 * lengths, offsets, address),
      (14 + 5 * lengths + offsets, lengths, image.length().to_uint64()),
      (14 + 6 * lengths + offsets, lengths, capacity),
    ] {
    patch_little_endian_uint(
      output,
      start + field,
      width,
      value,
      "free-space manager field",
    )
  }
  patch_metadata_checksum_in_place(output, start, start + header_size - 4)
  let extension = self.object_header_at(
    self.superblock.superblock_extension_address,
  )
  let message = extension.first(0x17).unwrap()
  if message.data[0] == b'\x01' {
    let final_length = self.bytes.length().to_uint64() +
      tail.length().to_uint64()
    patch_little_endian_uint(
      output,
      message.data_offset + 5 + 2 * lengths,
      offsets,
      file_offset_to_stored_address(final_length, self.superblock.base_address),
      "file-space allocation boundary",
    )
    if message.checksum_offset >= 0 {
      patch_metadata_checksum_in_place(
        output,
        message.chunk_start,
        message.checksum_offset,
      )
    }
  }
}

///|
fn Hdf5File::reuse_contiguous_space(
  self : Hdf5File,
  payload : Bytes,
  output : FixedArray[Byte],
  tail : @buffer.Buffer,
  released : Hdf5FreeSection?,
  alignment : UInt64,
) -> UInt64? raise Hdf5Error {
  if payload.is_empty() {
    return None
  }
  let info = match self.file_space_info() {
    Some(info) if info.strategy == FsmAggregate && info.persistent => info
    _ => return None
  }
  let managers = self.free_space_managers()
  let address = info.manager_addresses[2]
  let mut found : Hdf5FreeSpaceManager? = None
  for manager in managers {
    if manager.address == address {
      found = Some(manager)
    }
  }
  let manager = match found {
    Some(manager) => manager
    None => return None
  }
  guard manager.ghost_sections == 0UL &&
    manager.sections.all(section => section.section_type == 0) else {
    raise UnsupportedFeature(
      "raw space reuse requires simple sections without ghosts",
    )
  }
  if released is Some(live) {
    guard !managers.any(item => {
      item.sections.any(section => {
        section.address < live.address + live.size &&
        live.address < section.address + section.size
      })
    }) else {
      raise InvalidField("live dataset storage overlaps recorded free space")
    }
  }
  let (destination, remaining) = match
    space_allocate(manager.sections, payload.length().to_uint64(), alignment) {
    Some(result) => result
    None => return None
  }
  if released is Some(section) {
    remaining.push(section)
  }
  let start = address_to_offset(manager.address, self.superblock.base_address)
  let maximum = read_uint_at(
    self.bytes,
    start + 14 + 4 * self.superblock.length_size,
    self.superblock.length_size,
    true,
  )
  let sections = space_coalesce(remaining, maximum).filter(section => {
    section.size >= info.threshold
  })
  self.write_space_sections(manager, sections, output, tail)
  let absolute = address_to_offset(destination, self.superblock.base_address)
  output.blit_from_bytes(absolute, payload, 0, payload.length())
  Some(absolute.to_uint64())
}

///|
/// Moves an allocated, internal contiguous payload into an existing persisted
/// raw-data free section and returns its previous storage to that manager.
/// Shapes, values, references, attributes and object addresses are preserved.
/// Requires a bytes-backed file with a persistent aggregate strategy and a
/// suitable free section. This does not shrink the file or reclaim old heaps.
pub fn Hdf5File::relocate_dataset_storage(
  self : Hdf5File,
  path : String,
  alignment~ : UInt64 = 8UL,
) -> Bytes raise Hdf5Error {
  guard self.source is None else {
    raise InvalidArgument("storage relocation requires a bytes-backed Hdf5File")
  }
  guard alignment > 0UL &&
    alignment <= 0x40000000UL &&
    (alignment & (alignment - 1UL)) == 0UL &&
    self.superblock.base_address % alignment == 0UL else {
    raise InvalidArgument("invalid storage relocation alignment")
  }
  let node = self.resolve(path)
  guard node.kind == Dataset else {
    raise InvalidArgument("storage relocation requires a dataset")
  }
  let header = self.object_header_at(node.address)
  let metadata = self.read_dataset_metadata(path)
  let absent = btree_v1_undefined_address(self.superblock.offset_size)
  guard metadata.layout.kind() == Contiguous &&
    metadata.layout.data_address != absent &&
    header.first(msg_external_data_files) is None &&
    header.first(msg_filter_pipeline) is None &&
    metadata.layout.data_size > 0UL &&
    metadata.layout.data_size <= 0x7FFFFFFFUL else {
    raise UnsupportedFeature(
      "storage relocation requires allocated internal contiguous data",
    )
  }
  let payload = self.space_read(
    metadata.layout.data_address,
    metadata.layout.data_size.to_int(),
  )
  let output = self.bytes.to_fixedarray()
  let tail = @buffer.Buffer::Buffer()
  let offset = match
    self.reuse_contiguous_space(
      payload,
      output,
      tail,
      Some({
        address: metadata.layout.data_address,
        size: metadata.layout.data_size,
        section_type: 0,
      }),
      alignment,
    ) {
    Some(offset) => offset
    None =>
      raise UnsupportedFeature(
        "no suitable persisted aggregate raw-data free section",
      )
  }
  patch_little_endian_uint(
    output,
    metadata.layout.data_address_offset,
    metadata.layout.data_address_width,
    file_offset_to_stored_address(offset, self.superblock.base_address),
    "relocated contiguous address",
  )
  let message = header.first(msg_data_layout).unwrap()
  if message.checksum_offset >= 0 {
    patch_metadata_checksum_in_place(
      output,
      message.chunk_start,
      message.checksum_offset,
    )
  }
  self.finish_appended_image(output, tail)
}
