///|
/// Reserves raw-data space in new files and records it in a persistent
/// aggregate free-space manager. Zero disables the option (the default).
/// Reserved bytes are zero-filled by the sink. Metadata allocation and paged
/// strategies are not enabled by this option.
/// Shared-message output must be disabled when a reserve is enabled.
pub fn Hdf5Writer::set_file_space_reserve(
  self : Hdf5Writer,
  bytes : Int32,
) -> Unit raise Hdf5Error {
  guard bytes >= 0 && bytes <= 0x7FFFFFFF - 512 else {
    raise InvalidArgument("raw file-space reserve is out of range")
  }
  self.raw_space_reserve = bytes
}

///|
fn encode_reserved_space_sections(
  manager_address : UInt64,
  address : UInt64,
  size : UInt64,
) -> Bytes {
  let buffer = @buffer.Buffer::Buffer()
  buffer.write_bytes(b"FSSE\x00")
  buffer.write_uint64_le(manager_address)
  buffer.write_byte(b'\x01')
  buffer.write_uint64_le(size)
  buffer.write_uint64_le(address)
  buffer.write_byte(b'\x00')
  buffer.write_uint_le(metadata_checksum(buffer.to_bytes(), 0, buffer.length()))
  buffer.to_bytes()
}

///|
fn encode_reserved_space_header(
  size : UInt64,
  section_address : UInt64,
  section_size : UInt64,
) -> Bytes {
  let buffer = @buffer.Buffer::Buffer()
  buffer.write_bytes(b"FSHD\x00\x01")
  for value in [size, 1UL, 1UL, 0UL] {
    buffer.write_uint64_le(value)
  }
  for value in [3, 80, 120, 63] {
    buffer.write_uint16_le(value.to_uint16())
  }
  buffer.write_uint64_le(0x7FFFFFFFFFFFFFFFUL)
  buffer.write_uint64_le(section_address)
  buffer.write_uint64_le(section_size)
  buffer.write_uint64_le(section_size)
  buffer.write_uint_le(metadata_checksum(buffer.to_bytes(), 0, buffer.length()))
  buffer.to_bytes()
}

///|
fn encode_reserved_space_extension(
  address : UInt64,
  manager_address : UInt64,
  allocation_end : UInt64,
) -> Bytes raise Hdf5Error {
  let settings = @buffer.Buffer::Buffer()
  settings.write_bytes(b"\x01\x00\x01")
  settings.write_uint64_le(1UL)
  settings.write_uint64_le(4096UL)
  settings.write_uint16_le(10)
  settings.write_uint64_le(allocation_end)
  for index in 0..<12 {
    settings.write_uint64_le(
      if index == 2 {
        manager_address
      } else {
        undefined_address
      },
    )
  }
  let payload = @buffer.Buffer::Buffer()
  write_header_message(payload, 0x17, settings.to_bytes())
  encode_object_header(payload.to_bytes(), address)
}

///|
fn Hdf5Writer::write_reserved_file_space(
  self : Hdf5Writer,
  sink : &Hdf5Sink,
  start : Int32,
) -> (UInt64, Int32) raise Hdf5Error {
  let reserve_end = start.to_int64() + self.raw_space_reserve.to_int64()
  guard reserve_end <= 0x7FFFFFFFL - 512L - self.user_block_size.to_int64() else {
    raise UnsupportedFeature("file is too large to reserve raw storage")
  }
  let manager = writer_align(reserve_end.to_int())
  let sections = writer_align(manager + 82)
  let section_image = encode_reserved_space_sections(
    manager.to_uint64(),
    start.to_uint64(),
    self.raw_space_reserve.to_uint64(),
  )
  let extension = writer_align(sections + section_image.length())
  let extension_image = encode_reserved_space_extension(
    extension.to_uint64(),
    manager.to_uint64(),
    reserve_end.reinterpret_as_uint64(),
  )
  let end = writer_align(extension + extension_image.length())
  writer_sink_write_at(
    sink,
    self.user_block_size,
    manager.to_uint64(),
    encode_reserved_space_header(
      self.raw_space_reserve.to_uint64(),
      sections.to_uint64(),
      section_image.length().to_uint64(),
    ),
  )
  writer_sink_write_at(
    sink,
    self.user_block_size,
    sections.to_uint64(),
    section_image,
  )
  writer_sink_write_at(
    sink,
    self.user_block_size,
    extension.to_uint64(),
    extension_image,
  )
  (extension.to_uint64(), end)
}
