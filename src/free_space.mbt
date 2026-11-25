///|
/// File-space allocation strategy recorded in the superblock extension.
#derive(Eq, @debug.Debug)
pub(all) enum Hdf5FileSpaceStrategy {
  FsmAggregate
  Paged
  Aggregate
  Unmanaged
}

///|
pub extend Hdf5FileSpaceStrategy with Eq::{not_equal, equal}

///|
pub extend Hdf5FileSpaceStrategy with @debug.Debug::{to_repr}

///|
/// Explicit file-space settings. Manager slots follow HDF5 memory-type order
/// (six legacy slots or twelve version-1 slots); undefined addresses use the
/// file's address width. Absent settings use the HDF5 defaults.
#derive(@debug.Debug)
pub(all) struct Hdf5FileSpaceInfo {
  strategy : Hdf5FileSpaceStrategy
  persistent : Bool
  threshold : UInt64
  page_size : UInt64
  manager_addresses : Array[UInt64]
}

///|
pub extend Hdf5FileSpaceInfo with @debug.Debug::{to_repr}

///|
/// A persisted free region; addresses are relative to the file's base address.
#derive(Eq, @debug.Debug)
pub(all) struct Hdf5FreeSection {
  address : UInt64
  size : UInt64
  section_type : Int32
}

///|
pub extend Hdf5FreeSection with Eq::{not_equal, equal}

///|
pub extend Hdf5FreeSection with @debug.Debug::{to_repr}

///|
/// One file-space manager and its validated serialized sections.
#derive(@debug.Debug)
pub(all) struct Hdf5FreeSpaceManager {
  address : UInt64
  total_space : UInt64
  total_sections : UInt64
  ghost_sections : UInt64
  sections : Array[Hdf5FreeSection]
}

///|
pub extend Hdf5FreeSpaceManager with @debug.Debug::{to_repr}

///|
fn Hdf5File::space_read(
  self : Hdf5File,
  address : UInt64,
  length : Int32,
) -> Bytes raise Hdf5Error {
  let absolute = source_address_to_offset(address, self.superblock.base_address)
  guard absolute <= self.superblock.end_of_file_address &&
    length >= 0 &&
    length.to_uint64() <= self.superblock.end_of_file_address - absolute else {
    raise InvalidField("free-space metadata lies outside the file")
  }
  match self.source {
    Some(source) => source.read_at(absolute, length)
    None => Hdf5Source::read_at(self.bytes, absolute, length)
  }
}

///|
/// Reads explicit file-space settings, or None when no settings are stored.
pub fn Hdf5File::file_space_info(
  self : Hdf5File,
) -> Hdf5FileSpaceInfo? raise Hdf5Error {
  let superblock = self.superblock
  let absent = 0xFFFFFFFFFFFFFFFFUL >> ((8 - superblock.offset_size) * 8)
  if superblock.version < 2 || superblock.superblock_extension_address == absent {
    return None
  }
  let header = self.object_header_at(superblock.superblock_extension_address)
  let messages = header.find(0x17)
  guard messages.length() <= 1 else {
    raise InvalidField("duplicate file-space information message")
  }
  if messages.is_empty() {
    return None
  }
  let reader = Reader::new(messages[0].data)
  let version = reader.read_u8().to_int()
  guard version == 0 || version == 1 else {
    raise UnsupportedFeature("file-space information version")
  }
  let encoded_strategy = reader.read_u8().to_int()
  let (strategy, persistent) = if version == 0 {
    match encoded_strategy {
      1 => (FsmAggregate, true)
      2 => (FsmAggregate, false)
      3 => (Aggregate, false)
      4 => (Unmanaged, false)
      _ => raise InvalidField("invalid legacy file-space strategy")
    }
  } else {
    let strategy = match encoded_strategy {
      0 => FsmAggregate
      1 => Paged
      2 => Aggregate
      3 => Unmanaged
      _ => raise InvalidField("invalid file-space strategy")
    }
    let persist = reader.read_u8().to_int()
    guard persist <= 1 else {
      raise InvalidField("invalid file-space persistence flag")
    }
    (strategy, persist == 1)
  }
  guard !persistent || strategy == FsmAggregate || strategy == Paged else {
    raise InvalidField("file-space strategy cannot persist managers")
  }
  let threshold = reader.read_uint_le(superblock.length_size)
  let page_size = if version == 1 {
    let size = reader.read_uint_le(superblock.length_size)
    let _page_end_threshold = reader.read_u16_le()
    let _previous_eoa = reader.read_uint_le(superblock.offset_size)
    guard size >= 512UL && (size & (size - 1UL)) == 0UL else {
      raise InvalidField("invalid file-space page size")
    }
    size
  } else {
    4096UL
  }
  let manager_addresses : Array[UInt64] = []
  if persistent {
    let count = if version == 0 { 6 } else { 12 }
    for _index in 0..<count {
      manager_addresses.push(reader.read_uint_le(superblock.offset_size))
    }
  }
  Some({ strategy, persistent, threshold, page_size, manager_addresses, })
}

///|
fn space_encoding_width(value : UInt64) -> Int32 {
  let mut width = 1
  let mut remaining = value >> 8
  while remaining != 0UL {
    width = width + 1
    remaining = remaining >> 8
  }
  width
}

///|
fn Hdf5File::read_free_space_manager(
  self : Hdf5File,
  address : UInt64,
) -> Hdf5FreeSpaceManager raise Hdf5Error {
  let superblock = self.superblock
  let header_size = 18 + 7 * superblock.length_size + superblock.offset_size
  let bytes = self.space_read(address, header_size)
  verify_metadata_checksum(bytes, 0, header_size - 4, "free-space manager")
  let reader = Reader::new(bytes)
  guard reader.read_bytes(4) == b"FSHD" else {
    raise InvalidField("invalid free-space manager signature")
  }
  guard reader.read_u8() == b'\x00' else {
    raise UnsupportedFeature("free-space manager version")
  }
  guard reader.read_u8() == b'\x01' else {
    raise UnsupportedFeature("free-space manager is not a file-space client")
  }
  let total_space = reader.read_uint_le(superblock.length_size)
  let total_sections = reader.read_uint_le(superblock.length_size)
  let serialized = reader.read_uint_le(superblock.length_size)
  let ghost_sections = reader.read_uint_le(superblock.length_size)
  let classes = reader.read_u16_le().to_int()
  let _shrink = reader.read_u16_le()
  let _expand = reader.read_u16_le()
  let address_bits = reader.read_u16_le().to_int()
  let maximum_size = reader.read_uint_le(superblock.length_size)
  let section_address = reader.read_uint_le(superblock.offset_size)
  let section_size = reader.read_uint_le(superblock.length_size)
  let allocated_size = reader.read_uint_le(superblock.length_size)
  guard classes >= 1 &&
    classes <= 3 &&
    address_bits >= 1 &&
    address_bits <= 64 &&
    maximum_size > 0UL &&
    serialized <= total_sections &&
    ghost_sections == total_sections - serialized &&
    section_size <= allocated_size else {
    raise InvalidField("invalid free-space manager dimensions or counts")
  }
  let section_absolute = if allocated_size > 0UL {
    source_address_to_offset(section_address, superblock.base_address)
  } else {
    0UL
  }
  guard allocated_size == 0UL ||
    (
      section_absolute <= superblock.end_of_file_address &&
      allocated_size <= superblock.end_of_file_address - section_absolute
    ) else {
    raise InvalidField("free-space section allocation lies outside the file")
  }
  let sections : Array[Hdf5FreeSection] = []
  if serialized > 0UL {
    guard section_size >= (9 + superblock.offset_size).to_uint64() &&
      section_size <= 0x7FFFFFFFUL else {
      raise InvalidField("invalid free-space section image size")
    }
    let image = self.space_read(section_address, section_size.to_int())
    verify_metadata_checksum(
      image,
      0,
      image.length() - 4,
      "free-space sections",
    )
    let cursor = Reader::new(image).sub(0, image.length() - 4)
    guard cursor.read_bytes(4) == b"FSSE" else {
      raise InvalidField("invalid free-space sections signature")
    }
    guard cursor.read_u8() == b'\x00' else {
      raise UnsupportedFeature("free-space section version")
    }
    guard cursor.read_uint_le(superblock.offset_size) == address else {
      raise InvalidField("free-space section owner mismatch")
    }
    let count_width = space_encoding_width(serialized)
    let size_width = space_encoding_width(maximum_size)
    let offset_width = (address_bits + 7) / 8
    let mut seen = 0UL
    let mut sum = 0UL
    let mut previous_size = 0UL
    while seen < serialized {
      let count = cursor.read_var_uint_le(count_width)
      let size = cursor.read_var_uint_le(size_width)
      guard count > 0UL &&
        count <= serialized - seen &&
        size > previous_size &&
        size <= maximum_size &&
        count <= cursor.remaining().to_uint64() / (offset_width + 1).to_uint64() else {
        raise InvalidField("invalid free-space section size group")
      }
      previous_size = size
      for _index in 0..<count.to_int() {
        let start = cursor.read_var_uint_le(offset_width)
        let section_type = cursor.read_u8().to_int()
        guard section_type < classes else {
          raise InvalidField("invalid free-space section class")
        }
        let absolute = source_address_to_offset(start, superblock.base_address)
        guard absolute >= superblock.superblock_end.to_uint64() &&
          absolute <= superblock.end_of_file_address &&
          size <= superblock.end_of_file_address - absolute &&
          sum <= 0xFFFFFFFFFFFFFFFFUL - size else {
          raise InvalidField("free-space section lies outside the file")
        }
        sum = sum + size
        sections.push({ address: start, size, section_type, })
      }
      seen = seen + count
    }
    guard (ghost_sections == 0UL && sum == total_space) ||
      (ghost_sections > 0UL && sum <= total_space) else {
      raise InvalidField("free-space section total mismatch")
    }
  } else {
    guard ghost_sections > 0UL || total_space == 0UL else {
      raise InvalidField("empty free-space manager has nonzero space")
    }
  }
  sections.sort_by(fn(left, right) { left.address.compare(right.address) })
  let mut end = 0UL
  for section in sections {
    guard section.address >= end else {
      raise InvalidField("overlapping free-space sections")
    }
    end = section.address + section.size
    guard !(section.address < address + header_size.to_uint64() && address < end) &&
      !(allocated_size > 0UL &&
      section.address < section_address + allocated_size &&
      section_address < end) else {
      raise InvalidField("free-space section overlaps its manager metadata")
    }
  }
  { address, total_space, total_sections, ghost_sections, sections, }
}

///|
/// Reads persisted file-space managers. This does not discover untracked holes
/// or include transient aggregator space, and does not change the file.
pub fn Hdf5File::free_space_managers(
  self : Hdf5File,
) -> Array[Hdf5FreeSpaceManager] raise Hdf5Error {
  let available = match self.source {
    Some(source) => source.length()
    None => self.bytes.length().to_uint64()
  }
  guard self.superblock.end_of_file_address <= available else {
    raise UnexpectedEof(available.reinterpret_as_int64())
  }
  let result : Array[Hdf5FreeSpaceManager] = []
  let info = match self.file_space_info() {
    Some(info) => info
    None => return result
  }
  let absent = 0xFFFFFFFFFFFFFFFFUL >> ((8 - self.superblock.offset_size) * 8)
  for address in info.manager_addresses {
    if address != absent && !result.any(manager => manager.address == address) {
      result.push(self.read_free_space_manager(address))
    }
  }
  let all : Array[Hdf5FreeSection] = []
  for manager in result {
    for section in manager.sections {
      all.push(section)
    }
  }
  all.sort_by(fn(left, right) { left.address.compare(right.address) })
  let mut end = 0UL
  for section in all {
    guard section.address >= end else {
      raise InvalidField("overlapping sections across free-space managers")
    }
    end = section.address + section.size
  }
  result
}
