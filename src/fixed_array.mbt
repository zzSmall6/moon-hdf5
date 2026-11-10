///|
fn read_fixed_array_chunks(
  bytes : Bytes,
  layout : DataLayout,
  dataspace : Dataspace,
  datatype : Datatype,
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
  filters : Array[Filter],
  output : FixedArray[Byte],
  undefined : UInt64,
  page_bits : Int32,
  selection : Array[UInt64]?,
  records~ : Array[RawChunkRecord]? = None,
) -> Unit raise Hdf5Error {
  guard page_bits > 0 && page_bits <= 31 else {
    raise InvalidField("invalid fixed array page bits")
  }
  let rank = dataspace.dimensions.length()
  guard dataspace.maximum_dimensions.length() == rank else {
    raise InvalidField("fixed array requires maximum dimensions")
  }
  let current_grid : Array[Int64] = []
  let maximum_grid : Array[Int64] = []
  for dimension in 0..<rank {
    let current = dataspace.dimensions[dimension]
    let maximum = dataspace.maximum_dimensions[dimension]
    let width = layout.chunk_dimensions[dimension].to_int64()
    guard current <= maximum && maximum <= 0x7FFFFFFFUL && width > 0 else {
      raise InvalidField("invalid fixed array dimensions")
    }
    current_grid.push((current.reinterpret_as_int64() + width - 1) / width)
    maximum_grid.push((maximum.reinterpret_as_int64() + width - 1) / width)
  }
  let maximum_total = product(maximum_grid)
  let chunk_size = chunk_byte_size(layout.chunk_dimensions, datatype.size)
  let header_offset = address_to_offset(layout.data_address, base_address)
  let header = Reader::at(bytes, header_offset)
  guard header.read_bytes(4) == b"FAHD" && header.read_u8() == b'\x00' else {
    raise InvalidField("invalid fixed array header")
  }
  let client = header.read_u8().to_int()
  let entry_size = header.read_u8().to_int()
  let size_width = entry_size - offset_size - 4
  guard header.read_u8().to_int() == page_bits &&
    (
      (client == 0 && filters.length() == 0 && entry_size == offset_size) ||
      (
        client == 1 &&
        filters.length() > 0 &&
        size_width >= 1 &&
        size_width <= 8
      )
    ) else {
    raise InvalidField("invalid fixed array entry encoding")
  }
  guard header.read_uint_le(length_size) ==
    maximum_total.reinterpret_as_uint64() else {
    raise InvalidField(
      "fixed array entry count does not match maximum dimensions",
    )
  }
  let block_address = header.read_uint_le(offset_size)
  verify_metadata_checksum(
    bytes,
    header_offset,
    header.position(),
    "fixed array header",
  )
  header.skip(4)
  if block_address == undefined {
    return
  }
  let block_offset = address_to_offset(block_address, base_address)
  let block = Reader::at(bytes, block_offset)
  guard block.read_bytes(4) == b"FADB" &&
    block.read_u8() == b'\x00' &&
    block.read_u8().to_int() == client &&
    block.read_uint_le(offset_size) == layout.data_address else {
    raise InvalidField("invalid fixed array data block")
  }
  let page_entries = 1L << page_bits
  let paged = maximum_total > page_entries
  let page_count = (maximum_total + page_entries - 1) / page_entries
  let bitmap_start = block.position()
  let bitmap_size = if paged { ((page_count + 7) / 8).to_int() } else { 0 }
  let bitmap_checksum_offset = bitmap_start + bitmap_size
  let bitmap = if paged {
    let bitmap = block.read_bytes(bitmap_size)
    verify_metadata_checksum(
      bytes, block_offset, bitmap_checksum_offset, "fixed array data block",
    )
    block.skip(4)
    bitmap
  } else {
    b""
  }
  let entries_start = block.position().to_int64()
  if !paged {
    verify_metadata_checksum(
      bytes,
      block_offset,
      (entries_start + maximum_total * entry_size.to_int64()).to_int(),
      "fixed array data block",
    )
  }
  let storage_size = maximum_total * entry_size.to_int64() +
    (if paged { page_count } else { 1L }) * 4
  guard storage_size <= bytes.length().to_int64() - entries_start else {
    raise InvalidField("fixed array entries extend beyond input")
  }
  if output.length() == 0 && records is None {
    return
  }
  let total = product(current_grid)
  for linear in 0..<total.to_int() {
    let coordinates : Array[Int64] = Array::make(rank, 0L)
    let offsets : Array[UInt64] = Array::make(rank, 0UL)
    let mut remaining = linear.to_int64()
    let mut dimension = rank
    while dimension > 0 {
      dimension = dimension - 1
      coordinates[dimension] = remaining % current_grid[dimension]
      remaining = remaining / current_grid[dimension]
      offsets[dimension] = (coordinates[dimension] *
      layout.chunk_dimensions[dimension].to_int64()).reinterpret_as_uint64()
    }
    let mut stored_index = 0L
    for dimension in 0..<rank {
      stored_index = stored_index * maximum_grid[dimension] +
        coordinates[dimension]
    }
    let page = stored_index / page_entries
    let page_elements = if !paged {
      maximum_total
    } else if page + 1 == page_count {
      maximum_total - page * page_entries
    } else {
      page_entries
    }
    let page_offset = if paged {
      entries_start + page * page_entries * entry_size.to_int64() + page * 4L
    } else {
      entries_start
    }
    let page_checksum_offset = page_offset +
      page_elements * entry_size.to_int64()
    let bitmap_index = page.to_int() / 8
    let bitmap_mask = (0x80 >> (page.to_int() % 8)).to_byte()
    let page_allocated = if paged {
      guard bitmap_index < bitmap.length() else {
        raise InvalidField("invalid fixed array page bitmap")
      }
      (bitmap[bitmap_index].to_int() & bitmap_mask.to_int()) != 0
    } else {
      true
    }
    if paged && page_allocated {
      verify_metadata_checksum(
        bytes,
        page_offset.to_int(),
        page_checksum_offset.to_int(),
        "fixed array data block page",
      )
    }
    let position = entries_start +
      stored_index * entry_size.to_int64() +
      (if paged { page * 4 } else { 0L })
    let stored_size_offset = position + offset_size.to_int64()
    let mask_offset = if client == 1 {
      position.to_int() + offset_size + size_width
    } else {
      -1
    }
    guard stored_size_offset >= 0L &&
      stored_size_offset <= 0x7FFFFFFFL &&
      page_offset >= 0L &&
      page_offset <= 0x7FFFFFFFL &&
      page_checksum_offset >= 0L &&
      page_checksum_offset <= 0x7FFFFFFFL &&
      bitmap_start >= 0 &&
      bitmap_checksum_offset >= 0 else {
      raise InvalidField("fixed array metadata offset is out of range")
    }
    if paged && !page_allocated {
      push_chunk_page_slot_record(
        records,
        offsets.copy(),
        undefined,
        chunk_size.to_uint64(),
        0U,
        position.to_int(),
        offset_size,
        if client == 1 {
          stored_size_offset.to_int()
        } else {
          -1
        },
        if client == 1 {
          size_width
        } else {
          0
        },
        mask_offset,
        page_offset.to_int(),
        page_checksum_offset.to_int(),
        Some({
          start: page_offset.to_int(),
          entries: page_elements.to_int(),
          entry_size,
          checksum_offset: page_checksum_offset.to_int(),
          bitmap_offset: bitmap_start + bitmap_index,
          bitmap_mask,
          bitmap_metadata_start: block_offset,
          bitmap_checksum_offset,
          address_width: offset_size,
          stored_size_width: if client == 1 {
            size_width
          } else {
            0
          },
        }),
      )
      continue
    }
    let entry = Reader::at(bytes, position.to_int())
    let address = entry.read_uint_le(offset_size)
    let mut stored_size = chunk_size.to_uint64()
    let mut mask = 0U
    if client == 1 {
      stored_size = entry.read_var_uint_le(size_width)
      mask = entry.read_u32_le()
    }
    let metadata_start = if paged {
      page_offset
    } else {
      block_offset.to_int64()
    }
    let metadata_checksum_offset = if paged {
      page_checksum_offset
    } else {
      entries_start + maximum_total * entry_size.to_int64()
    }
    guard metadata_start >= 0L &&
      metadata_start <= 0x7FFFFFFFL &&
      metadata_checksum_offset >= 0L &&
      metadata_checksum_offset <= 0x7FFFFFFFL else {
      raise InvalidField("fixed array metadata offset is out of range")
    }
    push_chunk_slot_record(
      records,
      offsets.copy(),
      address,
      stored_size,
      mask,
      position.to_int(),
      offset_size,
      if client == 1 {
        stored_size_offset.to_int()
      } else {
        -1
      },
      if client == 1 {
        size_width
      } else {
        0
      },
      mask_offset,
      metadata_start.to_int(),
      metadata_checksum_offset.to_int(),
    )
    if address == undefined {
      continue
    }
    if !chunk_contains_selection(
        offsets,
        layout.chunk_dimensions,
        dataspace.dimensions,
        selection,
      ) {
      continue
    }
    let decoded = read_filtered_chunk(
      bytes,
      address,
      base_address,
      stored_size,
      filters,
      mask,
      datatype.size,
      chunk_size,
    )
    copy_chunk(
      decoded,
      decoded.length(),
      offsets,
      layout.chunk_dimensions,
      dataspace.dimensions,
      datatype.size,
      output,
      selection,
    )
  }
}

///|
/// A missing fixed-array data block prepared at a new file offset. The image
/// contains undefined entries and valid checksums for every unallocated page.
priv struct MissingFixedArrayStorage {
  image : FixedArray[Byte]
  records : Array[RawChunkRecord]
  block_start : Int32
  block_address_offset : Int32
  block_address_width : Int32
  header_checksum_offset : Int32
}

///|
fn initialize_fixed_array_entry(
  image : FixedArray[Byte],
  position : Int32,
  offset_size : Int32,
  client : Int32,
  size_width : Int32,
  undefined : UInt64,
) -> Unit raise Hdf5Error {
  patch_little_endian_uint(
    image, position, offset_size, undefined, "fixed array chunk address",
  )
  if client == 1 {
    patch_little_endian_uint(
      image,
      position + offset_size,
      size_width,
      0UL,
      "fixed array chunk stored size",
    )
    patch_little_endian_uint(
      image,
      position + offset_size + size_width,
      4,
      0UL,
      "fixed array chunk filter mask",
    )
  }
}

///|
/// Builds an empty FADB for a fixed-array header whose data-block address is
/// undefined. The returned records address every current chunk slot so patching
/// can reuse the same page-initialization and tail-allocation path as an
/// existing data block.
fn prepare_missing_fixed_array_storage(
  bytes : Bytes,
  layout : DataLayout,
  dataspace : Dataspace,
  datatype : Datatype,
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
  filters : Array[Filter],
  undefined : UInt64,
  page_bits : Int32,
  block_start : Int32,
) -> MissingFixedArrayStorage raise Hdf5Error {
  guard page_bits > 0 && page_bits <= 31 else {
    raise InvalidField("invalid fixed array page bits")
  }
  let rank = dataspace.dimensions.length()
  guard dataspace.maximum_dimensions.length() == rank else {
    raise InvalidField("fixed array requires maximum dimensions")
  }
  let current_grid : Array[Int64] = []
  let maximum_grid : Array[Int64] = []
  for dimension in 0..<rank {
    let current = dataspace.dimensions[dimension]
    let maximum = dataspace.maximum_dimensions[dimension]
    let width = layout.chunk_dimensions[dimension].to_int64()
    guard current <= maximum && maximum <= 0x7FFFFFFFUL && width > 0 else {
      raise InvalidField("invalid fixed array dimensions")
    }
    current_grid.push((current.reinterpret_as_int64() + width - 1) / width)
    maximum_grid.push((maximum.reinterpret_as_int64() + width - 1) / width)
  }
  let maximum_total = product(maximum_grid)
  let chunk_size = chunk_byte_size(layout.chunk_dimensions, datatype.size)
  let header_offset = address_to_offset(layout.data_address, base_address)
  let header = Reader::at(bytes, header_offset)
  guard header.read_bytes(4) == b"FAHD" && header.read_u8() == b'\x00' else {
    raise InvalidField("invalid fixed array header")
  }
  let client = header.read_u8().to_int()
  let entry_size = header.read_u8().to_int()
  let size_width = entry_size - offset_size - 4
  guard header.read_u8().to_int() == page_bits &&
    (
      (client == 0 && filters.length() == 0 && entry_size == offset_size) ||
      (
        client == 1 &&
        filters.length() > 0 &&
        size_width >= 1 &&
        size_width <= 8
      )
    ) else {
    raise InvalidField("invalid fixed array entry encoding")
  }
  guard header.read_uint_le(length_size) ==
    maximum_total.reinterpret_as_uint64() else {
    raise InvalidField(
      "fixed array entry count does not match maximum dimensions",
    )
  }
  let block_address_offset = header.position()
  let block_address = header.read_uint_le(offset_size)
  let header_checksum_offset = header.position()
  verify_metadata_checksum(
    bytes, header_offset, header_checksum_offset, "fixed array header",
  )
  guard block_address == undefined else {
    raise InvalidField("fixed array data block is already allocated")
  }
  guard header_checksum_offset.to_int64() + 4L <= bytes.length().to_int64() &&
    block_start >= 0 else {
    raise UnexpectedEof(header_checksum_offset.to_int64())
  }
  let page_entries = 1L << page_bits
  let paged = maximum_total > page_entries
  let page_count = (maximum_total + page_entries - 1) / page_entries
  let bitmap_size = if paged { ((page_count + 7) / 8).to_int() } else { 0 }
  let block_header_size = 6 + offset_size
  let bitmap_checksum_offset = block_header_size + bitmap_size
  let entries_start = block_header_size +
    (if paged { bitmap_size + 4 } else { 0 })
  let entry_bytes = maximum_total * entry_size.to_int64()
  let block_size = entries_start.to_int64() +
    entry_bytes +
    (if paged { page_count } else { 1L }) * 4L
  guard entry_bytes >= 0L &&
    block_size <= 0x7FFFFFFFL &&
    block_start.to_int64() + block_size <= 0x7FFFFFFFL else {
    raise InvalidField("fixed array data block is too large")
  }
  let image = FixedArray::make(block_size.to_int(), b'\x00')
  image[0] = b'F'
  image[1] = b'A'
  image[2] = b'D'
  image[3] = b'B'
  image[4] = b'\x00'
  image[5] = client.to_byte()
  patch_little_endian_uint(
    image,
    6,
    offset_size,
    layout.data_address,
    "fixed array data block header",
  )
  if paged {
    patch_metadata_checksum_in_place(image, 0, bitmap_checksum_offset)
    let mut page_start = entries_start
    for page in 0..<page_count.to_int() {
      let page_elements = if page + 1 == page_count.to_int() {
        maximum_total - page.to_int64() * page_entries
      } else {
        page_entries
      }
      for entry in 0..<page_elements.to_int() {
        initialize_fixed_array_entry(
          image,
          page_start + entry * entry_size,
          offset_size,
          client,
          size_width,
          undefined,
        )
      }
      patch_metadata_checksum_in_place(
        image,
        page_start,
        page_start + page_elements.to_int() * entry_size,
      )
      page_start = page_start + page_elements.to_int() * entry_size + 4
    }
  } else {
    for entry in 0..<maximum_total.to_int() {
      initialize_fixed_array_entry(
        image,
        entries_start + entry * entry_size,
        offset_size,
        client,
        size_width,
        undefined,
      )
    }
    patch_metadata_checksum_in_place(
      image,
      0,
      entries_start + maximum_total.to_int() * entry_size,
    )
  }
  let records : Array[RawChunkRecord] = []
  let total = product(current_grid)
  for linear in 0..<total.to_int() {
    let coordinates : Array[Int64] = Array::make(rank, 0L)
    let offsets : Array[UInt64] = Array::make(rank, 0UL)
    let mut remaining = linear.to_int64()
    let mut dimension = rank
    while dimension > 0 {
      dimension = dimension - 1
      coordinates[dimension] = remaining % current_grid[dimension]
      remaining = remaining / current_grid[dimension]
      offsets[dimension] = (coordinates[dimension] *
      layout.chunk_dimensions[dimension].to_int64()).reinterpret_as_uint64()
    }
    let mut stored_index = 0L
    for dimension in 0..<rank {
      stored_index = stored_index * maximum_grid[dimension] +
        coordinates[dimension]
    }
    let page = stored_index / page_entries
    let page_elements = if !paged {
      maximum_total
    } else if page + 1 == page_count {
      maximum_total - page * page_entries
    } else {
      page_entries
    }
    let relative_position = entries_start.to_int64() +
      stored_index * entry_size.to_int64() +
      (if paged { page * 4L } else { 0L })
    let position = block_start.to_int64() + relative_position
    let stored_size_offset = position + offset_size.to_int64()
    let mask_offset = if client == 1 {
      position + offset_size.to_int64() + size_width.to_int64()
    } else {
      -1L
    }
    let relative_page_start = if paged {
      entries_start.to_int64() +
      page * page_entries * entry_size.to_int64() +
      page * 4L
    } else {
      entries_start.to_int64()
    }
    let page_start = block_start.to_int64() + relative_page_start
    let page_checksum_offset = page_start +
      page_elements * entry_size.to_int64()
    guard position >= 0L &&
      position <= 0x7FFFFFFFL &&
      stored_size_offset <= 0x7FFFFFFFL &&
      mask_offset <= 0x7FFFFFFFL &&
      page_start <= 0x7FFFFFFFL &&
      page_checksum_offset <= 0x7FFFFFFFL else {
      raise InvalidField("fixed array metadata offset is out of range")
    }
    let bitmap_index = page.to_int() / 8
    let bitmap_mask = (0x80 >> (page.to_int() % 8)).to_byte()
    let page_record : RawChunkPage? = if paged {
      Some({
        start: page_start.to_int(),
        entries: page_elements.to_int(),
        entry_size,
        checksum_offset: page_checksum_offset.to_int(),
        bitmap_offset: block_start + block_header_size + bitmap_index,
        bitmap_mask,
        bitmap_metadata_start: block_start,
        bitmap_checksum_offset: block_start + bitmap_checksum_offset,
        address_width: offset_size,
        stored_size_width: if client == 1 {
          size_width
        } else {
          0
        },
      })
    } else {
      None
    }
    push_chunk_page_slot_record(
      Some(records),
      offsets.copy(),
      undefined,
      chunk_size.to_uint64(),
      0U,
      position.to_int(),
      offset_size,
      if client == 1 {
        stored_size_offset.to_int()
      } else {
        -1
      },
      if client == 1 {
        size_width
      } else {
        0
      },
      if client == 1 {
        mask_offset.to_int()
      } else {
        -1
      },
      if paged {
        page_start.to_int()
      } else {
        block_start
      },
      if paged {
        page_checksum_offset.to_int()
      } else {
        (block_start.to_int64() +
        entries_start.to_int64() +
        maximum_total * entry_size.to_int64()).to_int()
      },
      page_record,
    )
  }
  {
    image,
    records,
    block_start,
    block_address_offset,
    block_address_width: offset_size,
    header_checksum_offset,
  }
}

///|
fn read_source_fixed_array_chunks(
  source : &Hdf5Source,
  layout : DataLayout,
  dataspace : Dataspace,
  datatype : Datatype,
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
  filters : Array[Filter],
  output : FixedArray[Byte],
  undefined : UInt64,
  page_bits : Int32,
  selection : Array[UInt64]?,
) -> Unit raise Hdf5Error {
  guard page_bits > 0 && page_bits <= 31 else {
    raise InvalidField("invalid fixed array page bits")
  }
  let rank = dataspace.dimensions.length()
  guard dataspace.maximum_dimensions.length() == rank else {
    raise InvalidField("fixed array requires maximum dimensions")
  }
  let current_grid : Array[Int64] = []
  let maximum_grid : Array[Int64] = []
  for dimension in 0..<rank {
    let current = dataspace.dimensions[dimension]
    let maximum = dataspace.maximum_dimensions[dimension]
    let width = layout.chunk_dimensions[dimension].to_int64()
    guard current <= maximum && maximum <= 0x7FFFFFFFUL && width > 0 else {
      raise InvalidField("invalid fixed array dimensions")
    }
    current_grid.push((current.reinterpret_as_int64() + width - 1) / width)
    maximum_grid.push((maximum.reinterpret_as_int64() + width - 1) / width)
  }
  let maximum_total = product(maximum_grid)
  let chunk_size = chunk_byte_size(layout.chunk_dimensions, datatype.size)
  let header_offset = source_address_to_offset(
    layout.data_address,
    base_address,
  )
  let header_content = 8 + length_size + offset_size
  let header = Reader::new(
    source_read_checksummed(
      source, header_offset, header_content, "fixed array header",
    ),
  )
  guard header.read_bytes(4) == b"FAHD" && header.read_u8() == b'\x00' else {
    raise InvalidField("invalid fixed array header")
  }
  let client = header.read_u8().to_int()
  let entry_size = header.read_u8().to_int()
  let size_width = entry_size - offset_size - 4
  guard header.read_u8().to_int() == page_bits &&
    (
      (client == 0 && filters.length() == 0 && entry_size == offset_size) ||
      (
        client == 1 &&
        filters.length() > 0 &&
        size_width >= 1 &&
        size_width <= 8
      )
    ) else {
    raise InvalidField("invalid fixed array entry encoding")
  }
  guard header.read_uint_le(length_size) ==
    maximum_total.reinterpret_as_uint64() else {
    raise InvalidField(
      "fixed array entry count does not match maximum dimensions",
    )
  }
  let block_address = header.read_uint_le(offset_size)
  if block_address == undefined || output.length() == 0 {
    return
  }
  let block_offset = source_address_to_offset(block_address, base_address)
  let block_prefix_size = 6 + offset_size
  let prefix = Reader::new(source.read_at(block_offset, block_prefix_size))
  guard prefix.read_bytes(4) == b"FADB" &&
    prefix.read_u8() == b'\x00' &&
    prefix.read_u8().to_int() == client &&
    prefix.read_uint_le(offset_size) == layout.data_address else {
    raise InvalidField("invalid fixed array data block")
  }
  let page_entries = 1L << page_bits
  let paged = maximum_total > page_entries
  let page_count = (maximum_total + page_entries - 1) / page_entries
  let bitmap_size = if paged { ((page_count + 7) / 8).to_int() } else { 0 }
  let block_header = if paged {
    source_read_checksummed(
      source,
      block_offset,
      block_prefix_size + bitmap_size,
      "fixed array data block",
    )
  } else {
    b""
  }
  let bitmap = if paged {
    block_header
    .get_view(start=block_prefix_size, end=block_prefix_size + bitmap_size)
    .unwrap()
    .to_owned()
  } else {
    b""
  }
  let entries_start = block_offset +
    block_prefix_size.to_uint64() +
    (if paged { bitmap_size + 4 } else { 0 }).to_uint64()
  let unpaged = if paged {
    b""
  } else {
    guard maximum_total <= 0x7FFFFFFFL / entry_size.to_int64() else {
      raise InvalidField("fixed array entries extend beyond input")
    }
    source_read_checksummed(
      source,
      block_offset,
      block_prefix_size + maximum_total.to_int() * entry_size,
      "fixed array data block",
    )
  }
  let total = product(current_grid)
  let mut cached_page = -1L
  let mut cached_entries = b""
  for linear in 0..<total.to_int() {
    let coordinates : Array[Int64] = Array::make(rank, 0L)
    let offsets : Array[UInt64] = Array::make(rank, 0UL)
    let mut remaining = linear.to_int64()
    let mut dimension = rank
    while dimension > 0 {
      dimension = dimension - 1
      coordinates[dimension] = remaining % current_grid[dimension]
      remaining = remaining / current_grid[dimension]
      offsets[dimension] = (coordinates[dimension] *
      layout.chunk_dimensions[dimension].to_int64()).reinterpret_as_uint64()
    }
    let mut stored_index = 0L
    for dimension in 0..<rank {
      stored_index = stored_index * maximum_grid[dimension] +
        coordinates[dimension]
    }
    let page = stored_index / page_entries
    if paged &&
      (bitmap[page.to_int() / 8].to_int() & (0x80 >> (page.to_int() % 8))) == 0 {
      continue
    }
    if !chunk_contains_selection(
        offsets,
        layout.chunk_dimensions,
        dataspace.dimensions,
        selection,
      ) {
      continue
    }
    let entry = if paged {
      if cached_page != page {
        let page_elements = if page + 1 == page_count {
          maximum_total - page * page_entries
        } else {
          page_entries
        }
        guard page_elements <= 0x7FFFFFFFL / entry_size.to_int64() else {
          raise InvalidField("fixed array page is too large")
        }
        let page_offset = entries_start +
          page.reinterpret_as_uint64() *
          (page_entries * entry_size.to_int64() + 4L).reinterpret_as_uint64()
        cached_entries = source_read_checksummed(
          source,
          page_offset,
          page_elements.to_int() * entry_size,
          "fixed array data block page",
        )
        cached_page = page
      }
      Reader::at(
        cached_entries,
        ((stored_index - page * page_entries) * entry_size.to_int64()).to_int(),
      )
    } else {
      Reader::at(
        unpaged,
        block_prefix_size + stored_index.to_int() * entry_size,
      )
    }
    let address = entry.read_uint_le(offset_size)
    if address == undefined {
      continue
    }
    let stored_size = if client == 1 {
      entry.read_var_uint_le(size_width)
    } else {
      chunk_size.to_uint64()
    }
    let mask = if client == 1 { entry.read_u32_le() } else { 0U }
    let decoded = read_source_filtered_chunk(
      source,
      address,
      base_address,
      stored_size,
      filters,
      mask,
      datatype.size,
      chunk_size,
    )
    copy_chunk(
      decoded,
      decoded.length(),
      offsets,
      layout.chunk_dimensions,
      dataspace.dimensions,
      datatype.size,
      output,
      selection,
    )
  }
}
