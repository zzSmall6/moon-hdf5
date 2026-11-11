///|
fn array_log2(value : Int32) -> Int32 {
  let mut bits = 0
  let mut remaining = value
  while remaining > 1 {
    remaining = remaining >> 1
    bits = bits + 1
  }
  bits
}

///|
fn validate_extensible_config(
  config : ExtensibleArrayConfig,
) -> Unit raise Hdf5Error {
  guard config.max_bits > 0 &&
    config.max_bits <= 64 &&
    config.index_elements > 0 &&
    config.index_elements <= 255 &&
    config.min_elements > 0 &&
    config.min_elements <= 128 &&
    (config.min_elements & (config.min_elements - 1)) == 0 &&
    config.min_pointers >= 2 &&
    config.min_pointers <= 128 &&
    (config.min_pointers & (config.min_pointers - 1)) == 0 &&
    config.page_bits > 0 &&
    config.page_bits <= 31 &&
    config.page_bits <= config.max_bits else {
    raise InvalidField("invalid extensible array configuration")
  }
  let first_super = 2 * array_log2(config.min_pointers)
  guard 1 + config.max_bits - array_log2(config.min_elements) >= first_super &&
    1L << config.page_bits >= config.index_elements.to_int64() &&
    1L << config.page_bits >=
    config.min_elements.to_int64() << (first_super / 2) else {
    raise InvalidField("inconsistent extensible array configuration")
  }
}

///|
fn extensible_slot_offsets(
  slot_index : Int64,
  order : Array[Int32],
  grid : Array[Int64],
  chunk_dimensions : Array[UInt32],
) -> Array[UInt64] {
  let rank = order.length()
  let offsets : Array[UInt64] = Array::make(rank, 0UL)
  let mut remaining = slot_index
  let mut slot = rank
  while slot > 0 {
    slot = slot - 1
    let dimension = order[slot]
    let coordinate = remaining % grid[dimension]
    remaining = remaining / grid[dimension]
    offsets[dimension] = (coordinate * chunk_dimensions[dimension].to_int64()).reinterpret_as_uint64()
  }
  offsets
}

///|
priv struct MissingExtensibleDataBlock {
  count : Int64
  start : Int64
  limit : Int64
  block_offset : Int64
  order : Array[Int32]
  grid : Array[Int64]
  pointer_offset : Int32
  pointer_width : Int32
  pointer_metadata_start : Int32
  pointer_checksum_offset : Int32
  bitmap_start : Int64
  bitmap_offset : Int32
  bitmap_metadata_start : Int32
  bitmap_checksum_offset : Int32
  client : Int32
  entry_size : Int32
}

///|
priv struct ExtensibleReader {
  source : &Hdf5Source
  layout : DataLayout
  dataspace : Dataspace
  datatype : Datatype
  offset_size : Int32
  base_address : UInt64
  undefined : UInt64
  filters : Array[Filter]
  output : FixedArray[Byte]
  selection : Array[UInt64]?
  records : Array[RawChunkRecord]?
  config : ExtensibleArrayConfig
  client : Int32
  entry_size : Int32
  chunk_size : Int32
  order : Array[Int32]
  grid : Array[Int64]
  limit : Int64
  missing_blocks : Array[MissingExtensibleDataBlock]?
}

///|
fn ExtensibleReader::entries(
  self : ExtensibleReader,
  reader : Reader,
  count : Int64,
  start : Int64,
  base_offset : Int32,
  metadata_start : Int32,
  metadata_checksum_offset : Int32,
  unallocated_page : RawChunkPage?,
  missing_block_index~ : Int32 = -1,
) -> Unit raise Hdf5Error {
  guard count <=
    (reader.remaining().to_int64() - 4) / self.entry_size.to_int64() else {
    raise InvalidField("extensible array entries extend beyond input")
  }
  let end = if start + count < self.limit { count } else { self.limit - start }
  for entry_index in 0..<end.to_int() {
    let address_offset = base_offset + reader.position()
    let stored_size_offset = base_offset + reader.position() + self.offset_size
    let unallocated = unallocated_page is Some(_)
    let address = if unallocated {
      reader.skip(self.entry_size)
      self.undefined
    } else {
      reader.read_uint_le(self.offset_size)
    }
    let stored_size = if unallocated {
      self.chunk_size.to_uint64()
    } else if self.client == 1 {
      reader.read_var_uint_le(self.entry_size - self.offset_size - 4)
    } else {
      self.chunk_size.to_uint64()
    }
    let mask = if unallocated || self.client != 1 {
      0U
    } else {
      reader.read_u32_le()
    }
    let offsets = extensible_slot_offsets(
      start + entry_index.to_int64(),
      self.order,
      self.grid,
      self.layout.chunk_dimensions,
    )
    let mut outside = false
    for dimension in 0..<offsets.length() {
      if offsets[dimension] >= self.dataspace.dimensions[dimension] {
        outside = true
      }
    }
    if outside {
      continue
    }
    push_chunk_page_slot_record(
      self.records,
      offsets,
      address,
      stored_size,
      mask,
      address_offset,
      self.offset_size,
      if self.client == 1 {
        stored_size_offset
      } else {
        -1
      },
      if self.client == 1 {
        self.entry_size - self.offset_size - 4
      } else {
        0
      },
      if self.client == 1 {
        stored_size_offset + self.entry_size - self.offset_size - 4
      } else {
        -1
      },
      metadata_start,
      metadata_checksum_offset,
      unallocated_page,
      missing_block_index~,
    )
    if address == self.undefined {
      continue
    }
    if !chunk_contains_selection(
        offsets,
        self.layout.chunk_dimensions,
        self.dataspace.dimensions,
        self.selection,
      ) {
      continue
    }
    let decoded = read_source_filtered_chunk(
      self.source,
      address,
      self.base_address,
      stored_size,
      self.filters,
      mask,
      self.datatype.size,
      self.chunk_size,
    )
    copy_chunk(
      decoded,
      decoded.length(),
      offsets,
      self.layout.chunk_dimensions,
      self.dataspace.dimensions,
      self.datatype.size,
      self.output,
      self.selection,
    )
  }
}

///|
fn ExtensibleReader::data_block(
  self : ExtensibleReader,
  address : UInt64,
  count : Int64,
  start : Int64,
  block_offset : Int64,
  pointer_offset : Int32,
  pointer_width : Int32,
  pointer_metadata_start : Int32,
  pointer_checksum_offset : Int32,
  bitmap : Bytes,
  bitmap_start : Int64,
  bitmap_offset : Int32,
  bitmap_metadata_start : Int32,
  bitmap_checksum_offset : Int32,
) -> Unit raise Hdf5Error {
  if start >= self.limit {
    return
  }
  if address == self.undefined {
    match self.missing_blocks {
      Some(blocks) =>
        blocks.push({
          count,
          start,
          limit: self.limit,
          block_offset,
          order: self.order,
          grid: self.grid,
          pointer_offset,
          pointer_width,
          pointer_metadata_start,
          pointer_checksum_offset,
          bitmap_start,
          bitmap_offset,
          bitmap_metadata_start,
          bitmap_checksum_offset,
          client: self.client,
          entry_size: self.entry_size,
        })
      None => ()
    }
    return
  }
  let prefix_size = 6 + self.offset_size + (self.config.max_bits + 7) / 8
  let page_size = 1L << self.config.page_bits
  let total_size = if count <= page_size {
    prefix_size.to_int64() + count * self.entry_size.to_int64() + 4L
  } else {
    let pages = (count + page_size - 1) / page_size
    let page_bytes = page_size * self.entry_size.to_int64() + 4L
    let last_page_elements = count % page_size
    prefix_size.to_int64() +
    4L +
    (pages - 1) * page_bytes +
    (if last_page_elements > 0 {
      last_page_elements * self.entry_size.to_int64() + 4L
    } else {
      page_bytes
    })
  }
  guard total_size <= 0x7FFFFFFFL else {
    raise InvalidField("extensible array data block is too large")
  }
  let block_address = source_address_to_offset(address, self.base_address)
  guard block_address <= 0x7FFFFFFFUL else {
    raise InvalidField("extensible array block offset is out of range")
  }
  let block_start = block_address.to_int()
  let block = self.source.read_at(block_address, total_size.to_int())
  let reader = Reader::new(block)
  guard reader.read_bytes(4) == b"EADB" &&
    reader.read_u8() == b'\x00' &&
    reader.read_u8().to_int() == self.client &&
    reader.read_uint_le(self.offset_size) == self.layout.data_address else {
    raise InvalidField("invalid extensible array block")
  }
  guard reader.read_var_uint_le((self.config.max_bits + 7) / 8) ==
    block_offset.reinterpret_as_uint64() else {
    raise InvalidField("invalid extensible array data block offset")
  }
  if count <= page_size {
    verify_metadata_checksum(
      block,
      0,
      reader.position() + (count * self.entry_size.to_int64()).to_int(),
      "extensible array data block",
    )
    self.entries(
      reader,
      count,
      start,
      block_start,
      block_start,
      block_start +
      reader.position() +
      (count * self.entry_size.to_int64()).to_int(),
      None,
    )
  } else {
    verify_metadata_checksum(
      block,
      0,
      reader.position(),
      "extensible array data block",
    )
    reader.skip(4)
    let pages = (count + page_size - 1) / page_size
    let page_bytes = page_size * self.entry_size.to_int64() + 4
    let last_page_elements = count % page_size
    let pages_size = (pages - 1) * page_bytes +
      (if last_page_elements > 0 {
        last_page_elements * self.entry_size.to_int64() + 4
      } else {
        page_bytes
      })
    guard pages_size <= reader.remaining().to_int64() else {
      raise InvalidField("extensible array pages extend beyond input")
    }
    let page_start = reader.position().to_int64()
    for page in 0..<pages.to_int() {
      let first = start + page.to_int64() * page_size
      if first >= self.limit {
        break
      }
      let bit = bitmap_start + page.to_int64()
      guard bit / 8 < bitmap.length().to_int64() else {
        raise InvalidField("invalid extensible array page bitmap")
      }
      let page_elements = if page + 1 == pages.to_int() &&
        last_page_elements > 0 {
        last_page_elements
      } else {
        page_size
      }
      let page_offset = page_start + page.to_int64() * page_bytes
      let page_metadata_start = block_start + page_offset.to_int()
      let page_checksum_offset = page_metadata_start +
        (page_elements * self.entry_size.to_int64()).to_int()
      if (bitmap[(bit / 8).to_int()].to_int() & (0x80 >> (bit % 8).to_int())) !=
        0 {
        verify_metadata_checksum(
          block,
          page_offset.to_int(),
          (page_offset + page_elements * self.entry_size.to_int64()).to_int(),
          "extensible array data block page",
        )
        self.entries(
          Reader::at(block, page_offset.to_int()),
          page_elements,
          first,
          block_start,
          block_start + page_offset.to_int(),
          block_start +
          page_offset.to_int() +
          (page_elements * self.entry_size.to_int64()).to_int(),
          None,
        )
      } else {
        guard bitmap_offset >= 0 &&
          bitmap_metadata_start >= 0 &&
          bitmap_checksum_offset >= bitmap_metadata_start &&
          page_metadata_start >= block_start &&
          page_checksum_offset >= page_metadata_start else {
          raise InvalidField("invalid extensible array page metadata")
        }
        let page_record : RawChunkPage = {
          start: page_metadata_start,
          entries: page_elements.to_int(),
          entry_size: self.entry_size,
          checksum_offset: page_checksum_offset,
          bitmap_offset: bitmap_offset + (bit / 8).to_int(),
          bitmap_mask: (0x80 >> (bit % 8).to_int()).to_byte(),
          bitmap_metadata_start,
          bitmap_checksum_offset,
          address_width: self.offset_size,
          stored_size_width: if self.client == 1 {
            self.entry_size - self.offset_size - 4
          } else {
            0
          },
        }
        self.entries(
          Reader::at(block, page_offset.to_int()),
          page_elements,
          first,
          block_start,
          page_metadata_start,
          page_checksum_offset,
          Some(page_record),
        )
      }
    }
  }
}

///|
/// A newly created extensible-array data block. Records point at entries in the
/// image while the block pointer and page bitmap live in the existing index or
/// super block.
priv struct MissingExtensibleBlockStorage {
  image : FixedArray[Byte]
  records : Array[RawChunkRecord]
  block_start : Int32
  pointer_offset : Int32
  pointer_width : Int32
  pointer_metadata_start : Int32
  pointer_checksum_offset : Int32
}

///|
fn initialize_extensible_entry(
  image : FixedArray[Byte],
  position : Int32,
  offset_size : Int32,
  client : Int32,
  size_width : Int32,
  undefined : UInt64,
) -> Unit raise Hdf5Error {
  patch_little_endian_uint(
    image, position, offset_size, undefined, "extensible array chunk address",
  )
  if client == 1 {
    patch_little_endian_uint(
      image,
      position + offset_size,
      size_width,
      0UL,
      "extensible array chunk stored size",
    )
    patch_little_endian_uint(
      image,
      position + offset_size + size_width,
      4,
      0UL,
      "extensible array chunk filter mask",
    )
  }
}

///|
/// Builds an empty EADB for an extensible-array block pointer that is
/// undefined. Every entry starts as undefined, and paged blocks start with an
/// empty page bitmap.
fn prepare_missing_extensible_data_block(
  bytes : Bytes,
  layout : DataLayout,
  dataspace : Dataspace,
  datatype : Datatype,
  offset_size : Int32,
  base_address : UInt64,
  config : ExtensibleArrayConfig,
  filters : Array[Filter],
  undefined : UInt64,
  descriptor : MissingExtensibleDataBlock,
  block_index : Int32,
  block_start : Int32,
) -> MissingExtensibleBlockStorage raise Hdf5Error {
  let client = descriptor.client
  let entry_size = descriptor.entry_size
  let size_width = entry_size - offset_size - 4
  guard descriptor.count > 0L &&
    descriptor.block_offset >= 0L &&
    descriptor.start >= 0L &&
    descriptor.limit >= 0L &&
    block_start >= 0 &&
    (
      (client == 0 && filters.length() == 0 && entry_size == offset_size) ||
      (
        client == 1 &&
        filters.length() > 0 &&
        size_width >= 1 &&
        size_width <= 8
      )
    ) else {
    raise InvalidField("invalid missing extensible array data block")
  }
  let prefix_size = 6 + offset_size + (config.max_bits + 7) / 8
  let page_size = 1L << config.page_bits
  let paged = descriptor.count > page_size
  let pages = (descriptor.count + page_size - 1L) / page_size
  let page_bytes = page_size * entry_size.to_int64() + 4L
  let last_page_elements = descriptor.count % page_size
  let total_size = if !paged {
    prefix_size.to_int64() + descriptor.count * entry_size.to_int64() + 4L
  } else {
    prefix_size.to_int64() +
    4L +
    (pages - 1L) * page_bytes +
    (if last_page_elements > 0L {
      last_page_elements * entry_size.to_int64() + 4L
    } else {
      page_bytes
    })
  }
  guard total_size <= 0x7FFFFFFFL &&
    block_start.to_int64() + total_size <= 0x7FFFFFFFL else {
    raise InvalidField("extensible array data block is too large")
  }
  let image = FixedArray::make(total_size.to_int(), b'\x00')
  image[0] = b'E'
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
    "extensible array data block header",
  )
  patch_little_endian_uint(
    image,
    6 + offset_size,
    (config.max_bits + 7) / 8,
    descriptor.block_offset.reinterpret_as_uint64(),
    "extensible array data block offset",
  )
  let chunk_size = chunk_byte_size(layout.chunk_dimensions, datatype.size)
  let records : Array[RawChunkRecord] = []
  let context : ExtensibleReader = {
    source: bytes,
    layout,
    dataspace,
    datatype,
    offset_size,
    base_address,
    undefined,
    filters,
    output: FixedArray::make(0, b'\x00'),
    selection: None,
    records: Some(records),
    config,
    client,
    entry_size,
    chunk_size,
    order: descriptor.order,
    grid: descriptor.grid,
    limit: descriptor.limit,
    missing_blocks: None,
  }
  if !paged {
    let entries_start = prefix_size
    for entry in 0..<descriptor.count.to_int() {
      initialize_extensible_entry(
        image,
        entries_start + entry * entry_size,
        offset_size,
        client,
        size_width,
        undefined,
      )
    }
    let checksum_offset = entries_start + descriptor.count.to_int() * entry_size
    patch_metadata_checksum_in_place(image, 0, checksum_offset)
    let image_bytes = Bytes::from_array(image)
    context.entries(
      Reader::at(image_bytes, entries_start),
      descriptor.count,
      descriptor.start,
      block_start,
      block_start,
      block_start + checksum_offset,
      None,
      missing_block_index=block_index,
    )
  } else {
    patch_metadata_checksum_in_place(image, 0, prefix_size)
    let image_bytes = Bytes::from_array(image)
    let pages_start = prefix_size + 4
    for page in 0..<pages.to_int() {
      let page_elements = if page + 1 == pages.to_int() &&
        last_page_elements > 0L {
        last_page_elements
      } else {
        page_size
      }
      let page_offset = pages_start + page * page_bytes.to_int()
      let page_length = page_elements.to_int() * entry_size
      for entry in 0..<page_elements.to_int() {
        initialize_extensible_entry(
          image,
          page_offset + entry * entry_size,
          offset_size,
          client,
          size_width,
          undefined,
        )
      }
      patch_metadata_checksum_in_place(
        image,
        page_offset,
        page_offset + page_length,
      )
      let first = descriptor.start + page.to_int64() * page_size
      if first >= descriptor.limit {
        continue
      }
      let bit = descriptor.bitmap_start + page.to_int64()
      guard descriptor.bitmap_offset >= 0 &&
        descriptor.bitmap_metadata_start >= 0 &&
        descriptor.bitmap_checksum_offset >= descriptor.bitmap_metadata_start &&
        bit >= 0L &&
        bit / 8L <= 0x7FFFFFFFL - descriptor.bitmap_offset.to_int64() else {
        raise InvalidField("invalid extensible array page bitmap")
      }
      let page_record : RawChunkPage = {
        start: block_start + page_offset,
        entries: page_elements.to_int(),
        entry_size,
        checksum_offset: block_start + page_offset + page_length,
        bitmap_offset: descriptor.bitmap_offset + (bit / 8L).to_int(),
        bitmap_mask: (0x80 >> (bit % 8L).to_int()).to_byte(),
        bitmap_metadata_start: descriptor.bitmap_metadata_start,
        bitmap_checksum_offset: descriptor.bitmap_checksum_offset,
        address_width: offset_size,
        stored_size_width: if client == 1 {
          size_width
        } else {
          0
        },
      }
      context.entries(
        Reader::at(image_bytes, page_offset),
        page_elements,
        first,
        block_start,
        block_start + page_offset,
        block_start + page_offset + page_length,
        Some(page_record),
        missing_block_index=block_index,
      )
    }
  }
  {
    image,
    records,
    block_start,
    pointer_offset: descriptor.pointer_offset,
    pointer_width: descriptor.pointer_width,
    pointer_metadata_start: descriptor.pointer_metadata_start,
    pointer_checksum_offset: descriptor.pointer_checksum_offset,
  }
}

///|
fn read_extensible_array_chunks(
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
  config : ExtensibleArrayConfig,
  selection : Array[UInt64]?,
  records~ : Array[RawChunkRecord]? = None,
  missing_blocks~ : Array[MissingExtensibleDataBlock]? = None,
) -> Unit raise Hdf5Error {
  let sources : Array[&Hdf5Source] = [bytes]
  read_source_extensible_array_chunks(
    sources[0],
    layout,
    dataspace,
    datatype,
    offset_size,
    length_size,
    base_address,
    filters,
    output,
    undefined,
    config,
    selection,
    records~,
    missing_blocks~,
  )
}

///|
fn read_source_extensible_array_chunks(
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
  config : ExtensibleArrayConfig,
  selection : Array[UInt64]?,
  records~ : Array[RawChunkRecord]? = None,
  missing_blocks~ : Array[MissingExtensibleDataBlock]? = None,
) -> Unit raise Hdf5Error {
  validate_extensible_config(config)
  let unlimited_value = match length_size {
    1 | 2 | 4 => (1UL << (length_size * 8)) - 1UL
    8 => 0xFFFFFFFFFFFFFFFFUL
    _ => raise UnsupportedAddressSize(length_size)
  }
  let rank = dataspace.dimensions.length()
  guard dataspace.maximum_dimensions.length() == rank else {
    raise InvalidField("extensible array requires maximum dimensions")
  }
  let mut unlimited = -1
  let grid : Array[Int64] = []
  for dimension in 0..<rank {
    let current = dataspace.dimensions[dimension]
    let maximum = dataspace.maximum_dimensions[dimension]
    let width = layout.chunk_dimensions[dimension].to_int64()
    guard current <= maximum && current <= 0x7FFFFFFFUL && width > 0 else {
      raise InvalidField("invalid extensible array dimensions")
    }
    let extent = if maximum == unlimited_value {
      guard unlimited == -1 else {
        raise InvalidField("extensible array requires one unlimited dimension")
      }
      unlimited = dimension
      current
    } else {
      guard maximum <= 0x7FFFFFFFUL else {
        raise InvalidField("extensible array maximum dimension is too large")
      }
      maximum
    }
    grid.push((extent.reinterpret_as_int64() + width - 1) / width)
  }
  guard unlimited >= 0 else {
    raise InvalidField("extensible array requires one unlimited dimension")
  }
  let order = [unlimited]
  for dimension in 0..<rank {
    if dimension != unlimited {
      order.push(dimension)
    }
  }
  let header_offset = source_address_to_offset(
    layout.data_address,
    base_address,
  )
  let header_content = 12 + length_size * 6 + offset_size
  let header = Reader::new(
    source_read_checksummed(
      source, header_offset, header_content, "extensible array header",
    ),
  )
  guard header.read_bytes(4) == b"EAHD" && header.read_u8() == b'\x00' else {
    raise InvalidField("invalid extensible array header")
  }
  let client = header.read_u8().to_int()
  let entry_size = header.read_u8().to_int()
  let actual : ExtensibleArrayConfig = {
    max_bits: header.read_u8().to_int(),
    index_elements: header.read_u8().to_int(),
    min_elements: header.read_u8().to_int(),
    min_pointers: header.read_u8().to_int(),
    page_bits: header.read_u8().to_int(),
  }
  let size_width = entry_size - offset_size - 4
  guard actual == config else {
    raise InvalidField("extensible array configuration does not match layout")
  }
  guard (client == 0 && filters.length() == 0 && entry_size == offset_size) ||
    (client == 1 && filters.length() > 0 && size_width >= 1 && size_width <= 8) else {
    raise InvalidField("invalid extensible array entry encoding")
  }
  for _ in 0..<4 {
    header.read_uint_le(length_size) |> ignore
  }
  let maximum_set = header.read_uint_le(length_size)
  header.read_uint_le(length_size) |> ignore
  let index_address = header.read_uint_le(offset_size)
  guard header.position() == header_content else {
    raise InvalidField("extensible array header size mismatch")
  }
  header.skip(4)
  guard config.max_bits == 64 || maximum_set <= 1UL << config.max_bits else {
    raise InvalidField("extensible array index exceeds capacity")
  }
  if index_address == undefined || (output.length() == 0 && records is None) {
    return
  }
  let total = product(grid)
  let limit = if maximum_set < total.reinterpret_as_uint64() {
    maximum_set.reinterpret_as_int64()
  } else {
    total
  }
  let context : ExtensibleReader = {
    source,
    layout,
    dataspace,
    datatype,
    offset_size,
    base_address,
    undefined,
    filters,
    output,
    selection,
    records,
    config,
    client,
    entry_size,
    chunk_size: chunk_byte_size(layout.chunk_dimensions, datatype.size),
    order,
    grid,
    limit,
    missing_blocks,
  }
  let first_super = 2 * array_log2(config.min_pointers)
  let levels = 1 + config.max_bits - array_log2(config.min_elements)
  let direct_count = 2 * (config.min_pointers - 1)
  let index_content = 6 +
    offset_size +
    config.index_elements * entry_size +
    direct_count * offset_size +
    (levels - first_super) * offset_size
  let index_bytes = source_read_checksummed(
    source,
    source_address_to_offset(index_address, base_address),
    index_content,
    "extensible array index block",
  )
  let index = Reader::new(index_bytes)
  guard index.read_bytes(4) == b"EAIB" &&
    index.read_u8() == b'\x00' &&
    index.read_u8().to_int() == client &&
    index.read_uint_le(offset_size) == layout.data_address else {
    raise InvalidField("invalid extensible array block")
  }
  let entry_start = index.position()
  index.skip(config.index_elements * entry_size)
  let direct : Array[UInt64] = []
  for _ in 0..<direct_count {
    direct.push(index.read_uint_le(offset_size))
  }
  let supers : Array[UInt64] = []
  for _ in first_super..<levels {
    supers.push(index.read_uint_le(offset_size))
  }
  guard index.position() == index_content else {
    raise InvalidField("extensible array index block size mismatch")
  }
  index.skip(4)
  let index_address_offset = source_address_to_offset(
    index_address, base_address,
  )
  guard index_address_offset <= 0x7FFFFFFFUL else {
    raise InvalidField("extensible array index offset is out of range")
  }
  let index_offset = index_address_offset.to_int()
  context.entries(
    Reader::at(index_bytes, entry_start),
    config.index_elements.to_int64(),
    0L,
    index_offset,
    index_offset,
    index_offset + index_content,
    None,
  )
  let mut start = 0L
  let mut direct_start = 0
  for level in 0..<levels {
    if start + config.index_elements.to_int64() >= limit {
      break
    }
    let blocks = 1L << (level / 2)
    let count = config.min_elements.to_int64() << ((level + 1) / 2)
    if level < first_super {
      for block_index in 0..<blocks.to_int() {
        let pointer_offset = index_offset +
          entry_start +
          config.index_elements * entry_size +
          (direct_start + block_index) * offset_size
        guard pointer_offset <= 0x7FFFFFFF else {
          raise InvalidField("extensible array pointer offset is out of range")
        }
        context.data_block(
          direct[direct_start + block_index],
          count,
          config.index_elements.to_int64() +
          start +
          block_index.to_int64() * count,
          start + (direct_start + block_index).to_int64() * count,
          pointer_offset,
          offset_size,
          index_offset,
          index_offset + index_content,
          b"",
          0L,
          -1,
          -1,
          -1,
        )
      }
      direct_start = direct_start + blocks.to_int()
    } else if supers[level - first_super] != undefined {
      let page_size = 1L << config.page_bits
      let pages = if count > page_size { count / page_size } else { 0L }
      let bitmap_size = blocks * ((pages + 7) / 8)
      let super_content = 6L +
        offset_size.to_int64() +
        ((config.max_bits + 7) / 8).to_int64() +
        bitmap_size +
        blocks * offset_size.to_int64()
      guard super_content <= 0x7FFFFFFFL else {
        raise InvalidField("extensible array super block is too large")
      }
      let super_address = source_address_to_offset(
        supers[level - first_super],
        base_address,
      )
      guard super_address <= 0x7FFFFFFFUL &&
        super_content + super_address.reinterpret_as_int64() <= 0x7FFFFFFFL else {
        raise InvalidField(
          "extensible array super block offset is out of range",
        )
      }
      let super_offset = super_address.to_int()
      let super_bytes = source_read_checksummed(
        source,
        super_address,
        super_content.to_int(),
        "extensible array super block",
      )
      let super_block = Reader::new(super_bytes)
      guard super_block.read_bytes(4) == b"EASB" &&
        super_block.read_u8() == b'\x00' &&
        super_block.read_u8().to_int() == client &&
        super_block.read_uint_le(offset_size) == layout.data_address else {
        raise InvalidField("invalid extensible array block")
      }
      guard super_block.read_var_uint_le((config.max_bits + 7) / 8) ==
        start.reinterpret_as_uint64() else {
        raise InvalidField("invalid extensible array super block offset")
      }
      guard bitmap_size + blocks * offset_size.to_int64() + 4 <=
        super_block.remaining().to_int64() else {
        raise InvalidField("extensible array super block extends beyond input")
      }
      let bitmap_position = super_block.position()
      let bitmap = super_block.read_bytes(bitmap_size.to_int())
      for block_index in 0..<blocks.to_int() {
        let pointer_position = super_block.position()
        let address = super_block.read_uint_le(offset_size)
        context.data_block(
          address,
          count,
          config.index_elements.to_int64() +
          start +
          block_index.to_int64() * count,
          start + block_index.to_int64() * count,
          super_offset + pointer_position,
          offset_size,
          super_offset,
          super_offset + super_content.to_int(),
          bitmap,
          block_index.to_int64() * pages,
          super_offset + bitmap_position,
          super_offset,
          super_offset + super_content.to_int(),
        )
      }
      guard super_block.position() == super_content.to_int() else {
        raise InvalidField("extensible array super block size mismatch")
      }
      super_block.skip(4)
    }
    start = start + blocks * count
  }
}
