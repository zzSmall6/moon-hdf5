///|
fn datatype_semantically_equal(left : Datatype, right : Datatype) -> Bool {
  if left.class != right.class ||
    left.size != right.size ||
    left.signed != right.signed ||
    left.little_endian != right.little_endian ||
    left.vlen_is_string != right.vlen_is_string ||
    left.bit_offset != right.bit_offset ||
    left.bit_precision != right.bit_precision ||
    left.string_padding != right.string_padding ||
    left.string_charset != right.string_charset ||
    left.opaque_tag != right.opaque_tag ||
    left.reference_kind != right.reference_kind ||
    left.array_dimensions != right.array_dimensions ||
    left.base_types.length() != right.base_types.length() ||
    left.compound_members.length() != right.compound_members.length() ||
    left.enum_members.length() != right.enum_members.length() {
    return false
  }
  if left.class == 1 && !is_ieee_float_layout(right) {
    return false
  }
  for index in 0..<left.base_types.length() {
    if !datatype_semantically_equal(
        left.base_types[index],
        right.base_types[index],
      ) {
      return false
    }
  }
  for index in 0..<left.compound_members.length() {
    let left_field = left.compound_members[index]
    let right_field = right.compound_members[index]
    if left_field.name != right_field.name ||
      left_field.offset != right_field.offset ||
      !datatype_semantically_equal(left_field.datatype, right_field.datatype) {
      return false
    }
  }
  for index in 0..<left.enum_members.length() {
    let left_member = left.enum_members[index]
    let right_member = right.enum_members[index]
    if left_member.name != right_member.name ||
      left_member.value != right_member.value {
      return false
    }
  }
  true
}

///|
fn patch_values_match_datatype(
  values : Hdf5WriteValues,
  datatype : Datatype,
) -> Bool {
  match values {
    Int8Values(_) =>
      datatype.class == 0 &&
      datatype.size == 1 &&
      datatype.signed &&
      datatype.bit_offset == 0 &&
      datatype.bit_precision == 8
    Int16Values(_) =>
      datatype.class == 0 &&
      datatype.size == 2 &&
      datatype.signed &&
      datatype.bit_offset == 0 &&
      datatype.bit_precision == 16
    Int32Values(_) =>
      datatype.class == 0 &&
      datatype.size == 4 &&
      datatype.signed &&
      datatype.bit_offset == 0 &&
      datatype.bit_precision == 32
    Int64Values(_) =>
      datatype.class == 0 &&
      datatype.size == 8 &&
      datatype.signed &&
      datatype.bit_offset == 0 &&
      datatype.bit_precision == 64
    UInt8Values(_) =>
      datatype.class == 0 &&
      datatype.size == 1 &&
      !datatype.signed &&
      datatype.bit_offset == 0 &&
      datatype.bit_precision == 8
    UInt16Values(_) =>
      datatype.class == 0 &&
      datatype.size == 2 &&
      !datatype.signed &&
      datatype.bit_offset == 0 &&
      datatype.bit_precision == 16
    UInt32Values(_) =>
      datatype.class == 0 &&
      datatype.size == 4 &&
      !datatype.signed &&
      datatype.bit_offset == 0 &&
      datatype.bit_precision == 32
    UInt64Values(_) =>
      datatype.class == 0 &&
      datatype.size == 8 &&
      !datatype.signed &&
      datatype.bit_offset == 0 &&
      datatype.bit_precision == 64
    Float32Values(_) =>
      datatype.class == 1 &&
      datatype.size == 4 &&
      is_ieee_float_layout(datatype)
    Float16Values(_) =>
      datatype.class == 1 &&
      datatype.size == 2 &&
      is_ieee_float_layout(datatype)
    Float64Values(_) =>
      datatype.class == 1 &&
      datatype.size == 8 &&
      is_ieee_float_layout(datatype)
    FixedStringValues(_, width) =>
      datatype.class == 3 &&
      datatype.size == width &&
      datatype.string_padding == 1 &&
      datatype.string_charset == 1
    Complex64Values(_) =>
      datatype.class == 11 &&
      datatype.size == 8 &&
      datatype.base_types.length() == 1 &&
      datatype.base_types[0].class == 1 &&
      datatype.base_types[0].size == 4 &&
      datatype.base_types[0].little_endian &&
      is_ieee_float_layout(datatype.base_types[0])
    Complex128Values(_) =>
      datatype.class == 11 &&
      datatype.size == 16 &&
      datatype.base_types.length() == 1 &&
      datatype.base_types[0].class == 1 &&
      datatype.base_types[0].size == 8 &&
      datatype.base_types[0].little_endian &&
      is_ieee_float_layout(datatype.base_types[0])
    StructuredValues(write_datatype, _) =>
      datatype_semantically_equal(write_datatype, datatype)
    _ => false
  }
}

///|
fn datatype_contains_reference(datatype : Datatype) -> Bool {
  if datatype.class == 7 {
    return true
  }
  for field in datatype.compound_members {
    if datatype_contains_reference(field.datatype) {
      return true
    }
  }
  for base in datatype.base_types {
    if datatype_contains_reference(base) {
      return true
    }
  }
  false
}

///|
fn patch_byteswap_elements(payload : Bytes, element_size : Int32) -> Bytes {
  if element_size <= 1 {
    return payload
  }
  let output = payload.to_array()
  let count = output.length() / element_size
  for element in 0..<count {
    let start = element * element_size
    for index in 0..<(element_size / 2) {
      let left = start + index
      let right = start + element_size - 1 - index
      let value = output[left]
      output[left] = output[right]
      output[right] = value
    }
  }
  Bytes::from_array(output)
}

///|
fn compact_data_relative_offset(
  message : HeaderMessage,
) -> Int32 raise Hdf5Error {
  let reader = Reader::new(message.data)
  let version = reader.read_u8().to_int()
  let layout_class = if version == 1 || version == 2 {
    let _dimensionality = reader.read_u8().to_int()
    let layout_class = reader.read_u8().to_int()
    reader.skip(5)
    layout_class
  } else if version == 3 || version == 4 {
    reader.read_u8().to_int()
  } else {
    raise InvalidField("unsupported data layout message version")
  }
  guard layout_class == 0 else {
    raise InvalidField("data layout is not compact")
  }
  if version == 1 || version == 2 {
    let size = reader.read_u32_le().reinterpret_as_int()
    guard size >= 0 else {
      raise InvalidField("compact dataset size is negative")
    }
  } else {
    let size = reader.read_u16_le().to_int()
    guard size >= 0 else {
      raise InvalidField("compact dataset size is negative")
    }
  }
  reader.position()
}

///|
fn patch_metadata_checksum(
  bytes : Bytes,
  start : Int32,
  checksum_offset : Int32,
) -> Bytes raise Hdf5Error {
  guard start >= 0 &&
    checksum_offset >= start &&
    checksum_offset.to_int64() + metadata_checksum_size.to_int64() <=
    bytes.length().to_int64() else {
    raise UnexpectedEof(checksum_offset.to_int64())
  }
  let checksum = metadata_checksum(bytes, start, checksum_offset - start)
  let output = bytes.to_array()
  let encoded = @buffer.Buffer::Buffer(size_hint=metadata_checksum_size)
  encoded.write_uint_le(checksum)
  let checksum_bytes = encoded.to_bytes()
  for index in 0..<metadata_checksum_size {
    output[checksum_offset + index] = checksum_bytes[index]
  }
  Bytes::from_array(output)
}

///|
fn Hdf5File::patch_compact_dataset_data(
  self : Hdf5File,
  header : ObjectHeader,
  metadata : DatasetMetadata,
  payload : Bytes,
) -> Bytes raise Hdf5Error {
  let message = match header.first(msg_data_layout) {
    Some(message) => message
    None =>
      raise InvalidField("dataset is missing a required object header message")
  }
  guard message.data_offset >= 0 else {
    raise UnsupportedFeature("shared compact data layout cannot be patched")
  }
  guard metadata.layout.compact_data.length() >= payload.length() else {
    raise InvalidField(
      "compact dataset storage is smaller than its declared shape",
    )
  }
  let offset = message.data_offset + compact_data_relative_offset(message)
  guard offset.to_int64() + payload.length().to_int64() <=
    self.bytes.length().to_int64() else {
    raise UnexpectedEof(offset.to_int64())
  }
  let output = self.bytes.to_array()
  for index in 0..<payload.length() {
    output[offset + index] = payload[index]
  }
  let patched = Bytes::from_array(output)
  if message.checksum_offset >= 0 {
    patch_metadata_checksum(
      patched,
      message.chunk_start,
      message.checksum_offset,
    )
  } else {
    patched
  }
}

///|
fn Hdf5File::patch_contiguous_dataset_data(
  self : Hdf5File,
  metadata : DatasetMetadata,
  payload : Bytes,
) -> Bytes raise Hdf5Error {
  let full_size = payload.length()
  let undefined = if self.superblock.offset_size == 8 {
    0xFFFFFFFFFFFFFFFFUL
  } else {
    (1UL << (self.superblock.offset_size * 8)) - 1UL
  }
  guard metadata.layout.data_address != undefined else {
    raise UnsupportedFeature("unallocated contiguous storage cannot be patched")
  }
  let stored_size = if metadata.layout.data_size > 0UL {
    metadata.layout.data_size
  } else {
    full_size.to_uint64()
  }
  guard stored_size >= full_size.to_uint64() else {
    raise InvalidField("dataset storage is smaller than its declared shape")
  }
  let offset = address_to_offset(
    metadata.layout.data_address,
    self.superblock.base_address,
  )
  guard offset.to_int64() + full_size.to_int64() <=
    self.bytes.length().to_int64() else {
    raise UnexpectedEof(offset.to_int64())
  }
  let output = self.bytes.to_array()
  for index in 0..<payload.length() {
    output[offset + index] = payload[index]
  }
  Bytes::from_array(output)
}

///|
fn collect_raw_chunk_records(
  bytes : Bytes,
  layout : DataLayout,
  dataspace : Dataspace,
  datatype : Datatype,
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
) -> Array[RawChunkRecord] raise Hdf5Error {
  let records : Array[RawChunkRecord] = []
  read_chunked_with_filters(
    bytes,
    layout,
    dataspace,
    datatype,
    offset_size,
    base_address,
    b"",
    [],
    length_size,
    selection=Some([]),
    records=Some(records),
  )
  |> ignore
  records
}

///|
fn effective_fill_element(
  header : ObjectHeader,
  datatype : Datatype,
) -> Bytes raise Hdf5Error {
  let fill = dataset_fill_value(header, datatype)
  if fill.length() == 0 {
    return Bytes::make(datatype.size, b'\x00')
  }
  guard fill.length() == datatype.size else {
    raise InvalidField("fill value size does not match datatype")
  }
  fill
}

///|
fn Hdf5File::patch_chunked_dataset_data(
  self : Hdf5File,
  header : ObjectHeader,
  metadata : DatasetMetadata,
  payload : Bytes,
) -> Bytes raise Hdf5Error {
  let layout = metadata.layout
  guard metadata.dataspace.kind == Simple &&
    metadata.dataspace.dimensions.length() > 0 else {
    raise InvalidField("chunked datasets require a simple dataspace")
  }
  let rank = metadata.dataspace.dimensions.length()
  guard layout.chunk_dimensions.length() == rank else {
    raise InvalidField("chunk rank does not match the dataspace rank")
  }
  guard layout.element_size == metadata.datatype.size else {
    raise InvalidField("chunk element size does not match datatype")
  }
  match layout.chunk_index {
    Single(Some(_), _) =>
      raise UnsupportedFeature("filtered chunk storage cannot be patched")
    _ => ()
  }
  let chunk_widths : Array[Int64] = []
  let grid : Array[Int64] = []
  let mut total_chunks = 1L
  for dimension in 0..<rank {
    let width = layout.chunk_dimensions[dimension].to_int64()
    guard width > 0L && width <= 0x7FFFFFFFL else {
      raise InvalidField("invalid chunk dimension")
    }
    chunk_widths.push(width)
    let extent = metadata.dataspace.dimensions[dimension]
    guard extent > 0UL && extent <= 0x7FFFFFFFUL else {
      raise InvalidField("invalid chunked dataspace dimension")
    }
    let count = (extent.reinterpret_as_int64() + width - 1L) / width
    guard count > 0L && total_chunks <= 0x7FFFFFFFL / count else {
      raise InvalidField("chunk grid is too large")
    }
    grid.push(count)
    total_chunks = total_chunks * count
  }
  let chunk_size = chunk_byte_size(
    layout.chunk_dimensions,
    metadata.datatype.size,
  )
  let chunk_elements = product(chunk_widths)
  let records = collect_raw_chunk_records(
    self.bytes,
    layout,
    metadata.dataspace,
    metadata.datatype,
    self.superblock.offset_size,
    self.superblock.length_size,
    self.superblock.base_address,
  )
  let record_for_chunk : Array[RawChunkRecord?] = Array::make(
    total_chunks.to_int(),
    None,
  )
  for record in records {
    guard record.offsets.length() == rank else {
      raise InvalidField("chunk record rank does not match the dataspace rank")
    }
    let mut chunk_linear = 0L
    for axis in 0..<rank {
      let width = chunk_widths[axis].reinterpret_as_uint64()
      let offset = record.offsets[axis]
      guard offset % width == 0UL else {
        raise InvalidField("chunk offset is not aligned to the chunk shape")
      }
      let coordinate = offset / width
      guard coordinate < grid[axis].reinterpret_as_uint64() else {
        raise InvalidField("chunk offset is outside the chunk grid")
      }
      chunk_linear = chunk_linear * grid[axis] +
        coordinate.reinterpret_as_int64()
    }
    guard record_for_chunk[chunk_linear.to_int()] is None else {
      raise InvalidField("duplicate chunk record")
    }
    record_for_chunk[chunk_linear.to_int()] = Some(record)
  }
  let fill = if records.length() == total_chunks.to_int() {
    b""
  } else {
    effective_fill_element(header, metadata.datatype)
  }
  let strides : Array[Int64] = Array::make(rank, 1L)
  let mut stride = 1L
  let mut dimension = rank
  while dimension > 0 {
    dimension = dimension - 1
    strides[dimension] = stride
    stride = stride *
      metadata.dataspace.dimensions[dimension].reinterpret_as_int64()
  }
  let output = self.bytes.to_array()
  for chunk_linear in 0..<total_chunks.to_int() {
    let offsets : Array[UInt64] = Array::make(rank, 0UL)
    let mut remaining = chunk_linear.to_int64()
    let mut axis = rank
    while axis > 0 {
      axis = axis - 1
      let coordinate = remaining % grid[axis]
      remaining = remaining / grid[axis]
      offsets[axis] = (coordinate * chunk_widths[axis]).reinterpret_as_uint64()
    }
    match record_for_chunk[chunk_linear] {
      Some(record) => {
        guard record.size == chunk_size.to_uint64() else {
          raise UnsupportedFeature(
            "chunked patching requires unfiltered chunks with native size",
          )
        }
        for axis in 0..<rank {
          guard record.offsets[axis] == offsets[axis] else {
            raise InvalidField(
              "chunk record offset does not match its grid slot",
            )
          }
        }
        let chunk_offset = address_to_offset(
          record.address,
          self.superblock.base_address,
        )
        guard chunk_offset >= 0 &&
          chunk_offset.to_int64() + chunk_size.to_int64() <=
          self.bytes.length().to_int64() else {
          raise UnexpectedEof(chunk_offset.to_int64())
        }
        let chunk = Reader::at(self.bytes, chunk_offset)
          .read_bytes(chunk_size)
          .to_array()
        for chunk_element in 0..<chunk_elements.to_int() {
          let mut remaining = chunk_element.to_int64()
          let mut dataset_linear = 0L
          let mut outside = false
          let mut axis = rank
          while axis > 0 {
            axis = axis - 1
            let coordinate = remaining % chunk_widths[axis]
            remaining = remaining / chunk_widths[axis]
            let global = offsets[axis] + coordinate.reinterpret_as_uint64()
            if global >= metadata.dataspace.dimensions[axis] {
              outside = true
            } else {
              dataset_linear = dataset_linear +
                global.reinterpret_as_int64() * strides[axis]
            }
          }
          if outside {
            continue
          }
          let source = dataset_linear * metadata.datatype.size.to_int64()
          let destination = chunk_element.to_int64() *
            metadata.datatype.size.to_int64()
          for byte_index in 0..<metadata.datatype.size {
            chunk[destination.to_int() + byte_index] = payload[source.to_int() +
              byte_index]
          }
        }
        for byte_index in 0..<chunk_size {
          output[chunk_offset + byte_index] = chunk[byte_index]
        }
      }
      None =>
        for chunk_element in 0..<chunk_elements.to_int() {
          let mut remaining = chunk_element.to_int64()
          let mut dataset_linear = 0L
          let mut outside = false
          let mut axis = rank
          while axis > 0 {
            axis = axis - 1
            let coordinate = remaining % chunk_widths[axis]
            remaining = remaining / chunk_widths[axis]
            let global = offsets[axis] + coordinate.reinterpret_as_uint64()
            if global >= metadata.dataspace.dimensions[axis] {
              outside = true
            } else {
              dataset_linear = dataset_linear +
                global.reinterpret_as_int64() * strides[axis]
            }
          }
          if outside {
            continue
          }
          let source = dataset_linear * metadata.datatype.size.to_int64()
          for byte_index in 0..<metadata.datatype.size {
            guard payload[source.to_int() + byte_index] == fill[byte_index] else {
              raise UnsupportedFeature(
                "chunked patching cannot allocate missing chunks for non-fill values",
              )
            }
          }
        }
    }
  }
  Bytes::from_array(output)
}

///|
fn Hdf5File::patch_dataset_values_impl(
  self : Hdf5File,
  path : String,
  values : Hdf5WriteValues,
  allow_noncontiguous : Bool,
) -> Bytes raise Hdf5Error {
  guard self.source is None else {
    raise InvalidArgument("in-place patching requires a bytes-backed Hdf5File")
  }
  let node = self.resolve(path)
  guard node.kind == Dataset else {
    raise InvalidField("object is not a dataset: " + path)
  }
  let header = self.object_header_at(node.address)
  guard header.first(msg_external_data_files) is None else {
    raise UnsupportedFeature("external storage cannot be patched")
  }
  guard header.first(msg_filter_pipeline) is None else {
    raise UnsupportedFeature("filtered storage cannot be patched")
  }
  let metadata = read_dataset_metadata(
    header,
    self.superblock.offset_size,
    self.superblock.length_size,
  )
  let compact = metadata.layout.layout_class == 0
  let contiguous = metadata.layout.layout_class == 1
  let chunked = metadata.layout.layout_class == 2
  guard contiguous || (allow_noncontiguous && (compact || chunked)) else {
    if allow_noncontiguous {
      raise UnsupportedFeature(
        "only compact, contiguous and unfiltered chunked storage can be patched",
      )
    }
    raise UnsupportedFeature("only contiguous storage can be patched")
  }
  guard !datatype_contains_vlen(metadata.datatype) &&
    !datatype_contains_reference(metadata.datatype) else {
    raise UnsupportedFeature(
      "variable-length and reference datatypes cannot be patched",
    )
  }
  guard patch_values_match_datatype(values, metadata.datatype) else {
    raise InvalidArgument("patched values must match the dataset datatype")
  }
  validate_write_values(metadata.dataspace.dimensions, values)
  let count = metadata.dataspace.element_count()
  guard count >= 0 && count <= 0x7FFFFFFFL / metadata.datatype.size.to_int64() else {
    raise InvalidField("dataset byte size is too large")
  }
  let full_size = count.to_int() * metadata.datatype.size
  let encoded = encode_write_values(values)
  let payload = match values {
    StructuredValues(_) => encoded.payload
    _ =>
      if !metadata.datatype.little_endian &&
        (metadata.datatype.class == 0 || metadata.datatype.class == 1) {
        patch_byteswap_elements(encoded.payload, metadata.datatype.size)
      } else {
        encoded.payload
      }
  }
  guard payload.length() == full_size else {
    raise InvalidArgument("patched values do not match the dataset shape")
  }
  if compact {
    self.patch_compact_dataset_data(header, metadata, payload)
  } else if chunked {
    self.patch_chunked_dataset_data(header, metadata, payload)
  } else {
    self.patch_contiguous_dataset_data(metadata, payload)
  }
}

///|
/// Replaces the values of an allocated contiguous dataset and returns a new
/// byte image. The shape and datatype must match exactly.
///
/// Only raw dataset bytes change: object headers, attributes, links and
/// checksums are preserved. Chunked, compact, external, filtered and
/// variable-length storage is rejected. Use `patch_dataset_values` to patch
/// compact storage as well.
pub fn Hdf5File::patch_contiguous_dataset(
  self : Hdf5File,
  path : String,
  values : Hdf5WriteValues,
) -> Bytes raise Hdf5Error {
  self.patch_dataset_values_impl(path, values, false)
}

///|
/// Replaces the values of a compact, allocated contiguous or unfiltered chunked
/// dataset and returns a new byte image. The shape and datatype must match
/// exactly.
///
/// Compact payloads stay inline; only the containing object-header chunk
/// checksum is recomputed. Contiguous payloads stay at their existing address.
/// Chunked datasets must have every storage chunk allocated and use their
/// native unfiltered size; chunk bytes are updated in place and edge padding is
/// preserved. External, filtered, unallocated, variable-length and reference
/// storage is rejected.
pub fn Hdf5File::patch_dataset_values(
  self : Hdf5File,
  path : String,
  values : Hdf5WriteValues,
) -> Bytes raise Hdf5Error {
  self.patch_dataset_values_impl(path, values, true)
}
