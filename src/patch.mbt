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
fn patch_metadata_checksum_in_place(
  output : FixedArray[Byte],
  start : Int32,
  checksum_offset : Int32,
) -> Unit raise Hdf5Error {
  guard start >= 0 &&
    checksum_offset >= start &&
    checksum_offset.to_int64() + metadata_checksum_size.to_int64() <=
    output.length().to_int64() else {
    raise UnexpectedEof(checksum_offset.to_int64())
  }
  let bytes = output.unsafe_reinterpret_as_bytes()
  let checksum = metadata_checksum(bytes, start, checksum_offset - start)
  for index in 0..<metadata_checksum_size {
    output[checksum_offset + index] = ((checksum >> (index * 8)) & 0xFFU).to_byte()
  }
}

///|
fn patch_little_endian_uint(
  output : FixedArray[Byte],
  offset : Int32,
  width : Int32,
  value : UInt64,
  field_description : String,
) -> Unit raise Hdf5Error {
  guard width >= 1 &&
    width <= 8 &&
    offset >= 0 &&
    offset.to_int64() + width.to_int64() <= output.length().to_int64() else {
    raise UnsupportedFeature(
      field_description + " is not safely writable in this file",
    )
  }
  if width < 8 {
    guard value < 1UL << (width * 8) else {
      raise UnsupportedFeature(field_description + " exceeds its index field")
    }
  }
  for index in 0..<width {
    output[offset + index] = ((value >> (index * 8)) & 0xFFUL).to_byte()
  }
}

///|
fn patch_stored_size_field(
  output : FixedArray[Byte],
  record : RawChunkRecord,
  value : UInt64,
) -> Unit raise Hdf5Error {
  guard record.stored_size_offset >= 0 && record.stored_size_width >= 1 else {
    raise UnsupportedFeature(
      "filtered chunk index does not expose a writable stored-size field",
    )
  }
  patch_little_endian_uint(
    output,
    record.stored_size_offset,
    record.stored_size_width,
    value,
    "filtered chunk encoded size",
  )
}

///|
fn patch_chunk_mask_field(
  output : FixedArray[Byte],
  record : RawChunkRecord,
  value : UInt32,
) -> Unit raise Hdf5Error {
  guard record.mask_offset >= 0 else {
    raise UnsupportedFeature(
      "filtered chunk index does not expose a writable filter-mask field",
    )
  }
  patch_little_endian_uint(
    output,
    record.mask_offset,
    4,
    value.to_uint64(),
    "allocated filtered chunk filter mask",
  )
}

///|
fn patch_chunk_address_field(
  output : FixedArray[Byte],
  record : RawChunkRecord,
  value : UInt64,
) -> Unit raise Hdf5Error {
  guard record.address_offset >= 0 && record.address_width >= 1 else {
    raise UnsupportedFeature(
      "filtered chunk index does not expose a writable address field",
    )
  }
  patch_little_endian_uint(
    output,
    record.address_offset,
    record.address_width,
    value,
    "relocated filtered chunk address",
  )
}

///|
fn patch_end_of_file_address(
  output : FixedArray[Byte],
  superblock : Superblock,
  value : UInt64,
) -> Unit raise Hdf5Error {
  guard superblock.end_of_file_offset >= 0 else {
    raise UnsupportedFeature(
      "superblock does not expose a writable end-of-file address",
    )
  }
  patch_little_endian_uint(
    output,
    superblock.end_of_file_offset,
    superblock.offset_size,
    value,
    "file end-of-file address",
  )
}

///|
fn append_aligned_bytes_to_tail(
  tail : @buffer.Buffer,
  file_length : Int32,
  encoded : Bytes,
  purpose : String,
) -> UInt64 raise Hdf5Error {
  guard file_length >= 0 && file_length.to_int64() <= 0x7FFFFFFFL - 7L else {
    raise UnsupportedFeature("file is too large to " + purpose)
  }
  guard tail.length().to_int64() <= 0x7FFFFFFFL - file_length.to_int64() else {
    raise UnsupportedFeature("file is too large to " + purpose)
  }
  let unpadded = file_length + tail.length()
  guard unpadded.to_int64() <= 0x7FFFFFFFL - 7L else {
    raise UnsupportedFeature("file is too large to " + purpose)
  }
  let start = (unpadded + 7) / 8 * 8
  guard encoded.length().to_int64() <= 0x7FFFFFFFL - start.to_int64() else {
    raise UnsupportedFeature("file is too large to " + purpose)
  }
  for _ in 0..<(start - unpadded) {
    tail.write_byte(b'\x00')
  }
  tail.write_bytes(encoded.get_view().unwrap())
  start.to_uint64()
}

///|
fn append_aligned_chunk_to_tail(
  tail : @buffer.Buffer,
  file_length : Int32,
  encoded : Bytes,
) -> UInt64 raise Hdf5Error {
  append_aligned_bytes_to_tail(
    tail, file_length, encoded, "relocate a filtered chunk",
  )
}

///|
fn Hdf5File::finish_appended_image(
  self : Hdf5File,
  output : FixedArray[Byte],
  tail : @buffer.Buffer,
) -> Bytes raise Hdf5Error {
  guard tail.length().to_int64() <= 0x7FFFFFFFL - self.bytes.length().to_int64() else {
    raise UnsupportedFeature("file is too large to append dataset storage")
  }
  let final_length = self.bytes.length() + tail.length()
  let eof_address = file_offset_to_stored_address(
    final_length.to_uint64(),
    self.superblock.base_address,
  )
  patch_end_of_file_address(output, self.superblock, eof_address)
  if self.superblock.superblock_checksum_offset >= 0 {
    patch_metadata_checksum_in_place(
      output,
      self.superblock._superblock_offset,
      self.superblock.superblock_checksum_offset,
    )
  }
  let tail_bytes = tail.to_bytes()
  let result = FixedArray::make(final_length, b'\x00')
  result.blit_from_bytes(
    0,
    output.unsafe_reinterpret_as_bytes(),
    0,
    self.bytes.length(),
  )
  result.blit_from_bytes(
    self.bytes.length(),
    tail_bytes,
    0,
    tail_bytes.length(),
  )
  Bytes::from_array(result)
}

///|
fn file_offset_to_stored_address(
  offset : UInt64,
  base_address : UInt64,
) -> UInt64 raise Hdf5Error {
  guard offset >= base_address else {
    raise InvalidField("file offset is below the HDF5 base address")
  }
  offset - base_address
}

///|
fn patch_write_filters(
  filters : Array[Filter],
  datatype : Datatype,
  fill : Bytes,
  custom_filters : Array[&Hdf5CustomFilter],
) -> Array[Hdf5WriteFilter] raise Hdf5Error {
  let result : Array[Hdf5WriteFilter] = []
  for filter in filters {
    match filter.id {
      1 => {
        guard filter.parameters.length() == 1 && filter.parameters[0] <= 9U else {
          raise InvalidField("invalid deflate filter parameters")
        }
        result.push(Deflate(filter.parameters[0].reinterpret_as_int()))
      }
      2 => {
        guard filter.parameters.length() == 1 &&
          filter.parameters[0] == datatype.size.reinterpret_as_uint() else {
          raise InvalidField("invalid shuffle filter parameters")
        }
        result.push(Shuffle)
      }
      3 => {
        guard filter.parameters.length() == 0 else {
          raise InvalidField("invalid Fletcher32 filter parameters")
        }
        result.push(Fletcher32)
      }
      6 => {
        let parameters = filter.parameters
        guard parameters.length() == 20 &&
          (datatype.class == 0 || datatype.class == 1) &&
          parameters[0] == (if datatype.class == 0 { 2U } else { 0U }) &&
          parameters[3] == datatype.class.reinterpret_as_uint() &&
          parameters[4] == datatype.size.reinterpret_as_uint() &&
          parameters[5] == (if datatype.signed { 1U } else { 0U }) &&
          parameters[6] == (if datatype.little_endian { 0U } else { 1U }) else {
          raise InvalidField("invalid scale-offset filter parameters")
        }
        guard parameters[7] == 1U else {
          raise UnsupportedFeature(
            "scale-offset patching requires a stored fill value",
          )
        }
        guard patch_scaleoffset_parameters_match_fill(
          parameters, datatype, fill,
        ) else {
          raise InvalidField(
            "scale-offset fill value does not match the dataset fill value",
          )
        }
        let scale = parameters[1].reinterpret_as_int()
        if datatype.class == 0 {
          guard datatype.size >= 1 &&
            datatype.size <= 8 &&
            scale >= 0 &&
            scale <= datatype.size * 8 else {
            raise InvalidField("invalid integer scale-offset scale")
          }
        } else {
          let maximum_digits = if datatype.size == 4 {
            38
          } else if datatype.size == 8 {
            308
          } else {
            raise UnsupportedFeature(
              "scale-offset floating-point width is not supported",
            )
          }
          guard scale >= 0 && scale <= maximum_digits else {
            raise InvalidField("invalid floating-point scale-offset scale")
          }
        }
        result.push(ScaleOffset(scale))
      }
      id => {
        find_custom_filter(custom_filters, id) |> ignore
        result.push(Custom(id, filter.parameters.copy()))
      }
    }
  }
  result
}

///|
fn patch_scaleoffset_parameters_match_fill(
  parameters : Array[UInt32],
  datatype : Datatype,
  fill : Bytes,
) -> Bool {
  if fill.length() != datatype.size {
    return false
  }
  let mut raw = 0UL
  for index in 0..<datatype.size {
    let shift = if datatype.little_endian {
      index * 8
    } else {
      (datatype.size - 1 - index) * 8
    }
    raw = raw | (fill[index].to_uint64() << shift)
  }
  scaleoffset_parameter_value(parameters, datatype.size) == raw
}

///|
fn patch_scaleoffset_needs_little_endian_input(
  filters : Array[Filter],
  mask : UInt32,
  datatype : Datatype,
  element_size : Int32,
) -> Bool raise Hdf5Error {
  let mut first_active = -1
  let mut scaleoffset_index = -1
  for index in 0..<filters.length() {
    if (mask & (1U << index)) != 0U {
      continue
    }
    if first_active < 0 {
      first_active = index
    }
    if filters[index].id == 6 {
      guard scaleoffset_index < 0 else {
        raise UnsupportedFeature(
          "multiple scale-offset filters cannot be re-encoded",
        )
      }
      scaleoffset_index = index
    }
  }
  if scaleoffset_index < 0 {
    return false
  }
  let scale = filters[scaleoffset_index].parameters[1].reinterpret_as_int()
  let passthrough = datatype.class == 0 && scale == element_size * 8
  if passthrough {
    return false
  }
  guard scaleoffset_index == first_active else {
    raise UnsupportedFeature(
      "scale-offset filter must be the first active filter to be re-encoded",
    )
  }
  !datatype.little_endian
}

///|
fn single_chunk_layout_fields(
  message : HeaderMessage,
  length_size : Int32,
) -> (Int32, Int32, Int32) raise Hdf5Error {
  let reader = Reader::new(message.data)
  guard reader.read_u8().to_int() == 4 && reader.read_u8().to_int() == 2 else {
    raise InvalidField("expected a version 4 chunked data layout message")
  }
  let flags = reader.read_u8().to_int()
  guard (flags & 0xFC) == 0 else {
    raise InvalidField("invalid v4 chunk flags")
  }
  let dimensionality = reader.read_u8().to_int()
  guard dimensionality >= 2 else {
    raise InvalidField("invalid chunk dimensionality")
  }
  let width = reader.read_u8().to_int()
  guard width >= 1 && width <= 8 else {
    raise InvalidField("invalid chunk dimension width")
  }
  for _ in 0..<dimensionality {
    reader.read_var_uint_le(width) |> ignore
  }
  guard reader.read_u8().to_int() == 1 else {
    raise InvalidField("expected a single chunk index")
  }
  if (flags & 2) == 0 {
    return (-1, 0, -1)
  }
  let size_offset = message.data_offset + reader.position()
  (size_offset, length_size, size_offset + length_size)
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
  header : ObjectHeader,
  metadata : DatasetMetadata,
  payload : Bytes,
  allow_allocate : Bool,
) -> Bytes raise Hdf5Error {
  let full_size = payload.length()
  let undefined = if self.superblock.offset_size == 8 {
    0xFFFFFFFFFFFFFFFFUL
  } else {
    (1UL << (self.superblock.offset_size * 8)) - 1UL
  }
  if metadata.layout.data_address == undefined {
    guard allow_allocate else {
      raise UnsupportedFeature(
        "unallocated contiguous storage cannot be patched",
      )
    }
    let message = match header.first(msg_data_layout) {
      Some(message) => message
      None =>
        raise InvalidField(
          "dataset is missing a required object header message",
        )
    }
    guard message.data_offset >= 0 &&
      metadata.layout.data_address_offset >= 0 &&
      metadata.layout.data_address_width >= 1 else {
      raise UnsupportedFeature(
        "contiguous layout does not expose a writable address field",
      )
    }
    let tail = @buffer.Buffer::Buffer()
    let new_offset = append_aligned_bytes_to_tail(
      tail,
      self.bytes.length(),
      payload,
      "allocate contiguous storage",
    )
    let new_address = file_offset_to_stored_address(
      new_offset,
      self.superblock.base_address,
    )
    let output = self.bytes.to_fixedarray()
    patch_little_endian_uint(
      output,
      metadata.layout.data_address_offset,
      metadata.layout.data_address_width,
      new_address,
      "allocated contiguous dataset address",
    )
    if metadata.layout.data_size_offset >= 0 &&
      metadata.layout.data_size_width >= 1 {
      patch_little_endian_uint(
        output,
        metadata.layout.data_size_offset,
        metadata.layout.data_size_width,
        full_size.to_uint64(),
        "allocated contiguous dataset size",
      )
    }
    if message.checksum_offset >= 0 {
      patch_metadata_checksum_in_place(
        output,
        message.chunk_start,
        message.checksum_offset,
      )
    }
    return self.finish_appended_image(output, tail)
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
  filters : Array[Filter],
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
    filters,
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
fn single_chunk_slot_record(
  header : ObjectHeader,
  metadata : DatasetMetadata,
  chunk_size : Int32,
  length_size : Int32,
) -> RawChunkRecord raise Hdf5Error {
  let layout = metadata.layout
  guard layout.chunk_index is Single(stored_size, mask) else {
    raise InvalidField("expected a single chunk index")
  }
  guard metadata.dataspace.maximum_dimensions == metadata.dataspace.dimensions &&
    metadata.dataspace.dimensions ==
    layout.chunk_dimensions.map(size => size.to_uint64()) else {
    raise InvalidField("single chunk dimensions must match dataspace")
  }
  let message = match header.first(msg_data_layout) {
    Some(message) => message
    None =>
      raise InvalidField("dataset is missing a required object header message")
  }
  let (stored_size_offset, stored_size_width, mask_offset) = single_chunk_layout_fields(
    message, length_size,
  )
  if stored_size is Some(_) && stored_size_offset < 0 {
    raise InvalidField("single chunk layout is missing its stored size")
  }
  {
    offsets: Array::make(metadata.dataspace.dimensions.length(), 0UL),
    address: layout.data_address,
    size: stored_size.unwrap_or(chunk_size.to_uint64()),
    mask,
    address_offset: layout.data_address_offset,
    address_width: layout.data_address_width,
    stored_size_offset,
    stored_size_width,
    mask_offset,
    metadata_start: message.chunk_start,
    metadata_checksum_offset: message.checksum_offset,
  }
}

///|
fn Hdf5File::patch_chunked_dataset_data(
  self : Hdf5File,
  header : ObjectHeader,
  metadata : DatasetMetadata,
  payload : Bytes,
  values : Hdf5WriteValues,
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
  let filters = match header.first(msg_filter_pipeline) {
    Some(message) => parse_filters(message, custom_filters=self.custom_filters)
    None => []
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
  let undefined = if self.superblock.offset_size == 8 {
    0xFFFFFFFFFFFFFFFFUL
  } else {
    (1UL << (self.superblock.offset_size * 8)) - 1UL
  }
  let records = match layout.chunk_index {
    Single(_, _) =>
      [
        single_chunk_slot_record(
          header,
          metadata,
          chunk_size,
          self.superblock.length_size,
        ),
      ]
    _ =>
      collect_raw_chunk_records(
        self.bytes,
        layout,
        metadata.dataspace,
        metadata.datatype,
        filters,
        self.superblock.offset_size,
        self.superblock.length_size,
        self.superblock.base_address,
      )
  }
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
  let mut all_chunks_allocated = records.length() == total_chunks.to_int()
  for record in records {
    if record.address == undefined {
      all_chunks_allocated = false
    }
  }
  let fill = if all_chunks_allocated && filters.length() == 0 {
    b""
  } else {
    effective_fill_element(header, metadata.datatype)
  }
  let write_filters = patch_write_filters(
    filters,
    metadata.datatype,
    fill,
    self.custom_filters,
  )
  let strides : Array[Int64] = Array::make(rank, 1L)
  let mut stride = 1L
  let mut dimension = rank
  while dimension > 0 {
    dimension = dimension - 1
    strides[dimension] = stride
    stride = stride *
      metadata.dataspace.dimensions[dimension].reinterpret_as_int64()
  }
  let output = self.bytes.to_fixedarray()
  let tail = @buffer.Buffer::Buffer()
  let mut relocated = false
  let checksum_starts : Array[Int32] = []
  let checksum_offsets : Array[Int32] = []
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
        let filtered = filters.length() > 0
        if !filtered {
          guard record.size == chunk_size.to_uint64() else {
            raise UnsupportedFeature(
              "chunked patching requires unfiltered chunks with native size",
            )
          }
        }
        for axis in 0..<rank {
          guard record.offsets[axis] == offsets[axis] else {
            raise InvalidField(
              "chunk record offset does not match its grid slot",
            )
          }
        }
        if record.address == undefined {
          let chunk = FixedArray::make(chunk_size, b'\x00')
          write_repeated_element(chunk, fill)
          let mut has_non_fill = false
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
              let value = payload[source.to_int() + byte_index]
              chunk[destination.to_int() + byte_index] = value
              if value != fill[byte_index] {
                has_non_fill = true
              }
            }
          }
          if has_non_fill {
            let encoded = if filtered {
              let scaleoffset_little_endian = patch_scaleoffset_needs_little_endian_input(
                filters,
                0U,
                metadata.datatype,
                metadata.datatype.size,
              )
              let filter_data = Bytes::from_array(chunk)
              let filter_fill = if scaleoffset_little_endian {
                patch_byteswap_elements(fill, metadata.datatype.size)
              } else {
                fill
              }
              encode_chunk_filters(
                if scaleoffset_little_endian {
                  patch_byteswap_elements(filter_data, metadata.datatype.size)
                } else {
                  filter_data
                },
                write_filters,
                0U,
                metadata.datatype.size,
                values,
                filter_fill,
                self.custom_filters,
              )
            } else {
              Bytes::from_array(chunk)
            }
            guard encoded.length() > 0 else {
              raise InvalidField("allocated chunk encoded to an empty payload")
            }
            let new_offset = append_aligned_chunk_to_tail(
              tail,
              self.bytes.length(),
              encoded,
            )
            let new_address = file_offset_to_stored_address(
              new_offset,
              self.superblock.base_address,
            )
            patch_chunk_address_field(output, record, new_address)
            if record.stored_size_offset >= 0 {
              patch_stored_size_field(
                output,
                record,
                if filtered {
                  encoded.length().to_uint64()
                } else {
                  chunk_size.to_uint64()
                },
              )
            } else if filtered {
              raise InvalidField(
                "filtered chunk index is missing its stored size",
              )
            }
            if filtered {
              patch_chunk_mask_field(output, record, 0U)
            }
            relocated = true
            if record.metadata_start >= 0 &&
              record.metadata_checksum_offset >= 0 {
              let mut known = false
              for region in 0..<checksum_starts.length() {
                if checksum_starts[region] == record.metadata_start &&
                  checksum_offsets[region] == record.metadata_checksum_offset {
                  known = true
                }
              }
              if !known {
                checksum_starts.push(record.metadata_start)
                checksum_offsets.push(record.metadata_checksum_offset)
              }
            }
          }
        } else {
          let chunk_offset = address_to_offset(
            record.address,
            self.superblock.base_address,
          )
          guard record.size > 0UL && record.size <= 0x7FFFFFFFUL else {
            raise InvalidField("invalid stored chunk size")
          }
          guard chunk_offset >= 0 &&
            chunk_offset.to_int64() + record.size.reinterpret_as_int64() <=
            self.bytes.length().to_int64() else {
            raise UnexpectedEof(chunk_offset.to_int64())
          }
          let chunk = if filtered {
            let decoded = read_filtered_chunk(
              self.bytes,
              record.address,
              self.superblock.base_address,
              record.size,
              filters,
              record.mask,
              metadata.datatype.size,
              chunk_size,
            ).to_array()
            guard decoded.length() == chunk_size else {
              raise InvalidField(
                "decoded filtered chunk size does not match the chunk shape",
              )
            }
            decoded
          } else {
            Reader::at(self.bytes, chunk_offset)
            .read_bytes(chunk_size)
            .to_array()
          }
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
          if filtered {
            let scaleoffset_little_endian = patch_scaleoffset_needs_little_endian_input(
              filters,
              record.mask,
              metadata.datatype,
              metadata.datatype.size,
            )
            let filter_data = Bytes::from_array(chunk)
            let filter_fill = if scaleoffset_little_endian {
              patch_byteswap_elements(fill, metadata.datatype.size)
            } else {
              fill
            }
            let encoded = encode_chunk_filters(
              if scaleoffset_little_endian {
                patch_byteswap_elements(filter_data, metadata.datatype.size)
              } else {
                filter_data
              },
              write_filters,
              record.mask,
              metadata.datatype.size,
              values,
              filter_fill,
              self.custom_filters,
            )
            guard encoded.length() > 0 else {
              raise InvalidField("filtered chunk encoded to an empty payload")
            }
            if encoded.length().to_uint64() <= record.size {
              for byte_index in 0..<encoded.length() {
                output[chunk_offset + byte_index] = encoded[byte_index]
              }
            } else {
              let new_offset = append_aligned_chunk_to_tail(
                tail,
                self.bytes.length(),
                encoded,
              )
              let new_address = file_offset_to_stored_address(
                new_offset,
                self.superblock.base_address,
              )
              patch_chunk_address_field(output, record, new_address)
              relocated = true
            }
            if encoded.length().to_uint64() != record.size {
              patch_stored_size_field(
                output,
                record,
                encoded.length().to_uint64(),
              )
              if record.metadata_start >= 0 &&
                record.metadata_checksum_offset >= 0 {
                let mut known = false
                for region in 0..<checksum_starts.length() {
                  if checksum_starts[region] == record.metadata_start &&
                    checksum_offsets[region] == record.metadata_checksum_offset {
                    known = true
                  }
                }
                if !known {
                  checksum_starts.push(record.metadata_start)
                  checksum_offsets.push(record.metadata_checksum_offset)
                }
              }
            }
          } else {
            for byte_index in 0..<chunk_size {
              output[chunk_offset + byte_index] = chunk[byte_index]
            }
          }
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
                "chunk index has no writable slot for a non-fill chunk",
              )
            }
          }
        }
    }
  }
  for region in 0..<checksum_starts.length() {
    patch_metadata_checksum_in_place(
      output,
      checksum_starts[region],
      checksum_offsets[region],
    )
  }
  if relocated {
    return self.finish_appended_image(output, tail)
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
  let metadata = read_dataset_metadata(
    header,
    self.superblock.offset_size,
    self.superblock.length_size,
  )
  let compact = metadata.layout.layout_class == 0
  let contiguous = metadata.layout.layout_class == 1
  let chunked = metadata.layout.layout_class == 2
  guard header.first(msg_filter_pipeline) is None || chunked else {
    raise UnsupportedFeature("filters require chunked storage")
  }
  guard contiguous || (allow_noncontiguous && (compact || chunked)) else {
    if allow_noncontiguous {
      raise UnsupportedFeature(
        "only compact, contiguous and chunked storage can be patched",
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
    self.patch_chunked_dataset_data(header, metadata, payload, values)
  } else {
    self.patch_contiguous_dataset_data(
      header, metadata, payload, allow_noncontiguous,
    )
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
/// Replaces the values of a compact, contiguous or chunked dataset and returns
/// a new byte image. The shape and datatype must match exactly.
///
/// Compact payloads stay inline; only the containing object-header chunk
/// checksum is recomputed. Allocated contiguous payloads stay at their existing
/// address; unallocated payloads are appended at an aligned file offset and
/// linked by updating the layout address and size. Chunk bytes are updated in
/// place and edge padding is preserved. Filtered chunks are decoded and
/// re-encoded with their existing mask and pipeline. Encodings that fit
/// overwrite the original payload; larger encodings are appended and linked by
/// updating the index address and size fields, the superblock EOF address and
/// affected checksums.
///
/// Missing chunks stay unallocated for fill values. A non-fill chunk can be
/// allocated only into a writable slot already present in the chunk index;
/// missing fixed-/extensible-array blocks or pages and B-tree records are not
/// created. External, variable-length and reference storage is rejected, and
/// the dataset shape and chunk grid are never changed.
pub fn Hdf5File::patch_dataset_values(
  self : Hdf5File,
  path : String,
  values : Hdf5WriteValues,
) -> Bytes raise Hdf5Error {
  self.patch_dataset_values_impl(path, values, true)
}
