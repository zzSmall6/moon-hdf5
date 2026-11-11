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
    VariableStringValues(_) => datatype.class == 9 && datatype.vlen_is_string
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
  target_start~ : Int32 = 0,
) -> Unit raise Hdf5Error {
  let offset = record.stored_size_offset - target_start
  guard offset >= 0 && record.stored_size_width >= 1 else {
    raise UnsupportedFeature(
      "filtered chunk index does not expose a writable stored-size field",
    )
  }
  patch_little_endian_uint(
    output,
    offset,
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
  target_start~ : Int32 = 0,
) -> Unit raise Hdf5Error {
  let offset = record.mask_offset - target_start
  guard offset >= 0 else {
    raise UnsupportedFeature(
      "filtered chunk index does not expose a writable filter-mask field",
    )
  }
  patch_little_endian_uint(
    output,
    offset,
    4,
    value.to_uint64(),
    "allocated filtered chunk filter mask",
  )
}

///|
fn initialize_chunk_page(
  output : FixedArray[Byte],
  page : RawChunkPage,
  target_start~ : Int32 = 0,
  bitmap_target_start~ : Int32 = -1,
  bitmap_output~ : FixedArray[Byte]? = None,
) -> Unit raise Hdf5Error {
  let page_start = page.start - target_start
  let checksum_offset = page.checksum_offset - target_start
  let bitmap_target = if bitmap_target_start >= 0 {
    bitmap_target_start
  } else {
    target_start
  }
  let bitmap_offset = page.bitmap_offset - bitmap_target
  let bitmap_metadata_start = page.bitmap_metadata_start - bitmap_target
  let bitmap_checksum_offset = page.bitmap_checksum_offset - bitmap_target
  let bitmap_image = match bitmap_output {
    Some(image) => image
    None => output
  }
  let page_length = page.entries.to_int64() * page.entry_size.to_int64()
  guard page.entries >= 0 &&
    page.entry_size >= page.address_width &&
    (
      page.stored_size_width == 0 ||
      page.address_width + page.stored_size_width + 4 == page.entry_size
    ) &&
    page_start >= 0 &&
    checksum_offset.to_int64() == page_start.to_int64() + page_length &&
    checksum_offset.to_int64() + 4L <= output.length().to_int64() &&
    bitmap_offset >= 0 &&
    bitmap_offset < bitmap_image.length() &&
    bitmap_metadata_start >= 0 &&
    bitmap_checksum_offset >= bitmap_metadata_start &&
    bitmap_checksum_offset.to_int64() + 4L <= bitmap_image.length().to_int64() else {
    raise InvalidField("invalid unallocated chunk-index page")
  }
  let undefined = if page.address_width == 8 {
    0xFFFFFFFFFFFFFFFFUL
  } else {
    (1UL << (page.address_width * 8)) - 1UL
  }
  for entry in 0..<page.entries {
    let offset = page_start + entry * page.entry_size
    patch_little_endian_uint(
      output,
      offset,
      page.address_width,
      undefined,
      "chunk-index page chunk address",
    )
    if page.stored_size_width > 0 {
      patch_little_endian_uint(
        output,
        offset + page.address_width,
        page.stored_size_width,
        0UL,
        "chunk-index page stored size",
      )
      patch_little_endian_uint(
        output,
        offset + page.address_width + page.stored_size_width,
        4,
        0UL,
        "chunk-index page filter mask",
      )
    }
  }
  patch_metadata_checksum_in_place(output, page_start, checksum_offset)
  bitmap_image[bitmap_offset] = (bitmap_image[bitmap_offset].to_int() |
  page.bitmap_mask.to_int()).to_byte()
  patch_metadata_checksum_in_place(
    bitmap_image, bitmap_metadata_start, bitmap_checksum_offset,
  )
}

///|
fn patch_chunk_address_field(
  output : FixedArray[Byte],
  record : RawChunkRecord,
  value : UInt64,
  target_start~ : Int32 = 0,
) -> Unit raise Hdf5Error {
  let offset = record.address_offset - target_start
  guard offset >= 0 && record.address_width >= 1 else {
    raise UnsupportedFeature(
      "filtered chunk index does not expose a writable address field",
    )
  }
  patch_little_endian_uint(
    output,
    offset,
    record.address_width,
    value,
    "relocated filtered chunk address",
  )
}

///|
fn patch_extensible_pointer_in_image(
  output : FixedArray[Byte],
  target_start : Int32,
  pointer_offset : Int32,
  pointer_width : Int32,
  pointer_address : UInt64,
  metadata_start : Int32,
  metadata_checksum_offset : Int32,
) -> Unit raise Hdf5Error {
  let relative_pointer = pointer_offset - target_start
  let relative_metadata = metadata_start - target_start
  let relative_checksum = metadata_checksum_offset - target_start
  guard relative_pointer >= 0 && relative_metadata >= 0 else {
    raise InvalidField("extensible array parent pointer is outside its image")
  }
  patch_little_endian_uint(
    output, relative_pointer, pointer_width, pointer_address, "extensible array child block address",
  )
  patch_metadata_checksum_in_place(output, relative_metadata, relative_checksum)
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
fn next_aligned_tail_start(
  file_length : Int32,
  tail_length : Int32,
  purpose : String,
) -> UInt64 raise Hdf5Error {
  guard file_length >= 0 && file_length.to_int64() <= 0x7FFFFFFFL - 7L else {
    raise UnsupportedFeature("file is too large to " + purpose)
  }
  guard tail_length >= 0 &&
    tail_length.to_int64() <= 0x7FFFFFFFL - file_length.to_int64() else {
    raise UnsupportedFeature("file is too large to " + purpose)
  }
  let unpadded = file_length + tail_length
  guard unpadded.to_int64() <= 0x7FFFFFFFL - 7L else {
    raise UnsupportedFeature("file is too large to " + purpose)
  }
  ((unpadded + 7) / 8 * 8).to_uint64()
}

///|
fn append_aligned_bytes_to_tail(
  tail : @buffer.Buffer,
  file_length : Int32,
  encoded : Bytes,
  purpose : String,
) -> UInt64 raise Hdf5Error {
  let start = next_aligned_tail_start(file_length, tail.length(), purpose)
  let unpadded = file_length + tail.length()
  guard encoded.length().to_int64() <=
    0x7FFFFFFFL - start.reinterpret_as_int64() else {
    raise UnsupportedFeature("file is too large to " + purpose)
  }
  for _ in 0..<(start.to_int() - unpadded) {
    tail.write_byte(b'\x00')
  }
  tail.write_bytes(encoded.get_view().unwrap())
  start
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
        guard filter.parameters.length() == 0 ||
          (
            filter.parameters.length() == 1 &&
            filter.parameters[0] == datatype.size.reinterpret_as_uint()
          ) else {
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
  tail_prefix : Bytes,
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
  let output = self.bytes.to_fixedarray()
  for index in 0..<payload.length() {
    output[offset + index] = payload[index]
  }
  if message.checksum_offset >= 0 {
    patch_metadata_checksum_in_place(
      output,
      message.chunk_start,
      message.checksum_offset,
    )
  }
  if tail_prefix.length() == 0 {
    Bytes::from_array(output)
  } else {
    let tail = @buffer.Buffer::Buffer()
    writer_write_bytes(tail, tail_prefix)
    self.finish_appended_image(output, tail)
  }
}

///|
fn Hdf5File::patch_contiguous_dataset_data(
  self : Hdf5File,
  header : ObjectHeader,
  metadata : DatasetMetadata,
  payload : Bytes,
  allow_allocate : Bool,
  tail_prefix : Bytes,
) -> Bytes raise Hdf5Error {
  let full_size = payload.length()
  let tail = @buffer.Buffer::Buffer()
  writer_write_bytes(tail, tail_prefix)
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
  let output = self.bytes.to_fixedarray()
  for index in 0..<payload.length() {
    output[offset + index] = payload[index]
  }
  if tail.length() > 0 {
    self.finish_appended_image(output, tail)
  } else {
    Bytes::from_array(output)
  }
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
  missing_blocks~ : Array[MissingExtensibleDataBlock]? = None,
  missing_supers~ : Array[MissingExtensibleSuperBlock]? = None,
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
    missing_blocks~,
    missing_supers~,
  )
  |> ignore
  records
}

///|
fn chunk_slot_has_non_fill(
  offsets : Array[UInt64],
  chunk_widths : Array[Int64],
  dataspace_dimensions : Array[UInt64],
  strides : Array[Int64],
  chunk_elements : Int64,
  element_size : Int32,
  payload : Bytes,
  fill : Bytes,
) -> Bool raise Hdf5Error {
  let rank = chunk_widths.length()
  guard offsets.length() == rank &&
    dataspace_dimensions.length() == rank &&
    strides.length() == rank &&
    fill.length() == element_size else {
    raise InvalidField("invalid chunk payload comparison")
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
      if global >= dataspace_dimensions[axis] {
        outside = true
      } else {
        dataset_linear = dataset_linear +
          global.reinterpret_as_int64() * strides[axis]
      }
    }
    if outside {
      continue
    }
    let source = dataset_linear * element_size.to_int64()
    guard source >= 0L &&
      source + element_size.to_int64() <= payload.length().to_int64() else {
      raise InvalidField("chunk payload is smaller than the dataspace")
    }
    for byte_index in 0..<element_size {
      if payload[source.to_int() + byte_index] != fill[byte_index] {
        return true
      }
    }
  }
  false
}

///|
/// Builds and encodes a non-fill chunk for an index slot that has no writable
/// record storage. New chunks use filter mask zero because every filter is
/// applied by this write path.
fn encode_missing_chunk_from_payload(
  offsets : Array[UInt64],
  chunk_widths : Array[Int64],
  dataspace_dimensions : Array[UInt64],
  strides : Array[Int64],
  chunk_elements : Int64,
  element_size : Int32,
  datatype : Datatype,
  filters : Array[Filter],
  payload : Bytes,
  fill : Bytes,
  filtered : Bool,
  write_filters : Array[Hdf5WriteFilter],
  values : Hdf5WriteValues,
  custom_filters : Array[&Hdf5CustomFilter],
) -> (Bytes, Bool) raise Hdf5Error {
  let rank = chunk_widths.length()
  guard offsets.length() == rank &&
    dataspace_dimensions.length() == rank &&
    strides.length() == rank &&
    chunk_elements > 0L &&
    element_size > 0 &&
    fill.length() == element_size &&
    chunk_elements <= 0x7FFFFFFFL / element_size.to_int64() else {
    raise InvalidField("invalid missing chunk payload geometry")
  }
  let chunk_size = (chunk_elements * element_size.to_int64()).to_int()
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
      if global >= dataspace_dimensions[axis] {
        outside = true
      } else {
        dataset_linear = dataset_linear +
          global.reinterpret_as_int64() * strides[axis]
      }
    }
    if outside {
      continue
    }
    let source = dataset_linear * element_size.to_int64()
    let destination = chunk_element.to_int64() * element_size.to_int64()
    guard source >= 0L &&
      source + element_size.to_int64() <= payload.length().to_int64() else {
      raise InvalidField("chunk payload is smaller than the dataspace")
    }
    for byte_index in 0..<element_size {
      let value = payload[source.to_int() + byte_index]
      chunk[destination.to_int() + byte_index] = value
      if value != fill[byte_index] {
        has_non_fill = true
      }
    }
  }
  if !has_non_fill {
    return (b"", false)
  }
  if !filtered {
    return (Bytes::from_array(chunk), has_non_fill)
  }
  let scaleoffset_little_endian = patch_scaleoffset_needs_little_endian_input(
    filters, 0U, datatype, element_size,
  )
  let filter_data = Bytes::from_array(chunk)
  let filter_fill = if scaleoffset_little_endian {
    patch_byteswap_elements(fill, element_size)
  } else {
    fill
  }
  let encoded = encode_chunk_filters(
    if scaleoffset_little_endian {
      patch_byteswap_elements(filter_data, element_size)
    } else {
      filter_data
    },
    write_filters,
    0U,
    element_size,
    values,
    filter_fill,
    custom_filters,
  )
  guard encoded.length() > 0 else {
    raise InvalidField("allocated chunk encoded to an empty payload")
  }
  (encoded, has_non_fill)
}

///|
fn missing_extensible_block_used(
  descriptor : MissingExtensibleDataBlock,
  layout : DataLayout,
  dataspace : Dataspace,
  chunk_widths : Array[Int64],
  grid : Array[Int64],
  strides : Array[Int64],
  chunk_elements : Int64,
  element_size : Int32,
  payload : Bytes,
  fill : Bytes,
) -> Bool raise Hdf5Error {
  missing_extensible_range_used(
    descriptor.start,
    descriptor.count,
    descriptor.limit,
    descriptor.order,
    descriptor.grid,
    layout,
    dataspace,
    chunk_widths,
    grid,
    strides,
    chunk_elements,
    element_size,
    payload,
    fill,
  )
}

///|
fn missing_extensible_range_used(
  start : Int64,
  count : Int64,
  limit : Int64,
  order : Array[Int32],
  slot_grid : Array[Int64],
  layout : DataLayout,
  dataspace : Dataspace,
  chunk_widths : Array[Int64],
  grid : Array[Int64],
  strides : Array[Int64],
  chunk_elements : Int64,
  element_size : Int32,
  payload : Bytes,
  fill : Bytes,
) -> Bool raise Hdf5Error {
  let rank = chunk_widths.length()
  guard order.length() == rank &&
    slot_grid.length() == rank &&
    grid.length() == rank else {
    raise InvalidField("invalid extensible array slot geometry")
  }
  let end = if start + count < limit { start + count } else { limit }
  let mut slot = start
  while slot < end {
    let offsets = extensible_slot_offsets(
      slot,
      order,
      slot_grid,
      layout.chunk_dimensions,
    )
    let mut outside = false
    for axis in 0..<rank {
      if offsets[axis] >= dataspace.dimensions[axis] {
        outside = true
      }
    }
    if !outside &&
      chunk_slot_has_non_fill(
        offsets,
        chunk_widths,
        dataspace.dimensions,
        strides,
        chunk_elements,
        element_size,
        payload,
        fill,
      ) {
      return true
    }
    slot = slot + 1L
  }
  false
}

///|
/// Returns one past the highest extensible-array slot with non-fill values, or
/// zero when the whole payload is fill. This is used when rebuilding an EAIB
/// whose header also lost its maximum-set watermark.
fn highest_non_fill_extensible_slot(
  order : Array[Int32],
  slot_grid : Array[Int64],
  layout : DataLayout,
  dataspace : Dataspace,
  chunk_widths : Array[Int64],
  grid : Array[Int64],
  strides : Array[Int64],
  chunk_elements : Int64,
  element_size : Int32,
  payload : Bytes,
  fill : Bytes,
) -> Int64 raise Hdf5Error {
  let rank = chunk_widths.length()
  guard order.length() == rank &&
    slot_grid.length() == rank &&
    grid.length() == rank else {
    raise InvalidField("invalid extensible array slot geometry")
  }
  let slots = product(slot_grid)
  guard slots >= 0L else {
    raise InvalidField("extensible array slot geometry is too large")
  }
  let mut highest = 0L
  for slot in 0L..<slots {
    let offsets = extensible_slot_offsets(
      slot,
      order,
      slot_grid,
      layout.chunk_dimensions,
    )
    let mut outside = false
    for axis in 0..<rank {
      if offsets[axis] >= dataspace.dimensions[axis] {
        outside = true
      }
    }
    if !outside &&
      chunk_slot_has_non_fill(
        offsets,
        chunk_widths,
        dataspace.dimensions,
        strides,
        chunk_elements,
        element_size,
        payload,
        fill,
      ) {
      highest = slot + 1L
    }
  }
  highest
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
fn Hdf5File::btree_v1_k(self : Hdf5File) -> Int32 raise Hdf5Error {
  let mut value = self.superblock.indexed_storage_node_k.to_int()
  let undefined = heap_undefined_address(self.superblock.offset_size)
  if self.superblock.version >= 2 &&
    self.superblock.superblock_extension_address != undefined {
    let extension = self.object_header_at(
      self.superblock.superblock_extension_address,
    )
    if extension.first(msg_btree_k) is Some(message) {
      let reader = Reader::new(message.data)
      guard reader.remaining() >= 7 else {
        raise InvalidField("B-tree K message is too short")
      }
      guard reader.read_u8() == b'\x00' else {
        raise UnsupportedFeature("unsupported B-tree K message version")
      }
      value = reader.read_u16_le().to_int()
    }
  }
  guard value > 0 else {
    raise InvalidField("indexed storage B-tree K must be positive")
  }
  value
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
    page: None,
    missing_block_index: -1,
  }
}

///|
fn Hdf5File::patch_chunked_dataset_data(
  self : Hdf5File,
  header : ObjectHeader,
  metadata : DatasetMetadata,
  payload : Bytes,
  values : Hdf5WriteValues,
  tail_prefix : Bytes,
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
  let mut missing_fixed_array : MissingFixedArrayStorage? = None
  let missing_extensible_blocks : Array[MissingExtensibleDataBlock] = []
  let missing_extensible_supers : Array[MissingExtensibleSuperBlock] = []
  let extensible_config : ExtensibleArrayConfig? = match layout.chunk_index {
    ExtensibleArray(config) => Some(config)
    _ => None
  }
  let btree_v2_config : BTreeV2Config? = match layout.chunk_index {
    BTreeV2(config) => Some(config)
    _ => None
  }
  let is_btree_v1 = layout.chunk_index is BTreeV1
  let is_btree_v2 = btree_v2_config is Some(_)
  let extensible_header_info : ExtensibleArrayHeaderInfo? = match
    extensible_config {
    Some(_) =>
      Some(
        read_extensible_array_header_info(
          self.bytes,
          layout,
          self.superblock.offset_size,
          self.superblock.length_size,
          self.superblock.base_address,
        ),
      )
    None => None
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
    FixedArray(page_bits) => {
      let records = collect_raw_chunk_records(
        self.bytes,
        layout,
        metadata.dataspace,
        metadata.datatype,
        filters,
        self.superblock.offset_size,
        self.superblock.length_size,
        self.superblock.base_address,
      )
      if records.length() == 0 && layout.data_address != undefined {
        guard tail_prefix.length() == 0 else {
          raise UnsupportedFeature(
            "variable-length dataset patching cannot create a fixed-array data block",
          )
        }
        let block_start = (self.bytes.length() + 7) / 8 * 8
        let storage = prepare_missing_fixed_array_storage(
          self.bytes,
          layout,
          metadata.dataspace,
          metadata.datatype,
          self.superblock.offset_size,
          self.superblock.length_size,
          self.superblock.base_address,
          filters,
          undefined,
          page_bits,
          block_start,
        )
        missing_fixed_array = Some(storage)
        storage.records
      } else {
        records
      }
    }
    ExtensibleArray(_) =>
      collect_raw_chunk_records(
        self.bytes,
        layout,
        metadata.dataspace,
        metadata.datatype,
        filters,
        self.superblock.offset_size,
        self.superblock.length_size,
        self.superblock.base_address,
        missing_blocks=Some(missing_extensible_blocks),
        missing_supers=Some(missing_extensible_supers),
      )
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
  let mut index_output = match missing_fixed_array {
    Some(storage) => storage.image
    None => output
  }
  let mut index_start = match missing_fixed_array {
    Some(storage) => storage.block_start
    None => 0
  }
  let mut chunk_tail_base = match missing_fixed_array {
    Some(storage) => storage.block_start + storage.image.length()
    None => self.bytes.length()
  }
  let missing_extensible_storage : Array[MissingExtensibleBlockStorage] = []
  let extensible_index_records : Array[RawChunkRecord] = []
  let appended_metadata_images : Array[FixedArray[Byte]] = []
  let appended_metadata_starts : Array[Int32] = []
  let mut next_extensible_start = (self.bytes.length() + 7) / 8 * 8
  let mut created_index = false
  let mut new_secondary_block_count = 0L
  let mut new_secondary_block_size = 0L
  let mut new_data_block_count = 0L
  let mut new_data_block_size = 0L
  let mut new_realized_elements = 0L
  match extensible_config {
    Some(config) => {
      let info = match extensible_header_info {
        Some(info) => info
        None => raise InvalidField("extensible array header is unavailable")
      }
      guard info.config == config &&
        (
          (info.client == 0 && filters.length() == 0) ||
          (info.client == 1 && filters.length() > 0)
        ) else {
        raise InvalidField("extensible array header does not match layout")
      }
      let (slot_order, slot_grid, existing_limit) = extensible_array_geometry(
        layout,
        metadata.dataspace,
        self.superblock.length_size,
        info.maximum_set,
      )
      let mut effective_maximum_set = info.maximum_set
      if info.index_address == undefined {
        let required = highest_non_fill_extensible_slot(
          slot_order,
          slot_grid,
          layout,
          metadata.dataspace,
          chunk_widths,
          grid,
          strides,
          chunk_elements,
          metadata.datatype.size,
          payload,
          fill,
        )
        if required > existing_limit {
          effective_maximum_set = required.reinterpret_as_uint64()
        }
      }
      let (slot_order, slot_grid, limit) = extensible_array_geometry(
        layout,
        metadata.dataspace,
        self.superblock.length_size,
        effective_maximum_set,
      )
      if info.index_address == undefined &&
        limit > 0L &&
        missing_extensible_range_used(
          0L,
          limit,
          limit,
          slot_order,
          slot_grid,
          layout,
          metadata.dataspace,
          chunk_widths,
          grid,
          strides,
          chunk_elements,
          metadata.datatype.size,
          payload,
          fill,
        ) {
        let storage = prepare_missing_extensible_index(
          self.bytes,
          layout,
          metadata.dataspace,
          metadata.datatype,
          self.superblock.offset_size,
          self.superblock.length_size,
          self.superblock.base_address,
          config,
          info.client,
          info.entry_size,
          filters,
          undefined,
          effective_maximum_set,
          next_extensible_start,
        )
        created_index = true
        index_output = storage.image
        index_start = storage.block_start
        appended_metadata_images.push(storage.image)
        appended_metadata_starts.push(storage.block_start)
        for record in storage.records {
          extensible_index_records.push(record)
        }
        patch_little_endian_uint(
          output,
          info.index_address_offset,
          info.index_address_width,
          file_offset_to_stored_address(
            storage.block_start.to_uint64(),
            self.superblock.base_address,
          ),
          "extensible array index address",
        )
        if effective_maximum_set > info.maximum_set {
          patch_little_endian_uint(
            output,
            info.maximum_set_offset,
            self.superblock.length_size,
            effective_maximum_set,
            "extensible array maximum set",
          )
        }
        new_realized_elements = config.index_elements.to_int64()
        next_extensible_start = (
            storage.block_start + storage.image.length() + 7
          ) /
          8 *
          8
        let mut level_start = 0L
        let mut direct_index = 0
        for level in 0..<storage.levels {
          if config.index_elements.to_int64() + level_start >= limit {
            break
          }
          let blocks = 1L << (level / 2)
          let count = config.min_elements.to_int64() << ((level + 1) / 2)
          if level < storage.first_super {
            for block_index in 0..<blocks.to_int() {
              let descriptor : MissingExtensibleDataBlock = {
                count,
                start: config.index_elements.to_int64() +
                level_start +
                block_index.to_int64() * count,
                limit,
                block_offset: level_start +
                (direct_index.to_int64() + block_index.to_int64()) * count,
                order: slot_order,
                grid: slot_grid,
                pointer_offset: storage.block_start +
                storage.direct_start +
                (direct_index + block_index) * self.superblock.offset_size,
                pointer_width: self.superblock.offset_size,
                pointer_metadata_start: storage.block_start,
                pointer_checksum_offset: storage.block_start +
                storage.index_content,
                bitmap_start: 0L,
                bitmap_offset: -1,
                bitmap_metadata_start: -1,
                bitmap_checksum_offset: -1,
                client: info.client,
                entry_size: info.entry_size,
              }
              if missing_extensible_block_used(
                  descriptor,
                  layout,
                  metadata.dataspace,
                  chunk_widths,
                  grid,
                  strides,
                  chunk_elements,
                  metadata.datatype.size,
                  payload,
                  fill,
                ) {
                let child = prepare_missing_extensible_data_block(
                  self.bytes,
                  layout,
                  metadata.dataspace,
                  metadata.datatype,
                  self.superblock.offset_size,
                  self.superblock.base_address,
                  config,
                  filters,
                  undefined,
                  descriptor,
                  missing_extensible_storage.length(),
                  next_extensible_start,
                  index_output,
                  index_start,
                )
                missing_extensible_storage.push(child)
                appended_metadata_images.push(child.image)
                appended_metadata_starts.push(child.block_start)
                next_extensible_start = (
                    child.block_start + child.image.length() + 7
                  ) /
                  8 *
                  8
                new_data_block_count = new_data_block_count + 1L
                new_data_block_size = new_data_block_size +
                  child.image.length().to_int64()
                new_realized_elements = new_realized_elements + count
              }
            }
            direct_index = direct_index + blocks.to_int()
          } else {
            let page_size = 1L << config.page_bits
            let pages = if count > page_size { count / page_size } else { 0L }
            let bitmap_size = blocks * ((pages + 7L) / 8L)
            let super_content = 6L +
              self.superblock.offset_size.to_int64() +
              ((config.max_bits + 7) / 8).to_int64() +
              bitmap_size +
              blocks * self.superblock.offset_size.to_int64()
            let candidate : MissingExtensibleSuperBlock = {
              level_start,
              count,
              blocks,
              pages,
              bitmap_size,
              super_content,
              pointer_offset: storage.block_start +
              storage.super_start +
              (level - storage.first_super) * self.superblock.offset_size,
              pointer_width: self.superblock.offset_size,
              pointer_metadata_start: storage.block_start,
              pointer_checksum_offset: storage.block_start +
              storage.index_content,
              order: slot_order,
              grid: slot_grid,
              client: info.client,
              entry_size: info.entry_size,
              limit,
            }
            let used_blocks : Array[Int32] = []
            for block_index in 0..<blocks.to_int() {
              if missing_extensible_range_used(
                  config.index_elements.to_int64() +
                  level_start +
                  block_index.to_int64() * count,
                  count,
                  limit,
                  slot_order,
                  slot_grid,
                  layout,
                  metadata.dataspace,
                  chunk_widths,
                  grid,
                  strides,
                  chunk_elements,
                  metadata.datatype.size,
                  payload,
                  fill,
                ) {
                used_blocks.push(block_index)
              }
            }
            if used_blocks.length() > 0 {
              let super_storage = prepare_missing_extensible_super_block(
                layout,
                self.superblock.offset_size,
                config,
                candidate,
                next_extensible_start,
              )
              appended_metadata_images.push(super_storage.image)
              appended_metadata_starts.push(super_storage.block_start)
              next_extensible_start = (
                  super_storage.block_start + super_storage.image.length() + 7
                ) /
                8 *
                8
              patch_extensible_pointer_in_image(
                index_output,
                index_start,
                candidate.pointer_offset,
                candidate.pointer_width,
                file_offset_to_stored_address(
                  super_storage.block_start.to_uint64(),
                  self.superblock.base_address,
                ),
                candidate.pointer_metadata_start,
                candidate.pointer_checksum_offset,
              )
              new_secondary_block_count = new_secondary_block_count + 1L
              new_secondary_block_size = new_secondary_block_size +
                super_storage.image.length().to_int64()
              for block_index in used_blocks {
                let descriptor : MissingExtensibleDataBlock = {
                  count,
                  start: config.index_elements.to_int64() +
                  level_start +
                  block_index.to_int64() * count,
                  limit,
                  block_offset: level_start + block_index.to_int64() * count,
                  order: slot_order,
                  grid: slot_grid,
                  pointer_offset: super_storage.block_pointer_offsets[block_index],
                  pointer_width: self.superblock.offset_size,
                  pointer_metadata_start: super_storage.block_start,
                  pointer_checksum_offset: super_storage.bitmap_checksum_offset,
                  bitmap_start: block_index.to_int64() * pages,
                  bitmap_offset: super_storage.bitmap_offset,
                  bitmap_metadata_start: super_storage.bitmap_metadata_start,
                  bitmap_checksum_offset: super_storage.bitmap_checksum_offset,
                  client: info.client,
                  entry_size: info.entry_size,
                }
                let child = prepare_missing_extensible_data_block(
                  self.bytes,
                  layout,
                  metadata.dataspace,
                  metadata.datatype,
                  self.superblock.offset_size,
                  self.superblock.base_address,
                  config,
                  filters,
                  undefined,
                  descriptor,
                  missing_extensible_storage.length(),
                  next_extensible_start,
                  super_storage.image,
                  super_storage.block_start,
                )
                missing_extensible_storage.push(child)
                appended_metadata_images.push(child.image)
                appended_metadata_starts.push(child.block_start)
                next_extensible_start = (
                    child.block_start + child.image.length() + 7
                  ) /
                  8 *
                  8
                new_data_block_count = new_data_block_count + 1L
                new_data_block_size = new_data_block_size +
                  child.image.length().to_int64()
                new_realized_elements = new_realized_elements + count
              }
            }
          }
          level_start = level_start + blocks * count
        }
      }
      for descriptor in missing_extensible_supers {
        let used_blocks : Array[Int32] = []
        for block_index in 0..<descriptor.blocks.to_int() {
          if missing_extensible_range_used(
              config.index_elements.to_int64() +
              descriptor.level_start +
              block_index.to_int64() * descriptor.count,
              descriptor.count,
              descriptor.limit,
              descriptor.order,
              descriptor.grid,
              layout,
              metadata.dataspace,
              chunk_widths,
              grid,
              strides,
              chunk_elements,
              metadata.datatype.size,
              payload,
              fill,
            ) {
            used_blocks.push(block_index)
          }
        }
        if used_blocks.length() > 0 {
          let super_storage = prepare_missing_extensible_super_block(
            layout,
            self.superblock.offset_size,
            config,
            descriptor,
            next_extensible_start,
          )
          appended_metadata_images.push(super_storage.image)
          appended_metadata_starts.push(super_storage.block_start)
          next_extensible_start = (
              super_storage.block_start + super_storage.image.length() + 7
            ) /
            8 *
            8
          patch_extensible_pointer_in_image(
            index_output,
            index_start,
            descriptor.pointer_offset,
            descriptor.pointer_width,
            file_offset_to_stored_address(
              super_storage.block_start.to_uint64(),
              self.superblock.base_address,
            ),
            descriptor.pointer_metadata_start,
            descriptor.pointer_checksum_offset,
          )
          new_secondary_block_count = new_secondary_block_count + 1L
          new_secondary_block_size = new_secondary_block_size +
            super_storage.image.length().to_int64()
          for block_index in used_blocks {
            let child_descriptor : MissingExtensibleDataBlock = {
              count: descriptor.count,
              start: config.index_elements.to_int64() +
              descriptor.level_start +
              block_index.to_int64() * descriptor.count,
              limit: descriptor.limit,
              block_offset: descriptor.level_start +
              block_index.to_int64() * descriptor.count,
              order: descriptor.order,
              grid: descriptor.grid,
              pointer_offset: super_storage.block_pointer_offsets[block_index],
              pointer_width: self.superblock.offset_size,
              pointer_metadata_start: super_storage.block_start,
              pointer_checksum_offset: super_storage.bitmap_checksum_offset,
              bitmap_start: block_index.to_int64() * descriptor.pages,
              bitmap_offset: super_storage.bitmap_offset,
              bitmap_metadata_start: super_storage.bitmap_metadata_start,
              bitmap_checksum_offset: super_storage.bitmap_checksum_offset,
              client: descriptor.client,
              entry_size: descriptor.entry_size,
            }
            let child = prepare_missing_extensible_data_block(
              self.bytes,
              layout,
              metadata.dataspace,
              metadata.datatype,
              self.superblock.offset_size,
              self.superblock.base_address,
              config,
              filters,
              undefined,
              child_descriptor,
              missing_extensible_storage.length(),
              next_extensible_start,
              super_storage.image,
              super_storage.block_start,
            )
            missing_extensible_storage.push(child)
            appended_metadata_images.push(child.image)
            appended_metadata_starts.push(child.block_start)
            next_extensible_start = (
                child.block_start + child.image.length() + 7
              ) /
              8 *
              8
            new_data_block_count = new_data_block_count + 1L
            new_data_block_size = new_data_block_size +
              child.image.length().to_int64()
            new_realized_elements = new_realized_elements + descriptor.count
          }
        }
      }
      for descriptor in missing_extensible_blocks {
        if missing_extensible_block_used(
            descriptor,
            layout,
            metadata.dataspace,
            chunk_widths,
            grid,
            strides,
            chunk_elements,
            metadata.datatype.size,
            payload,
            fill,
          ) {
          let storage = prepare_missing_extensible_data_block(
            self.bytes,
            layout,
            metadata.dataspace,
            metadata.datatype,
            self.superblock.offset_size,
            self.superblock.base_address,
            config,
            filters,
            undefined,
            descriptor,
            missing_extensible_storage.length(),
            next_extensible_start,
            index_output,
            index_start,
          )
          missing_extensible_storage.push(storage)
          appended_metadata_images.push(storage.image)
          appended_metadata_starts.push(storage.block_start)
          next_extensible_start = (
              storage.block_start + storage.image.length() + 7
            ) /
            8 *
            8
          new_data_block_count = new_data_block_count + 1L
          new_data_block_size = new_data_block_size +
            storage.image.length().to_int64()
          new_realized_elements = new_realized_elements + descriptor.count
        }
      }
      if created_index ||
        new_secondary_block_count > 0L ||
        new_data_block_count > 0L {
        patch_little_endian_uint(
          output,
          info.secondary_block_count_offset,
          self.superblock.length_size,
          info.secondary_block_count +
          new_secondary_block_count.reinterpret_as_uint64(),
          "extensible array secondary block count",
        )
        patch_little_endian_uint(
          output,
          info.secondary_block_size_offset,
          self.superblock.length_size,
          info.secondary_block_size +
          new_secondary_block_size.reinterpret_as_uint64(),
          "extensible array secondary block size",
        )
        patch_little_endian_uint(
          output,
          info.data_block_count_offset,
          self.superblock.length_size,
          info.data_block_count + new_data_block_count.reinterpret_as_uint64(),
          "extensible array data block count",
        )
        patch_little_endian_uint(
          output,
          info.data_block_size_offset,
          self.superblock.length_size,
          info.data_block_size + new_data_block_size.reinterpret_as_uint64(),
          "extensible array data block size",
        )
        patch_little_endian_uint(
          output,
          info.realized_elements_offset,
          info.realized_elements_width,
          info.realized_elements + new_realized_elements.reinterpret_as_uint64(),
          "extensible array realized element count",
        )
        patch_metadata_checksum_in_place(
          output,
          address_to_offset(layout.data_address, self.superblock.base_address),
          info.header_checksum_offset,
        )
      }
    }
    None => ()
  }
  if appended_metadata_images.length() > 0 {
    chunk_tail_base = next_extensible_start
  }
  let all_records : Array[RawChunkRecord] = []
  for record in records {
    all_records.push(record)
  }
  for storage in missing_extensible_storage {
    for record in storage.records {
      all_records.push(record)
    }
  }
  for record in extensible_index_records {
    all_records.push(record)
  }
  let record_for_chunk : Array[RawChunkRecord?] = Array::make(
    total_chunks.to_int(),
    None,
  )
  let v1_present = Array::make(total_chunks.to_int(), false)
  let v1_addresses = Array::make(total_chunks.to_int(), undefined)
  let v1_sizes = Array::make(total_chunks.to_int(), 0UL)
  let v1_masks = Array::make(total_chunks.to_int(), 0U)
  let v2_present = Array::make(total_chunks.to_int(), false)
  let v2_addresses = Array::make(total_chunks.to_int(), undefined)
  let v2_sizes = Array::make(total_chunks.to_int(), 0UL)
  let v2_masks = Array::make(total_chunks.to_int(), 0U)
  for record in all_records {
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
    if is_btree_v1 {
      v1_present[chunk_linear.to_int()] = true
      v1_addresses[chunk_linear.to_int()] = record.address
      v1_sizes[chunk_linear.to_int()] = record.size
      v1_masks[chunk_linear.to_int()] = record.mask
    } else if is_btree_v2 {
      v2_present[chunk_linear.to_int()] = true
      v2_addresses[chunk_linear.to_int()] = record.address
      v2_sizes[chunk_linear.to_int()] = record.size
      v2_masks[chunk_linear.to_int()] = record.mask
    }
  }
  let tail = @buffer.Buffer::Buffer()
  writer_write_bytes(tail, tail_prefix)
  let mut relocated = false
  let mut rebuild_v1_tree = false
  let mut rebuild_v2_tree = false
  let allocated_pages : Array[Int32] = []
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
        let record_output = if record.missing_block_index >= 0 {
          guard record.missing_block_index < missing_extensible_storage.length() else {
            raise InvalidField(
              "missing extensible array block index is invalid",
            )
          }
          missing_extensible_storage[record.missing_block_index].image
        } else {
          index_output
        }
        let record_start = if record.missing_block_index >= 0 {
          missing_extensible_storage[record.missing_block_index].block_start
        } else {
          index_start
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
            match record.page {
              Some(page) =>
                if !allocated_pages.contains(page.start) {
                  initialize_chunk_page(
                    record_output,
                    page,
                    target_start=record_start,
                    bitmap_target_start=if record.missing_block_index >= 0 {
                      missing_extensible_storage[record.missing_block_index].pointer_target_start
                    } else {
                      index_start
                    },
                    bitmap_output=if record.missing_block_index >= 0 {
                      Some(
                        missing_extensible_storage[record.missing_block_index].pointer_output,
                      )
                    } else {
                      None
                    },
                  )
                  allocated_pages.push(page.start)
                }
              None => ()
            }
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
              tail, chunk_tail_base, encoded,
            )
            let new_address = file_offset_to_stored_address(
              new_offset,
              self.superblock.base_address,
            )
            patch_chunk_address_field(
              record_output,
              record,
              new_address,
              target_start=record_start,
            )
            if record.stored_size_offset >= 0 {
              patch_stored_size_field(
                record_output,
                record,
                if filtered {
                  encoded.length().to_uint64()
                } else {
                  chunk_size.to_uint64()
                },
                target_start=record_start,
              )
            } else if filtered {
              raise InvalidField(
                "filtered chunk index is missing its stored size",
              )
            }
            if filtered {
              patch_chunk_mask_field(
                record_output,
                record,
                0U,
                target_start=record_start,
              )
            }
            relocated = true
            if record.missing_block_index >= 0 {
              let storage = missing_extensible_storage[record.missing_block_index]
              patch_metadata_checksum_in_place(
                record_output,
                record.metadata_start - record_start,
                record.metadata_checksum_offset - record_start,
              )
              patch_extensible_pointer_in_image(
                storage.pointer_output,
                storage.pointer_target_start,
                storage.pointer_offset,
                storage.pointer_width,
                file_offset_to_stored_address(
                  storage.block_start.to_uint64(),
                  self.superblock.base_address,
                ),
                storage.pointer_metadata_start,
                storage.pointer_checksum_offset,
              )
            } else if record.metadata_start >= 0 &&
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
            let final_address = if encoded.length().to_uint64() <= record.size {
              for byte_index in 0..<encoded.length() {
                output[chunk_offset + byte_index] = encoded[byte_index]
              }
              record.address
            } else {
              let new_offset = append_aligned_chunk_to_tail(
                tail, chunk_tail_base, encoded,
              )
              let new_address = file_offset_to_stored_address(
                new_offset,
                self.superblock.base_address,
              )
              patch_chunk_address_field(
                index_output,
                record,
                new_address,
                target_start=index_start,
              )
              relocated = true
              new_address
            }
            if encoded.length().to_uint64() != record.size {
              if is_btree_v2 &&
                btree_integer_width(encoded.length().to_uint64()) >
                record.stored_size_width {
                rebuild_v2_tree = true
              } else {
                patch_stored_size_field(
                  index_output,
                  record,
                  encoded.length().to_uint64(),
                  target_start=index_start,
                )
                if record.metadata_start >= 0 &&
                  record.metadata_checksum_offset >= 0 {
                  let mut known = false
                  for region in 0..<checksum_starts.length() {
                    if checksum_starts[region] == record.metadata_start &&
                      checksum_offsets[region] ==
                      record.metadata_checksum_offset {
                      known = true
                    }
                  }
                  if !known {
                    checksum_starts.push(record.metadata_start)
                    checksum_offsets.push(record.metadata_checksum_offset)
                  }
                }
              }
            }
            if is_btree_v1 {
              v1_addresses[chunk_linear] = final_address
              v1_sizes[chunk_linear] = encoded.length().to_uint64()
              v1_masks[chunk_linear] = record.mask
            } else if is_btree_v2 {
              v2_addresses[chunk_linear] = final_address
              v2_sizes[chunk_linear] = encoded.length().to_uint64()
              v2_masks[chunk_linear] = record.mask
            }
          } else {
            for byte_index in 0..<chunk_size {
              output[chunk_offset + byte_index] = chunk[byte_index]
            }
          }
        }
      }
      None =>
        if is_btree_v1 || is_btree_v2 {
          let (encoded, has_non_fill) = encode_missing_chunk_from_payload(
            offsets,
            chunk_widths,
            metadata.dataspace.dimensions,
            strides,
            chunk_elements,
            metadata.datatype.size,
            metadata.datatype,
            filters,
            payload,
            fill,
            filters.length() > 0,
            write_filters,
            values,
            self.custom_filters,
          )
          if has_non_fill {
            let new_offset = append_aligned_chunk_to_tail(
              tail, chunk_tail_base, encoded,
            )
            let new_address = file_offset_to_stored_address(
              new_offset,
              self.superblock.base_address,
            )
            if is_btree_v1 {
              v1_present[chunk_linear] = true
              v1_addresses[chunk_linear] = new_address
              v1_sizes[chunk_linear] = encoded.length().to_uint64()
              v1_masks[chunk_linear] = 0U
              rebuild_v1_tree = true
            } else {
              v2_present[chunk_linear] = true
              v2_addresses[chunk_linear] = new_address
              v2_sizes[chunk_linear] = encoded.length().to_uint64()
              v2_masks[chunk_linear] = 0U
              rebuild_v2_tree = true
            }
            relocated = true
          }
        } else {
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
  }
  for region in 0..<checksum_starts.length() {
    patch_metadata_checksum_in_place(
      index_output,
      checksum_starts[region] - index_start,
      checksum_offsets[region] - index_start,
    )
  }
  if is_btree_v1 && rebuild_v1_tree {
    let capacity = self.btree_v1_k() * 2
    let tree_records : Array[BTreeV1ChunkRecord] = []
    for chunk_linear in 0..<total_chunks.to_int() {
      if v1_present[chunk_linear] {
        tree_records.push({
          address: v1_addresses[chunk_linear],
          size: v1_sizes[chunk_linear],
          mask: v1_masks[chunk_linear],
          offsets: btree_v1_chunk_offsets(chunk_linear, grid, chunk_widths),
        })
      }
    }
    let tree_start_offset = next_aligned_tail_start(
      self.bytes.length(),
      tail.length(),
      "rebuild the v1 B-tree chunk index",
    )
    let tree_start_address = file_offset_to_stored_address(
      tree_start_offset,
      self.superblock.base_address,
    )
    let tree = encode_btree_v1_chunk_tree(
      tree_start_address,
      tree_records,
      chunk_widths,
      metadata.datatype.size,
      self.superblock.offset_size,
      capacity,
    )
    let appended_start = append_aligned_bytes_to_tail(
      tail,
      self.bytes.length(),
      tree,
      "rebuild the v1 B-tree chunk index",
    )
    guard appended_start == tree_start_offset else {
      raise InvalidField("v1 B-tree placement changed during encoding")
    }
    guard layout.data_address_offset >= 0 && layout.data_address_width >= 1 else {
      raise UnsupportedFeature(
        "v1 B-tree layout does not expose a writable address field",
      )
    }
    patch_little_endian_uint(
      output,
      layout.data_address_offset,
      layout.data_address_width,
      tree_start_address,
      "v1 B-tree chunk index address",
    )
    let layout_message = match header.first(msg_data_layout) {
      Some(message) => message
      None =>
        raise InvalidField(
          "dataset is missing a required object header message",
        )
    }
    if layout_message.checksum_offset >= 0 {
      patch_metadata_checksum_in_place(
        output,
        layout_message.chunk_start,
        layout_message.checksum_offset,
      )
    }
    relocated = true
  }
  if is_btree_v2 && rebuild_v2_tree {
    let config = match btree_v2_config {
      Some(config) => config
      None => raise InvalidField("v2 B-tree configuration is unavailable")
    }
    let filtered = filters.length() > 0
    let mut maximum_stored = 0UL
    for chunk_linear in 0..<total_chunks.to_int() {
      if v2_present[chunk_linear] && v2_sizes[chunk_linear] > maximum_stored {
        maximum_stored = v2_sizes[chunk_linear]
      }
    }
    let size_width = if filtered {
      btree_v2_stored_size_width(maximum_stored)
    } else {
      0
    }
    let record_size = self.superblock.offset_size +
      (if filtered { size_width + 4 } else { 0 }) +
      8 * rank
    let encoded_records : Array[Bytes] = []
    for chunk_linear in 0..<total_chunks.to_int() {
      if v2_present[chunk_linear] {
        let coordinates = btree_v2_coordinates(chunk_linear, grid)
        encoded_records.push(
          encode_btree_v2_chunk_record(
            v2_addresses[chunk_linear],
            v2_sizes[chunk_linear],
            v2_masks[chunk_linear],
            coordinates,
            filtered,
            size_width,
            self.superblock.offset_size,
          ),
        )
      }
    }
    let tree_start_offset = next_aligned_tail_start(
      self.bytes.length(),
      tail.length(),
      "rebuild the v2 B-tree chunk index",
    )
    let tree_start_address = file_offset_to_stored_address(
      tree_start_offset,
      self.superblock.base_address,
    )
    let tree = encode_btree_v2_chunk_tree(
      tree_start_address,
      encoded_records,
      config,
      self.superblock.offset_size,
      self.superblock.length_size,
      record_size,
      if filtered {
        11
      } else {
        10
      },
    )
    let appended_start = append_aligned_bytes_to_tail(
      tail,
      self.bytes.length(),
      tree,
      "rebuild the v2 B-tree chunk index",
    )
    guard appended_start == tree_start_offset else {
      raise InvalidField("v2 B-tree placement changed during encoding")
    }
    guard layout.data_address_offset >= 0 && layout.data_address_width >= 1 else {
      raise UnsupportedFeature(
        "v2 B-tree layout does not expose a writable address field",
      )
    }
    patch_little_endian_uint(
      output,
      layout.data_address_offset,
      layout.data_address_width,
      tree_start_address,
      "v2 B-tree chunk index address",
    )
    let layout_message = match header.first(msg_data_layout) {
      Some(message) => message
      None =>
        raise InvalidField(
          "dataset is missing a required object header message",
        )
    }
    if layout_message.checksum_offset >= 0 {
      patch_metadata_checksum_in_place(
        output,
        layout_message.chunk_start,
        layout_message.checksum_offset,
      )
    }
    relocated = true
  }
  guard tail_prefix.length() == 0 || appended_metadata_images.length() == 0 else {
    raise UnsupportedFeature(
      "variable-length dataset patching cannot create extensible-array metadata",
    )
  }
  if relocated {
    match missing_fixed_array {
      Some(storage) => {
        let new_address = file_offset_to_stored_address(
          storage.block_start.to_uint64(),
          self.superblock.base_address,
        )
        patch_little_endian_uint(
          output,
          storage.block_address_offset,
          storage.block_address_width,
          new_address,
          "fixed array data block address",
        )
        patch_metadata_checksum_in_place(
          output,
          address_to_offset(layout.data_address, self.superblock.base_address),
          storage.header_checksum_offset,
        )
        let combined = @buffer.Buffer::Buffer()
        for _ in 0..<(storage.block_start - self.bytes.length()) {
          combined.write_byte(b'\x00')
        }
        combined.write_bytes(
          storage.image.unsafe_reinterpret_as_bytes().get_view().unwrap(),
        )
        let chunk_bytes = tail.to_bytes()
        combined.write_bytes(chunk_bytes.get_view().unwrap())
        return self.finish_appended_image(output, combined)
      }
      None =>
        if appended_metadata_images.length() > 0 {
          let combined = @buffer.Buffer::Buffer()
          for index in 0..<appended_metadata_images.length() {
            let start = append_aligned_bytes_to_tail(
              combined,
              self.bytes.length(),
              appended_metadata_images[index].unsafe_reinterpret_as_bytes(),
              "create missing extensible array metadata",
            )
            guard start.to_int() == appended_metadata_starts[index] else {
              raise InvalidField("extensible array metadata placement mismatch")
            }
          }
          let tail_start = chunk_tail_base - self.bytes.length()
          for _ in combined.length()..<tail_start {
            combined.write_byte(b'\x00')
          }
          guard combined.length() == tail_start else {
            raise InvalidField("extensible array chunk tail placement mismatch")
          }
          let chunk_bytes = tail.to_bytes()
          combined.write_bytes(chunk_bytes.get_view().unwrap())
          return self.finish_appended_image(output, combined)
        } else {
          return self.finish_appended_image(output, tail)
        }
    }
  }
  if tail.length() > 0 {
    self.finish_appended_image(output, tail)
  } else {
    Bytes::from_array(output)
  }
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
  let variable_string = metadata.datatype.class == 9 &&
    metadata.datatype.vlen_is_string
  guard !datatype_contains_reference(metadata.datatype) else {
    raise UnsupportedFeature("reference datatypes cannot be patched")
  }
  guard variable_string || !datatype_contains_vlen(metadata.datatype) else {
    raise UnsupportedFeature(
      "only variable-length string datatypes can be patched",
    )
  }
  if variable_string && chunked {
    let filters = match header.first(msg_filter_pipeline) {
      Some(message) =>
        parse_filters(message, custom_filters=self.custom_filters)
      None => []
    }
    guard !filters.any(filter => filter.id != 1 && filter.id != 2) else {
      raise UnsupportedFeature(
        "variable-length datasets support only shuffle and deflate filters",
      )
    }
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
  let prefix_tail = @buffer.Buffer::Buffer()
  let payload = if variable_string {
    encode_patch_vlen_string_values(
      values,
      metadata.datatype,
      self.superblock.offset_size,
      self.superblock.length_size,
      self.superblock.base_address,
      self.bytes.length(),
      prefix_tail,
      "append a variable-length dataset heap",
    )
  } else {
    let encoded = encode_write_values(values)
    match values {
      StructuredValues(_) => encoded.payload
      _ =>
        if !metadata.datatype.little_endian &&
          (metadata.datatype.class == 0 || metadata.datatype.class == 1) {
          patch_byteswap_elements(encoded.payload, metadata.datatype.size)
        } else {
          encoded.payload
        }
    }
  }
  guard payload.length() == full_size else {
    raise InvalidArgument("patched values do not match the dataset shape")
  }
  let tail_prefix = prefix_tail.to_bytes()
  if compact {
    self.patch_compact_dataset_data(header, metadata, payload, tail_prefix)
  } else if chunked {
    self.patch_chunked_dataset_data(
      header, metadata, payload, values, tail_prefix,
    )
  } else {
    self.patch_contiguous_dataset_data(
      header, metadata, payload, allow_noncontiguous, tail_prefix,
    )
  }
}

///|
/// Replaces the values of an allocated contiguous dataset and returns a new
/// byte image. The shape and datatype must match exactly. Fixed-size values and
/// variable-length UTF-8 strings are supported.
///
/// Attributes and links are preserved. Variable-length strings append an aligned
/// global heap and update the superblock EOF/checksum. Chunked, compact, external,
/// filtered, reference and non-string variable-length storage is rejected. Use
/// `patch_dataset_values` to patch compact or unallocated storage as well.
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
/// allocated into writable storage already present in the chunk index, and a
/// missing v1 or v2 B-tree chunk record is created by rebuilding the complete
/// chunk tree from final addresses, stored sizes, masks and coordinates. V1
/// rebuilds use the superblock's indexed-storage K, sibling links and
/// datatype-sized termination keys; v2 rebuilds preserve the node size and
/// split/merge thresholds. Missing fixed- or extensible-array pages inside an
/// existing data block are initialized and marked allocated; other data blocks,
/// super blocks and index blocks are not created. Fixed-size values and
/// variable-length UTF-8 strings are supported. Variable-length strings use one
/// appended global heap; chunked values accept unfiltered storage, Shuffle or deflate,
/// and cannot create a missing fixed-array data block or extensible-array
/// metadata. External storage, reference datatypes and non-string variable-length
/// datatypes are rejected. The dataset shape and chunk grid are never changed.
pub fn Hdf5File::patch_dataset_values(
  self : Hdf5File,
  path : String,
  values : Hdf5WriteValues,
) -> Bytes raise Hdf5Error {
  self.patch_dataset_values_impl(path, values, true)
}
