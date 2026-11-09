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
      datatype.little_endian &&
      datatype.bit_offset == 0 &&
      datatype.bit_precision == 8
    Int16Values(_) =>
      datatype.class == 0 &&
      datatype.size == 2 &&
      datatype.signed &&
      datatype.little_endian &&
      datatype.bit_offset == 0 &&
      datatype.bit_precision == 16
    Int32Values(_) =>
      datatype.class == 0 &&
      datatype.size == 4 &&
      datatype.signed &&
      datatype.little_endian &&
      datatype.bit_offset == 0 &&
      datatype.bit_precision == 32
    Int64Values(_) =>
      datatype.class == 0 &&
      datatype.size == 8 &&
      datatype.signed &&
      datatype.little_endian &&
      datatype.bit_offset == 0 &&
      datatype.bit_precision == 64
    UInt8Values(_) =>
      datatype.class == 0 &&
      datatype.size == 1 &&
      !datatype.signed &&
      datatype.little_endian &&
      datatype.bit_offset == 0 &&
      datatype.bit_precision == 8
    UInt16Values(_) =>
      datatype.class == 0 &&
      datatype.size == 2 &&
      !datatype.signed &&
      datatype.little_endian &&
      datatype.bit_offset == 0 &&
      datatype.bit_precision == 16
    UInt32Values(_) =>
      datatype.class == 0 &&
      datatype.size == 4 &&
      !datatype.signed &&
      datatype.little_endian &&
      datatype.bit_offset == 0 &&
      datatype.bit_precision == 32
    UInt64Values(_) =>
      datatype.class == 0 &&
      datatype.size == 8 &&
      !datatype.signed &&
      datatype.little_endian &&
      datatype.bit_offset == 0 &&
      datatype.bit_precision == 64
    Float32Values(_) =>
      datatype.class == 1 &&
      datatype.size == 4 &&
      datatype.little_endian &&
      is_ieee_float_layout(datatype)
    Float64Values(_) =>
      datatype.class == 1 &&
      datatype.size == 8 &&
      datatype.little_endian &&
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
/// Replaces the values of an allocated contiguous dataset and returns a new
/// byte image. The shape and datatype must match exactly, and the operation is
/// limited to bytes-backed files with little-endian fixed-size values.
///
/// Only raw dataset bytes change: object headers, attributes, links and
/// checksums are preserved. Chunked, compact, external, filtered and
/// variable-length storage is rejected.
pub fn Hdf5File::patch_contiguous_dataset(
  self : Hdf5File,
  path : String,
  values : Hdf5WriteValues,
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
  guard metadata.layout.layout_class == 1 else {
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
  guard encoded.payload.length() == full_size else {
    raise InvalidArgument("patched values do not match the dataset shape")
  }
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
  for index in 0..<encoded.payload.length() {
    output[offset + index] = encoded.payload[index]
  }
  Bytes::from_array(output)
}
