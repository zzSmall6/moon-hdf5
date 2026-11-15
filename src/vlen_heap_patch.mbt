///|
priv struct PatchHeap {
  objects : Array[Bytes]
  data : Bytes
}

///|
fn patch_unique_heap_objects(
  values : Array[Bytes],
) -> Array[Bytes] raise Hdf5Error {
  let objects : Array[Bytes] = []
  for item in values {
    if item.length() > 0 && !objects.contains(item) {
      guard objects.length() < 65535 else {
        raise UnsupportedFeature(
          "variable-length values have too many heap objects",
        )
      }
      objects.push(item)
    }
  }
  objects
}

///|
fn encode_patch_heap(
  objects : Array[Bytes],
  length_size : Int32,
) -> PatchHeap raise Hdf5Error {
  if objects.length() == 0 {
    return { objects, data: b"", }
  }
  guard length_size >= 1 && length_size <= 8 else {
    raise UnsupportedFeature("unsupported global heap length size")
  }
  let header_size = 8 + length_size
  let object_header_size = 8 + length_size
  let mut used_size = header_size
  for item in objects {
    let length = item.length()
    guard length >= 0 && length <= 0x7FFFFFFF else {
      raise InvalidField("variable-length heap object is too large")
    }
    guard used_size <=
      0x7FFFFFFF - object_header_size - round_up_to_eight(length) else {
      raise UnsupportedFeature("variable-length heap is too large")
    }
    used_size = used_size + object_header_size + round_up_to_eight(length)
  }
  guard used_size.to_int64() + object_header_size.to_int64() <=
    0x7FFFFFFFL - 4095L else {
    raise UnsupportedFeature("variable-length heap is too large")
  }
  let collection_size = (used_size + object_header_size + 4095) / 4096 * 4096
  let buffer = @buffer.Buffer::Buffer(size_hint=collection_size)
  writer_write_bytes(buffer, b"GCOL")
  buffer.write_byte(b'\x01')
  writer_write_zeros(buffer, 3)
  write_uint_width(buffer, collection_size.to_uint64(), length_size)
  let mut object_index = 0
  for item in objects {
    object_index = object_index + 1
    buffer.write_uint16_le(object_index.to_uint16())
    buffer.write_uint16_le(0)
    buffer.write_uint_le(0U)
    write_uint_width(buffer, item.length().to_uint64(), length_size)
    writer_write_bytes(buffer, item)
    writer_write_zeros(buffer, round_up_to_eight(item.length()) - item.length())
  }
  buffer.write_uint16_le(0)
  buffer.write_uint16_le(0)
  buffer.write_uint_le(0U)
  write_uint_width(
    buffer,
    (collection_size - used_size).to_uint64(),
    length_size,
  )
  writer_write_zeros(buffer, collection_size - buffer.length())
  guard buffer.length() == collection_size else {
    raise InvalidField("variable-length heap size mismatch")
  }
  { objects, data: buffer.to_bytes(), }
}

///|
fn patch_vlen_string_charset_valid(value : String, charset : Int32) -> Bool {
  let encoded = @utf8.encode(value)
  if charset == 0 {
    for index in 0..<encoded.length() {
      if encoded.unsafe_get(index).to_int() >= 128 {
        return false
      }
    }
  }
  true
}

///|
fn Hdf5File::encode_patch_vlen_values(
  self : Hdf5File,
  values : Hdf5WriteValues,
  datatype : Datatype,
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
  file_length : Int32,
  tail : @buffer.Buffer,
  purpose : String,
) -> Bytes raise Hdf5Error {
  guard datatype.class == 9 else {
    raise UnsupportedFeature("only variable-length values can be patched")
  }
  let items : Array[DatasetValue] = if datatype.vlen_is_string {
    let strings = match values {
      VariableStringValues(items) => items
      _ =>
        raise InvalidArgument("patched values must be variable-length strings")
    }
    let converted : Array[DatasetValue] = []
    for item in strings {
      converted.push(Text(item))
    }
    converted
  } else {
    match values {
      StructuredValues(_, items) => items
      _ =>
        raise InvalidArgument(
          "patched values must be variable-length sequences",
        )
    }
  }
  let heap_start = next_aligned_tail_start(file_length, tail.length(), purpose)
  let heap_address = file_offset_to_stored_address(heap_start, base_address)
  let heap : PatchHeap = { objects: [], data: b"", }
  let payload = @buffer.Buffer::Buffer(size_hint=items.length() * datatype.size)
  for item in items {
    writer_write_bytes(
      payload,
      self.encode_patch_structured_reference_value(
        datatype, item, heap, heap_address, offset_size, 0,
      ),
    )
  }
  let bytes = payload.to_bytes()
  guard bytes.length() == items.length() * datatype.size else {
    raise InvalidField("variable-length payload size mismatch")
  }
  let encoded_heap = encode_patch_heap(heap.objects, length_size)
  if encoded_heap.data.length() > 0 {
    let start = append_aligned_bytes_to_tail(
      tail,
      file_length,
      encoded_heap.data,
      purpose,
    )
    guard start == heap_start else {
      raise InvalidField("variable-length heap address changed while encoding")
    }
  }
  bytes
}
