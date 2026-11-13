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
fn encode_patch_vlen_references(
  lengths : Array[Int32],
  objects : Array[Bytes],
  heap : PatchHeap,
  heap_address : UInt64,
  offset_size : Int32,
  element_size : Int32,
) -> Bytes raise Hdf5Error {
  guard lengths.length() == objects.length() else {
    raise InvalidField("variable-length value metadata length mismatch")
  }
  guard element_size >= 8 + offset_size else {
    raise UnsupportedFeature("variable-length datatype is too small")
  }
  guard lengths.length().to_int64() * element_size.to_int64() <= 0x7FFFFFFFL else {
    raise UnsupportedFeature("variable-length value is too large")
  }
  let buffer = @buffer.Buffer::Buffer(size_hint=lengths.length() * element_size)
  for index in 0..<lengths.length() {
    buffer.write_uint_le(lengths[index].reinterpret_as_uint())
    let item = objects[index]
    if item.length() == 0 {
      write_uint_width(buffer, 0UL, offset_size)
      buffer.write_uint_le(0U)
    } else {
      let heap_index = match heap.objects.search_by(object => object == item) {
        Some(heap_index) => heap_index
        None =>
          raise InvalidField("variable-length value is missing from its heap")
      }
      write_uint_width(buffer, heap_address, offset_size)
      buffer.write_uint_le((heap_index + 1).reinterpret_as_uint())
    }
    writer_write_zeros(buffer, element_size - 8 - offset_size)
  }
  let payload = buffer.to_bytes()
  guard payload.length() == lengths.length() * element_size else {
    raise InvalidField("variable-length reference size mismatch")
  }
  payload
}

///|
fn encode_patch_vlen_values(
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
  let lengths : Array[Int32] = []
  let objects : Array[Bytes] = []
  if datatype.vlen_is_string {
    guard datatype.string_charset >= 0 && datatype.string_charset <= 1 else {
      raise UnsupportedFeature("unsupported variable-length string charset")
    }
    let items = match values {
      VariableStringValues(items) => items
      _ =>
        raise InvalidArgument("patched values must be variable-length strings")
    }
    for item in items {
      guard patch_vlen_string_charset_valid(item, datatype.string_charset) else {
        raise InvalidArgument(
          "variable-length strings do not match the datatype charset",
        )
      }
      let encoded = @utf8.encode(item)
      lengths.push(encoded.length())
      objects.push(encoded)
    }
  } else {
    guard datatype.base_types.length() == 1 && datatype.base_types[0].size > 0 else {
      raise UnsupportedFeature(
        "variable-length sequence datatype is missing a base datatype",
      )
    }
    let base = datatype.base_types[0]
    guard !datatype_contains_vlen(base) else {
      raise UnsupportedFeature(
        "variable-length sequence base must be fixed-size",
      )
    }
    guard !datatype_contains_reference(base) else {
      raise UnsupportedFeature(
        "variable-length sequence base cannot contain references",
      )
    }
    let items = match values {
      StructuredValues(_, items) => items
      _ =>
        raise InvalidArgument(
          "patched values must be variable-length sequences",
        )
    }
    for item in items {
      let sequence = match item {
        Sequence(values) => values
        _ =>
          raise InvalidArgument(
            "patched values must be variable-length sequences",
          )
      }
      guard sequence.length().to_int64() <= 0x7FFFFFFFL else {
        raise UnsupportedFeature("variable-length sequence is too large")
      }
      if sequence.length() == 0 {
        lengths.push(0)
        objects.push(b"")
      } else {
        let encoded = @buffer.Buffer::Buffer(
          size_hint=sequence.length() * base.size,
        )
        for element in sequence {
          writer_write_bytes(encoded, encode_structured_value(base, element))
        }
        lengths.push(sequence.length())
        objects.push(encoded.to_bytes())
      }
    }
  }
  let heap = encode_patch_heap(patch_unique_heap_objects(objects), length_size)
  let heap_address = if heap.data.length() == 0 {
    0UL
  } else {
    let start = append_aligned_bytes_to_tail(
      tail,
      file_length,
      heap.data,
      purpose,
    )
    file_offset_to_stored_address(start, base_address)
  }
  encode_patch_vlen_references(
    lengths,
    objects,
    heap,
    heap_address,
    offset_size,
    datatype.size,
  )
}
