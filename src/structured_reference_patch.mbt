///|
fn patch_nested_reference_datatype(datatype : Datatype) -> Bool {
  (datatype.class == 6 || datatype.class == 10) &&
  datatype_contains_reference(datatype)
}

///|
fn patch_vlen_reference_datatype(datatype : Datatype) -> Bool {
  datatype.class == 9 &&
  !datatype.vlen_is_string &&
  datatype.base_types.length() == 1 &&
  datatype_contains_reference(datatype.base_types[0])
}

///|
fn validate_patch_nested_reference_datatype(
  datatype : Datatype,
  offset_size : Int32,
) -> Unit raise Hdf5Error {
  if datatype.version >= 4 {
    guard datatype.size == offset_size + 10 else {
      raise UnsupportedFeature("version-4 nested reference has the wrong size")
    }
    guard datatype.reference_kind == 2 ||
      datatype.reference_kind == 3 ||
      datatype.reference_kind == 4 else {
      raise UnsupportedFeature("unsupported version-4 nested reference kind")
    }
  } else if datatype.reference_kind == 0 {
    guard datatype.size == offset_size else {
      raise UnsupportedFeature("legacy object reference has the wrong size")
    }
  } else if datatype.reference_kind == 1 {
    guard datatype.size == offset_size + 4 else {
      raise UnsupportedFeature("legacy region reference has the wrong size")
    }
  } else {
    raise UnsupportedFeature("unsupported nested reference datatype")
  }
}

///|
fn Hdf5File::collect_patch_structured_reference_payloads(
  self : Hdf5File,
  datatype : Datatype,
  value : DatasetValue,
  payloads : Array[Bytes],
  offset_size : Int32,
  depth : Int32,
) -> Unit raise Hdf5Error {
  guard depth < 16 else {
    raise InvalidArgument("nested reference value is too deeply nested")
  }
  match datatype.class {
    6 => {
      let fields = match value {
        Compound(fields) => fields
        _ => raise InvalidArgument("compound value does not match its datatype")
      }
      guard fields.length() == datatype.compound_members.length() else {
        raise InvalidArgument("compound value does not match its datatype")
      }
      for index in 0..<fields.length() {
        let field = datatype.compound_members[index]
        guard fields[index].name == field.name else {
          raise InvalidArgument("compound value fields are out of order")
        }
        self.collect_patch_structured_reference_payloads(
          field.datatype,
          fields[index].value,
          payloads,
          offset_size,
          depth + 1,
        )
      }
    }
    7 => {
      validate_patch_nested_reference_datatype(datatype, offset_size)
      if datatype.version >= 4 {
        match datatype.reference_kind {
          2 =>
            match value {
              ObjectReference(None) => ()
              ObjectReference(Some(path)) => self.resolve(path) |> ignore
              _ =>
                raise InvalidArgument(
                  "version-4 object reference value has the wrong kind",
                )
            }
          3 =>
            match value {
              RegionReference(None, _) => ()
              RegionReference(Some(path), selection) =>
                payloads.push(
                  self.encode_patch_version4_region_reference_payload(
                    path, selection, offset_size,
                  ),
                )
              _ =>
                raise InvalidArgument(
                  "version-4 region reference value has the wrong kind",
                )
            }
          4 =>
            match value {
              AttributeReference(None, _) => ()
              AttributeReference(Some(path), name) =>
                payloads.push(
                  self.encode_patch_version4_attribute_reference_payload(
                    path, name, offset_size,
                  ),
                )
              _ =>
                raise InvalidArgument(
                  "version-4 attribute reference value has the wrong kind",
                )
            }
          _ =>
            raise UnsupportedFeature(
              "unsupported version-4 nested reference kind",
            )
        }
      } else if datatype.reference_kind == 0 {
        match value {
          ObjectReference(None) => ()
          ObjectReference(Some(path)) => self.resolve(path) |> ignore
          _ =>
            raise InvalidArgument("object reference value has the wrong kind")
        }
      } else {
        match value {
          RegionReference(None, _) => ()
          RegionReference(Some(path), selection) =>
            payloads.push(
              self.encode_patch_region_reference_payload(
                path, selection, offset_size,
              ),
            )
          _ =>
            raise InvalidArgument("region reference value has the wrong kind")
        }
      }
    }
    9 => ()
    10 => {
      let items = match value {
        Elements(items) => items
        _ => raise InvalidArgument("array value does not match its datatype")
      }
      guard datatype.base_types.length() == 1 &&
        datatype.base_types[0].size > 0 &&
        items.length() == datatype.size / datatype.base_types[0].size else {
        raise InvalidArgument("array value does not match its datatype")
      }
      for item in items {
        self.collect_patch_structured_reference_payloads(
          datatype.base_types[0],
          item,
          payloads,
          offset_size,
          depth + 1,
        )
      }
    }
    _ => ()
  }
}

///|
fn patch_add_heap_object(
  heap : PatchHeap,
  payload : Bytes,
) -> Int32 raise Hdf5Error {
  match heap.objects.search_by(object => object == payload) {
    Some(index) => index + 1
    None => {
      guard payload.length() > 0 else {
        raise InvalidField("empty heap objects must use a null descriptor")
      }
      guard heap.objects.length() < 65535 else {
        raise UnsupportedFeature(
          "variable-length values have too many heap objects",
        )
      }
      heap.objects.push(payload)
      heap.objects.length()
    }
  }
}

///|
fn Hdf5File::encode_patch_structured_reference_value(
  self : Hdf5File,
  datatype : Datatype,
  value : DatasetValue,
  heap : PatchHeap,
  heap_address : UInt64,
  offset_size : Int32,
  depth : Int32,
) -> Bytes raise Hdf5Error {
  guard depth < 16 else {
    raise InvalidArgument("nested reference value is too deeply nested")
  }
  match datatype.class {
    6 => {
      let fields = match value {
        Compound(fields) => fields
        _ => raise InvalidArgument("compound value does not match its datatype")
      }
      guard fields.length() == datatype.compound_members.length() else {
        raise InvalidArgument("compound value does not match its datatype")
      }
      let output = FixedArray::make(datatype.size, b'\x00')
      for index in 0..<fields.length() {
        let field = datatype.compound_members[index]
        guard fields[index].name == field.name else {
          raise InvalidArgument("compound value fields are out of order")
        }
        let encoded = self.encode_patch_structured_reference_value(
          field.datatype,
          fields[index].value,
          heap,
          heap_address,
          offset_size,
          depth + 1,
        )
        output.blit_from_bytes(field.offset, encoded, 0, encoded.length())
      }
      output.unsafe_reinterpret_as_bytes()
    }
    7 => {
      validate_patch_nested_reference_datatype(datatype, offset_size)
      let buffer = @buffer.Buffer::Buffer(size_hint=datatype.size)
      if datatype.version >= 4 {
        match datatype.reference_kind {
          2 =>
            match value {
              ObjectReference(None) => writer_write_zeros(buffer, datatype.size)
              ObjectReference(Some(path)) => {
                let node = self.resolve(path)
                buffer.write_byte(b'\x02')
                buffer.write_byte(b'\x00')
                buffer.write_byte(offset_size.to_byte())
                write_uint_width(buffer, node.address, offset_size)
                writer_write_zeros(buffer, datatype.size - offset_size - 3)
              }
              _ =>
                raise InvalidArgument(
                  "version-4 object reference value has the wrong kind",
                )
            }
          3 =>
            match value {
              RegionReference(None, _) =>
                writer_write_zeros(buffer, datatype.size)
              RegionReference(Some(path), selection) => {
                let payload = self.encode_patch_version4_region_reference_payload(
                  path, selection, offset_size,
                )
                buffer.write_byte(b'\x03')
                buffer.write_byte(b'\x00')
                buffer.write_uint_le(payload.length().reinterpret_as_uint())
                write_uint_width(buffer, heap_address, offset_size)
                buffer.write_uint_le(
                  patch_add_heap_object(heap, payload).reinterpret_as_uint(),
                )
              }
              _ =>
                raise InvalidArgument(
                  "version-4 region reference value has the wrong kind",
                )
            }
          4 =>
            match value {
              AttributeReference(None, _) =>
                writer_write_zeros(buffer, datatype.size)
              AttributeReference(Some(path), name) => {
                let payload = self.encode_patch_version4_attribute_reference_payload(
                  path, name, offset_size,
                )
                buffer.write_byte(b'\x04')
                buffer.write_byte(b'\x00')
                buffer.write_uint_le(payload.length().reinterpret_as_uint())
                write_uint_width(buffer, heap_address, offset_size)
                buffer.write_uint_le(
                  patch_add_heap_object(heap, payload).reinterpret_as_uint(),
                )
              }
              _ =>
                raise InvalidArgument(
                  "version-4 attribute reference value has the wrong kind",
                )
            }
          _ =>
            raise UnsupportedFeature(
              "unsupported version-4 nested reference kind",
            )
        }
      } else if datatype.reference_kind == 0 {
        match value {
          ObjectReference(None) => writer_write_zeros(buffer, datatype.size)
          ObjectReference(Some(path)) => {
            let node = self.resolve(path)
            write_uint_width(buffer, node.address, datatype.size)
          }
          _ =>
            raise InvalidArgument("object reference value has the wrong kind")
        }
      } else {
        match value {
          RegionReference(None, _) => writer_write_zeros(buffer, datatype.size)
          RegionReference(Some(path), selection) => {
            let payload = self.encode_patch_region_reference_payload(
              path, selection, offset_size,
            )
            write_uint_width(buffer, heap_address, offset_size)
            buffer.write_uint_le(
              patch_add_heap_object(heap, payload).reinterpret_as_uint(),
            )
          }
          _ =>
            raise InvalidArgument("region reference value has the wrong kind")
        }
      }
      guard buffer.length() == datatype.size else {
        raise InvalidField(
          "nested reference value size does not match its datatype",
        )
      }
      buffer.to_bytes()
    }
    9 =>
      if datatype.vlen_is_string {
        let text = match value {
          Text(text) => text
          _ =>
            raise InvalidArgument(
              "variable-length string value has the wrong kind",
            )
        }
        guard datatype.string_charset >= 0 && datatype.string_charset <= 1 else {
          raise UnsupportedFeature("unsupported variable-length string charset")
        }
        guard patch_vlen_string_charset_valid(text, datatype.string_charset) else {
          raise InvalidArgument(
            "variable-length strings do not match the datatype charset",
          )
        }
        guard datatype.size >= 8 + offset_size else {
          raise UnsupportedFeature("variable-length datatype is too small")
        }
        let encoded = @utf8.encode(text)
        let buffer = @buffer.Buffer::Buffer(size_hint=datatype.size)
        if encoded.length() == 0 {
          writer_write_zeros(buffer, datatype.size)
        } else {
          let index = patch_add_heap_object(heap, encoded)
          buffer.write_uint_le(encoded.length().reinterpret_as_uint())
          write_uint_width(buffer, heap_address, offset_size)
          buffer.write_uint_le(index.reinterpret_as_uint())
          writer_write_zeros(buffer, datatype.size - 8 - offset_size)
        }
        guard buffer.length() == datatype.size else {
          raise InvalidField(
            "variable-length string value size does not match its datatype",
          )
        }
        buffer.to_bytes()
      } else {
        guard datatype.base_types.length() == 1 &&
          datatype.base_types[0].size > 0 else {
          raise InvalidField(
            "variable-length sequence is missing its base type",
          )
        }
        guard datatype.size >= 8 + offset_size else {
          raise UnsupportedFeature("variable-length datatype is too small")
        }
        let items = match value {
          Sequence(items) => items
          _ =>
            raise InvalidArgument(
              "variable-length sequence value has the wrong kind",
            )
        }
        let base = datatype.base_types[0]
        guard items.length().to_int64() <= 0x7FFFFFFFL / base.size.to_int64() else {
          raise UnsupportedFeature(
            "variable-length sequence payload is too large",
          )
        }
        let encoded = @buffer.Buffer::Buffer(
          size_hint=items.length() * base.size,
        )
        for item in items {
          writer_write_bytes(
            encoded,
            self.encode_patch_structured_reference_value(
              base,
              item,
              heap,
              heap_address,
              offset_size,
              depth + 1,
            ),
          )
        }
        let payload = encoded.to_bytes()
        guard payload.length() == items.length() * base.size else {
          raise InvalidField("variable-length sequence payload size mismatch")
        }
        let buffer = @buffer.Buffer::Buffer(size_hint=datatype.size)
        if items.length() == 0 {
          writer_write_zeros(buffer, datatype.size)
        } else {
          let index = patch_add_heap_object(heap, payload)
          buffer.write_uint_le(items.length().reinterpret_as_uint())
          write_uint_width(buffer, heap_address, offset_size)
          buffer.write_uint_le(index.reinterpret_as_uint())
          writer_write_zeros(buffer, datatype.size - 8 - offset_size)
        }
        guard buffer.length() == datatype.size else {
          raise InvalidField(
            "variable-length sequence value size does not match its datatype",
          )
        }
        buffer.to_bytes()
      }
    10 => {
      let items = match value {
        Elements(items) => items
        _ => raise InvalidArgument("array value does not match its datatype")
      }
      guard datatype.base_types.length() == 1 &&
        datatype.base_types[0].size > 0 &&
        items.length() == datatype.size / datatype.base_types[0].size else {
        raise InvalidArgument("array value does not match its datatype")
      }
      let buffer = @buffer.Buffer::Buffer(size_hint=datatype.size)
      for item in items {
        writer_write_bytes(
          buffer,
          self.encode_patch_structured_reference_value(
            datatype.base_types[0],
            item,
            heap,
            heap_address,
            offset_size,
            depth + 1,
          ),
        )
      }
      guard buffer.length() == datatype.size else {
        raise InvalidField("array value size does not match its datatype")
      }
      buffer.to_bytes()
    }
    _ => encode_structured_value(datatype, value)
  }
}

///|
fn Hdf5File::encode_patch_nested_references(
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
  let items = match values {
    StructuredValues(_, items) => items
    _ => raise InvalidArgument("patched values must be structured values")
  }
  let payloads : Array[Bytes] = []
  for item in items {
    self.collect_patch_structured_reference_payloads(
      datatype, item, payloads, offset_size, 0,
    )
  }
  let heap_start = next_aligned_tail_start(file_length, tail.length(), purpose)
  let heap_address = file_offset_to_stored_address(heap_start, base_address)
  let heap : PatchHeap = {
    objects: patch_unique_heap_objects(payloads),
    data: b"",
  }
  let buffer = @buffer.Buffer::Buffer(size_hint=items.length() * datatype.size)
  for item in items {
    writer_write_bytes(
      buffer,
      self.encode_patch_structured_reference_value(
        datatype, item, heap, heap_address, offset_size, 0,
      ),
    )
  }
  let payload = buffer.to_bytes()
  guard payload.length() == items.length() * datatype.size else {
    raise InvalidField("nested structured payload size mismatch")
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
      raise InvalidField("nested heap address changed while encoding")
    }
  }
  payload
}
