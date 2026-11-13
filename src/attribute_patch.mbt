///|
priv struct PatchableAttribute {
  attribute : Attribute
  value_offset : Int32
  value_length : Int32
  inline_offset : Int32
  chunk_start : Int32
  checksum_offset : Int32
  heap : FractalHeap?
  heap_id : Bytes?
}

///|
fn attribute_value_offset(data : Bytes) -> (Int32, Int32) raise Hdf5Error {
  let reader = Reader::new(data)
  let version = reader.read_u8().to_int()
  if version == 1 {
    let _reserved = reader.read_u8()
  } else {
    let _flags = reader.read_u8()
  }
  let name_size = reader.read_u16_le().to_int()
  let datatype_size = reader.read_u16_le().to_int()
  let dataspace_size = reader.read_u16_le().to_int()
  if version >= 3 {
    let _encoding = reader.read_u8()
  }
  reader.skip(name_size)
  if version == 1 {
    reader.skip(round_up_to_eight(name_size) - name_size)
  }
  reader.skip(datatype_size)
  if version == 1 {
    reader.skip(round_up_to_eight(datatype_size) - datatype_size)
  }
  reader.skip(dataspace_size)
  let offset = reader.position()
  guard offset <= data.length() else {
    raise InvalidField("attribute message is truncated")
  }
  (offset, data.length() - offset)
}

///|
fn Hdf5File::find_patchable_attribute(
  self : Hdf5File,
  header : ObjectHeader,
  name : String,
) -> PatchableAttribute raise Hdf5Error {
  for message in header.find(msg_attribute_info) {
    let reader = Reader::new(message.data)
    guard reader.read_u8() == b'\x00' else {
      raise UnsupportedFeature("unsupported attribute info message version")
    }
    let flags = reader.read_u8().to_int()
    guard (flags & 0xFC) == 0 && (flags & 3) != 2 else {
      raise InvalidField("invalid attribute info flags")
    }
    if (flags & 1) != 0 {
      reader.skip(2)
    }
    let heap_address = reader.read_uint_le(self.superblock.offset_size)
    let name_index_address = reader.read_uint_le(self.superblock.offset_size)
    let undefined = heap_undefined_address(self.superblock.offset_size)
    let order_index_address = if (flags & 2) != 0 {
      reader.read_uint_le(self.superblock.offset_size)
    } else {
      undefined
    }
    if heap_address != undefined {
      let ordered = (flags & 2) != 0 && order_index_address != undefined
      let index_address = if ordered {
        order_index_address
      } else if name_index_address != undefined {
        name_index_address
      } else {
        raise InvalidField("dense attributes have no name or order index")
      }
      let heap = parse_fractal_heap(
        self.bytes,
        heap_address,
        self.superblock.offset_size,
        self.superblock.length_size,
        self.superblock.base_address,
      )
      let records = read_metadata_btree_records(
        self.bytes,
        index_address,
        self.superblock.offset_size,
        self.superblock.length_size,
        self.superblock.base_address,
        [8, 9],
      )
      for record in records {
        let (payload, flags) = dense_attribute_payload(
          heap,
          record,
          Some(self.shared_messages),
        )
        let attribute = parse_attribute_data(
          payload,
          flags,
          self.bytes,
          self.superblock.offset_size,
          self.superblock.length_size,
          self.superblock.base_address,
          shared_messages=Some(self.shared_messages),
        )
        if attribute.name == name {
          guard (record.flags.to_int() & shared_message_flag) == 0 else {
            raise UnsupportedFeature(
              "shared dense attributes cannot be patched",
            )
          }
          let (value_offset, value_length) = attribute_value_offset(payload)
          return {
            attribute,
            value_offset,
            value_length,
            inline_offset: -1,
            chunk_start: -1,
            checksum_offset: -1,
            heap: Some(heap),
            heap_id: Some(record.heap_id),
          }
        }
      }
    } else {
      guard name_index_address == undefined && order_index_address == undefined else {
        raise InvalidField("attribute index is present without a fractal heap")
      }
    }
  }
  for message in header.messages {
    if message.msg_type == msg_attribute {
      let attribute = parse_attribute_message(
        message,
        self.bytes,
        self.superblock.offset_size,
        self.superblock.length_size,
        self.superblock.base_address,
        shared_messages=Some(self.shared_messages),
      )
      if attribute.name == name {
        guard (message.flags.to_int() & shared_message_flag) == 0 else {
          raise UnsupportedFeature("shared attributes cannot be patched")
        }
        let (value_offset, value_length) = attribute_value_offset(message.data)
        return {
          attribute,
          value_offset,
          value_length,
          inline_offset: message.data_offset,
          chunk_start: message.chunk_start,
          checksum_offset: message.checksum_offset,
          heap: None,
          heap_id: None,
        }
      }
    }
  }
  raise InvalidArgument("attribute does not exist: " + name)
}

///|
/// Replaces the values of an existing fixed-size attribute and returns a new
/// byte image. Variable-length UTF-8 strings and fixed-base variable-length
/// sequences are also supported; their new values are placed in an appended
/// global heap. The attribute datatype, dataspace, name and creation order are
/// preserved. Inline messages are updated in their object-header chunk; dense
/// attributes are updated through their fractal-heap record. Legacy same-file
/// object references are rewritten in their existing address-width slots;
/// region references append deduplicated target-address plus selection payloads
/// to a global heap and rewrite their existing heap-address/index slots.
/// Same-file version-4 object references rewrite direct address tokens; version-4
/// region and attribute references append indirect global-heap payloads and
/// rewrite their kind, length, heap-address and index fields. External-file
/// version-4 references, compound/array/sequence reference datatypes, shared
/// attribute messages and nested variable-length datatypes are rejected. Legacy
/// object-reference slots must match the file address size; legacy
/// region-reference slots must be four bytes wider and target existing datasets
/// with non-null dataspaces. Version-4 region and attribute targets follow the
/// same dataset and existing-attribute rules. Adding, removing or renaming
/// attributes is not supported.
pub fn Hdf5File::patch_attribute_values(
  self : Hdf5File,
  path : String,
  name : String,
  values : Hdf5WriteValues,
) -> Bytes raise Hdf5Error {
  guard self.source is None else {
    raise InvalidArgument("attribute patching requires a bytes-backed Hdf5File")
  }
  let node = self.resolve(path)
  let header = self.object_header_at(node.address)
  let slot = self.find_patchable_attribute(header, name)
  let datatype = slot.attribute.datatype
  let variable_length = datatype.class == 9
  let legacy_object_reference = patch_legacy_object_reference(
    values,
    datatype,
    self.superblock.offset_size,
  )
  let legacy_region_reference = patch_legacy_region_reference(
    values,
    datatype,
    self.superblock.offset_size,
  )
  let version4_object_reference = patch_version4_object_reference(
    values,
    datatype,
    self.superblock.offset_size,
  )
  let version4_region_reference = patch_version4_region_reference(
    values,
    datatype,
    self.superblock.offset_size,
  )
  let version4_attribute_reference = patch_version4_attribute_reference(
    values,
    datatype,
    self.superblock.offset_size,
  )
  guard !datatype_contains_reference(datatype) ||
    legacy_object_reference ||
    legacy_region_reference ||
    version4_object_reference ||
    version4_region_reference ||
    version4_attribute_reference else {
    raise UnsupportedFeature("reference attributes cannot be patched")
  }
  guard variable_length || !datatype_contains_vlen(datatype) else {
    raise UnsupportedFeature(
      "only variable-length string or sequence attributes can be patched",
    )
  }
  guard patch_values_match_datatype(values, datatype) else {
    raise InvalidArgument("patched values must match the attribute datatype")
  }
  let dimensions = slot.attribute.dataspace.dimensions
  validate_write_values(
    dimensions,
    values,
    legacy_reference_address_size=Some(self.superblock.offset_size),
  )
  let count = slot.attribute.dataspace.element_count()
  guard count >= 0 && count <= 0x7FFFFFFFL / datatype.size.to_int64() else {
    raise InvalidField("attribute byte size is too large")
  }
  let full_size = count.to_int() * datatype.size
  let output = self.bytes.to_fixedarray()
  let tail = @buffer.Buffer::Buffer()
  let payload = if variable_length {
    encode_patch_vlen_values(
      values,
      datatype,
      self.superblock.offset_size,
      self.superblock.length_size,
      self.superblock.base_address,
      self.bytes.length(),
      tail,
      "append a variable-length attribute heap",
    )
  } else if legacy_object_reference {
    self.encode_patch_object_references(values, datatype)
  } else if legacy_region_reference {
    self.encode_patch_region_references(
      values,
      datatype,
      self.superblock.offset_size,
      self.superblock.length_size,
      self.superblock.base_address,
      self.bytes.length(),
      tail,
      "append a region-reference attribute heap",
    )
  } else if version4_object_reference {
    self.encode_patch_version4_object_references(
      values,
      datatype,
      self.superblock.offset_size,
    )
  } else if version4_region_reference {
    self.encode_patch_version4_region_references(
      values,
      datatype,
      self.superblock.offset_size,
      self.superblock.length_size,
      self.superblock.base_address,
      self.bytes.length(),
      tail,
      "append a version-4 region-reference attribute heap",
    )
  } else if version4_attribute_reference {
    self.encode_patch_version4_attribute_references(
      values,
      datatype,
      self.superblock.offset_size,
      self.superblock.length_size,
      self.superblock.base_address,
      self.bytes.length(),
      tail,
      "append a version-4 attribute-reference attribute heap",
    )
  } else {
    let encoded = encode_write_values(values)
    match values {
      StructuredValues(_) => encoded.payload
      _ =>
        if !datatype.little_endian &&
          (datatype.class == 0 || datatype.class == 1) {
          patch_byteswap_elements(encoded.payload, datatype.size)
        } else {
          encoded.payload
        }
    }
  }
  guard payload.length() == full_size else {
    raise InvalidArgument("patched values do not match the attribute shape")
  }
  guard payload.length() <= slot.value_length else {
    raise InvalidArgument("patched attribute value exceeds its message storage")
  }
  match slot.heap {
    Some(heap) => {
      let heap_id = match slot.heap_id {
        Some(heap_id) => heap_id
        None => raise InvalidField("dense attribute has no heap ID")
      }
      heap.patch_data(heap_id, slot.value_offset, payload, output)
    }
    None =>
      patch_heap_output_bytes(
        output,
        slot.inline_offset + slot.value_offset,
        payload,
      )
  }
  if slot.checksum_offset >= 0 {
    patch_metadata_checksum_in_place(
      output,
      slot.chunk_start,
      slot.checksum_offset,
    )
  }
  if tail.length() > 0 {
    self.finish_appended_image(output, tail)
  } else {
    Bytes::from_array(output)
  }
}
