///|
/// One attribute message that can be carried into rebuilt compact or dense
/// storage. `data` is the attribute-message payload without its object-header
/// message prefix.
priv struct EditableAttribute {
  name : String
  creation_order : UInt64
  flags : Byte
  data : Bytes
}

///|
fn editable_uint_fits_width(value : UInt64, width : Int32) -> Bool {
  width >= 8 || value < 1UL << (width * 8)
}

///|
/// Layout of an object header whose initial chunk will be replaced by a
/// continuation message. The chunk size itself is never changed.
priv struct ObjectHeaderRewritePlan {
  version : Int32
  offset : Int32
  flags : Int32
  body_start : Int32
  body_size : Int32
  checksum_offset : Int32
  track_order : Bool
}

///|
fn editable_attribute_name(data : Bytes) -> String raise Hdf5Error {
  let reader = Reader::new(data)
  let version = reader.read_u8().to_int()
  guard version >= 1 && version <= 3 else {
    raise UnsupportedFeature("unsupported attribute message version")
  }
  if version == 1 {
    let _reserved = reader.read_u8()
  } else {
    let _flags = reader.read_u8()
  }
  let name_size = reader.read_u16_le().to_int()
  let _datatype_size = reader.read_u16_le().to_int()
  let _dataspace_size = reader.read_u16_le().to_int()
  if version >= 3 {
    let _encoding = reader.read_u8()
  }
  guard name_size > 0 else {
    raise InvalidField("attribute message has no name")
  }
  let name_bytes = reader.read_bytes(name_size)
  decode_c_string(name_bytes)
}

///|
/// Collects every attribute message on an object in creation/read order.
/// Shared messages are resolved so rebuilt storage can contain local payloads.
fn Hdf5File::editable_attributes(
  self : Hdf5File,
  header : ObjectHeader,
) -> Array[EditableAttribute] raise Hdf5Error {
  let attributes : Array[EditableAttribute] = []
  let dense : Array[EditableAttribute] = []
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
        let record_flags = record.flags.to_int()
        let data = if (record_flags & shared_message_flag) != 0 {
          self.shared_messages.resolve_sohm_heap_id(
            record.heap_id,
            msg_attribute,
          )
        } else {
          heap.get_data(record.heap_id)
        }
        dense.push({
          name: editable_attribute_name(data),
          creation_order: record.creation_order,
          flags: (record_flags & (0xFF ^ shared_message_flag)).to_byte(),
          data,
        })
      }
      if !ordered {
        dense.sort_by((left, right) => {
          if left.name < right.name {
            -1
          } else if left.name > right.name {
            1
          } else {
            0
          }
        })
        for index in 0..<dense.length() {
          dense[index] = {
            name: dense[index].name,
            creation_order: index.to_uint64(),
            flags: dense[index].flags,
            data: dense[index].data,
          }
        }
      }
      attributes.append(dense)
    } else {
      guard name_index_address == undefined && order_index_address == undefined else {
        raise InvalidField("attribute index is present without a fractal heap")
      }
    }
  }
  let mut next_order = 0UL
  for attribute in attributes {
    if attribute.creation_order >= next_order {
      next_order = attribute.creation_order + 1UL
    }
  }
  for message in header.messages {
    if message.msg_type == msg_attribute {
      let message_flags = message.flags.to_int()
      let data = if (message_flags & shared_message_flag) != 0 {
        self.shared_messages.resolve_shared_message(
          message.data,
          msg_attribute,
          [],
        )
      } else {
        message.data
      }
      attributes.push({
        name: editable_attribute_name(data),
        creation_order: next_order,
        flags: (message_flags & (0xFF ^ shared_message_flag)).to_byte(),
        data,
      })
      next_order = next_order + 1UL
    }
  }
  attributes
}

///|
fn editable_attribute_index(
  attributes : Array[EditableAttribute],
  name : String,
) -> Int32? {
  attributes.search_by(attribute => attribute.name == name)
}

///|
fn validate_editable_attribute_name(name : String) -> Unit raise Hdf5Error {
  let encoded = @utf8.encode(name)
  guard encoded.length() > 0 && encoded.length() < 65535 else {
    raise InvalidArgument("attribute name is empty or too long")
  }
  for byte in encoded {
    guard byte != b'\x00' else {
      raise InvalidArgument("attribute names cannot contain NUL")
    }
  }
}

///|
fn rewrite_attribute_name(
  data : Bytes,
  new_name : String,
) -> Bytes raise Hdf5Error {
  let reader = Reader::new(data)
  let version = reader.read_u8().to_int()
  guard version >= 1 && version <= 3 else {
    raise UnsupportedFeature("unsupported attribute message version")
  }
  let flags = if version == 1 {
    let _reserved = reader.read_u8()
    b'\x00'
  } else {
    reader.read_u8()
  }
  let name_size = reader.read_u16_le().to_int()
  let datatype_size = reader.read_u16_le().to_int()
  let dataspace_size = reader.read_u16_le().to_int()
  if version >= 3 {
    let _encoding = reader.read_u8()
  }
  let _old_name = reader.read_bytes(name_size)
  if version == 1 {
    reader.skip(round_up_to_eight(name_size) - name_size)
  }
  let datatype = reader.read_bytes(datatype_size)
  if version == 1 {
    reader.skip(round_up_to_eight(datatype_size) - datatype_size)
  }
  let dataspace = reader.read_bytes(dataspace_size)
  let value = reader.read_bytes(reader.remaining())
  let encoded_name = @utf8.encode(new_name)
  guard encoded_name.length() + 1 <= 65535 &&
    datatype.length() <= 65535 &&
    dataspace.length() <= 65535 else {
    raise InvalidArgument("renamed attribute message is too large")
  }
  let buffer = @buffer.Buffer::Buffer(
    size_hint=9 +
      encoded_name.length() +
      1 +
      datatype.length() +
      dataspace.length() +
      value.length(),
  )
  buffer.write_byte(b'\x03')
  buffer.write_byte(flags)
  buffer.write_uint16_le((encoded_name.length() + 1).to_uint16())
  buffer.write_uint16_le(datatype.length().to_uint16())
  buffer.write_uint16_le(dataspace.length().to_uint16())
  buffer.write_byte(b'\x01')
  writer_write_bytes(buffer, encoded_name)
  buffer.write_byte(b'\x00')
  writer_write_bytes(buffer, datatype)
  writer_write_bytes(buffer, dataspace)
  writer_write_bytes(buffer, value)
  buffer.to_bytes()
}

///|
fn attribute_values_contain_reference(values : Hdf5WriteValues) -> Bool {
  match values {
    ObjectReferenceValues(_) => true
    StructuredValues(datatype, _) => datatype_contains_reference(datatype)
    _ => false
  }
}

///|
fn Hdf5File::editable_reference_objects(
  self : Hdf5File,
) -> Array[WritableObject] raise Hdf5Error {
  let objects : Array[WritableObject] = [
    {
      path: "/",
      kind: WritableGroup,
      address: self.superblock.root_group_object_header,
      raw_address: self.superblock.root_group_object_header,
      dataspace: None,
    },
  ]
  for node in self.walk() {
    if node.address == 0UL || node.kind == Unknown {
      continue
    }
    let kind = match node.kind {
      Group => WritableGroup
      Dataset => WritableDatasetObject(0)
      Datatype => WritableCommittedDatatype(0)
      Unknown => continue
    }
    let dataspace = if node.kind == Dataset {
      Some(self.read_dataset_metadata(node.path).dataspace)
    } else {
      None
    }
    objects.push({
      path: node.path,
      kind,
      address: node.address,
      raw_address: node.address,
      dataspace,
    })
  }
  objects
}

///|
fn Hdf5File::validate_editable_attribute_reference_target(
  self : Hdf5File,
  path : String,
  name : String,
  targets : Array[WritableAttribute],
) -> Unit raise Hdf5Error {
  let canonical = canonical_writer_path(path)
  let attributes = self.attributes(canonical)
  guard attributes.search_by(attribute => attribute.name == name) is Some(_) else {
    raise InvalidArgument(
      "attribute reference attribute does not exist: " + canonical + "." + name,
    )
  }
  if targets.search_by(target => {
      target.object_path == canonical && target.name == name
    })
    is None {
    targets.push({
      object_path: canonical,
      name,
      dimensions: [],
      values: Int8Values([]),
      creation_order: 0UL,
    })
  }
}

///|
fn Hdf5File::collect_editable_attribute_references(
  self : Hdf5File,
  datatype : Datatype,
  value : DatasetValue,
  targets : Array[WritableAttribute],
) -> Unit raise Hdf5Error {
  match datatype.class {
    6 =>
      match value {
        Compound(fields) =>
          for index in 0..<fields.length() {
            self.collect_editable_attribute_references(
              datatype.compound_members[index].datatype,
              fields[index].value,
              targets,
            )
          }
        _ => ()
      }
    7 =>
      if datatype.reference_kind == 4 {
        match value {
          AttributeReference(Some(path), name) =>
            self.validate_editable_attribute_reference_target(
              path, name, targets,
            )
          _ => ()
        }
      }
    9 =>
      match value {
        Sequence(items) =>
          for item in items {
            self.collect_editable_attribute_references(
              datatype.base_types[0],
              item,
              targets,
            )
          }
        _ => ()
      }
    10 =>
      match value {
        Elements(items) =>
          for item in items {
            self.collect_editable_attribute_references(
              datatype.base_types[0],
              item,
              targets,
            )
          }
        _ => ()
      }
    _ => ()
  }
}

///|
fn encode_new_attribute_message(
  name : String,
  dimensions : Array[UInt64],
  values : Hdf5WriteValues,
  heap : WritableGlobalHeap,
  objects : Array[WritableObject],
  reference_address_size : Int32,
) -> Bytes raise Hdf5Error {
  validate_write_values(
    dimensions,
    values,
    reference_address_size=Some(reference_address_size),
  )
  let encoded = encode_write_values(
    values,
    heap~,
    objects~,
    reference_address_size~,
  )
  let encoded_name = @utf8.encode(name)
  let dataspace = encode_dataspace(dimensions)
  guard encoded_name.length() + 1 <= 65535 &&
    encoded.datatype.length() <= 65535 &&
    dataspace.length() <= 65535 else {
    raise InvalidArgument("attribute message is too large")
  }
  let buffer = @buffer.Buffer::Buffer(
    size_hint=9 +
      encoded_name.length() +
      1 +
      encoded.datatype.length() +
      dataspace.length() +
      encoded.payload.length(),
  )
  buffer.write_byte(b'\x03')
  buffer.write_byte(b'\x00')
  buffer.write_uint16_le((encoded_name.length() + 1).to_uint16())
  buffer.write_uint16_le(encoded.datatype.length().to_uint16())
  buffer.write_uint16_le(dataspace.length().to_uint16())
  buffer.write_byte(b'\x01')
  writer_write_bytes(buffer, encoded_name)
  buffer.write_byte(b'\x00')
  writer_write_bytes(buffer, encoded.datatype)
  writer_write_bytes(buffer, dataspace)
  writer_write_bytes(buffer, encoded.payload)
  buffer.to_bytes()
}

///|
fn editable_values_heap(
  values : Hdf5WriteValues,
  objects : Array[WritableObject],
  reference_address_size : Int32,
) -> WritableGlobalHeap raise Hdf5Error {
  let strings : Array[String] = []
  collect_heap_strings(strings, values)
  let empty = {
    address: 0UL,
    region_address: 0UL,
    strings,
    regions: [],
    sequences: [],
  }
  let sequences : Array[Bytes] = []
  match values {
    StructuredValues(datatype, items) =>
      for item in items {
        collect_heap_sequences(
          sequences,
          datatype,
          item,
          heap=empty,
          objects~,
          reference_address_size~,
        )
      }
    _ => ()
  }
  { address: 0UL, region_address: 0UL, strings, regions: [], sequences, }
}

///|
fn encode_editable_compact_attribute_info(
  offset_size : Int32,
  track_order : Bool,
  max_creation_order : UInt64,
) -> Bytes {
  let buffer = @buffer.Buffer::Buffer(size_hint=20)
  buffer.write_byte(b'\x00')
  buffer.write_byte(if track_order { b'\x01' } else { b'\x00' })
  if track_order {
    buffer.write_uint16_le(max_creation_order.to_uint16())
  }
  write_uint_width(buffer, heap_undefined_address(offset_size), offset_size)
  write_uint_width(buffer, heap_undefined_address(offset_size), offset_size)
  buffer.to_bytes()
}

///|
fn encode_dense_attribute_info(
  offset_size : Int32,
  max_creation_order : UInt64,
  heap_address : UInt64,
  name_index_address : UInt64,
  order_index_address : UInt64,
) -> Bytes {
  let buffer = @buffer.Buffer::Buffer(size_hint=2 + 2 + 3 * offset_size)
  buffer.write_byte(b'\x00')
  buffer.write_byte(b'\x03')
  buffer.write_uint16_le(max_creation_order.to_uint16())
  write_uint_width(buffer, heap_address, offset_size)
  write_uint_width(buffer, name_index_address, offset_size)
  write_uint_width(buffer, order_index_address, offset_size)
  buffer.to_bytes()
}

///|
fn encode_editable_dense_storage(
  entries : Array[WritableDenseHeapEntry],
  heap_address : UInt64,
  offset_size : Int32,
  length_size : Int32,
) -> (Bytes, Bytes, Bytes) raise Hdf5Error {
  guard writer_dense_entries_fit(entries) else {
    raise InvalidField("dense attribute metadata is not available")
  }
  writer_sort_dense_entries(entries)
  let record_size = 17
  let heap_offset_size = 5
  let max_heap_bits = 40
  let id_length = 8
  let minimum_block_size = 1024
  let direct_overhead = 4 +
    1 +
    offset_size +
    heap_offset_size +
    metadata_checksum_size
  let mut data_size = direct_overhead
  let mut local_count = 0
  for entry in entries {
    if entry.shared_id is None {
      local_count = local_count + 1
      data_size = data_size + entry.data.length()
    }
  }
  let block_size = writer_next_power_of_two(data_size, minimum_block_size)
  let max_direct_size = writer_next_power_of_two(block_size, 65536)
  let heap_header_size = writer_dense_heap_header_size(offset_size, length_size)
  let root_address = if local_count > 0 {
    heap_address + heap_header_size.to_uint64()
  } else {
    heap_undefined_address(offset_size)
  }
  let index_address = if local_count > 0 {
    heap_address + heap_header_size.to_uint64() + block_size.to_uint64()
  } else {
    heap_address + heap_header_size.to_uint64()
  }
  let heap_header = encode_writer_dense_heap_header(
    root_address,
    id_length,
    block_size,
    if local_count > 0 {
      block_size
    } else {
      0
    },
    max_direct_size,
    max_heap_bits,
    local_count,
    writer_dense_max_managed_size,
    4,
    offset_size~,
    length_size~,
  )
  let (direct_block, ids) = if local_count > 0 {
    encode_writer_dense_direct_block(
      heap_address,
      block_size,
      heap_offset_size,
      entries,
      offset_size~,
    )
  } else {
    (b"", encode_writer_dense_shared_ids(entries))
  }
  let records : Array[Bytes] = []
  for index in 0..<entries.length() {
    let record = @buffer.Buffer::Buffer(size_hint=record_size)
    writer_write_bytes(record, ids[index])
    record.write_byte(entries[index].flags)
    record.write_uint_le(entries[index].creation_order.to_uint())
    record.write_uint_le(entries[index].hash)
    guard record.length() == record_size else {
      raise InvalidField("dense attribute B-tree record size mismatch")
    }
    records.push(record.to_bytes())
  }
  let order_indices : Array[Int32] = []
  for index in 0..<entries.length() {
    order_indices.push(index)
  }
  order_indices.sort_by((left, right) => {
    let left_order = entries[left].creation_order
    let right_order = entries[right].creation_order
    if left_order < right_order {
      -1
    } else if left_order > right_order {
      1
    } else if entries[left].name < entries[right].name {
      -1
    } else if entries[left].name > entries[right].name {
      1
    } else {
      0
    }
  })
  let order_records : Array[Bytes] = []
  for index in order_indices {
    let order_record = @buffer.Buffer::Buffer(size_hint=13)
    writer_write_bytes(order_record, ids[index])
    order_record.write_byte(entries[index].flags)
    order_record.write_uint_le(entries[index].creation_order.to_uint())
    guard order_record.length() == 13 else {
      raise InvalidField("dense attribute order B-tree record size mismatch")
    }
    order_records.push(order_record.to_bytes())
  }
  let node_size = writer_next_power_of_two(
    10 + records.length() * record_size,
    512,
  )
  let btree_header_size = (22 + offset_size + length_size).to_uint64()
  let node_address = index_address + btree_header_size
  let btree_header = encode_writer_dense_btree_header(
    8,
    node_size,
    record_size,
    node_address,
    records.length(),
    offset_size~,
    length_size~,
  )
  let btree_node = encode_writer_dense_btree_node(
    8, node_size, record_size, records,
  )
  let heap_bytes = @buffer.Buffer::Buffer(
    size_hint=heap_header.length() + direct_block.length(),
  )
  writer_write_bytes(heap_bytes, heap_header)
  writer_write_bytes(heap_bytes, direct_block)
  let index_bytes = @buffer.Buffer::Buffer(
    size_hint=btree_header.length() + btree_node.length(),
  )
  writer_write_bytes(index_bytes, btree_header)
  writer_write_bytes(index_bytes, btree_node)
  let order_index_address = index_address + index_bytes.length().to_uint64()
  let order_node_size = writer_next_power_of_two(
    10 + order_records.length() * 13,
    512,
  )
  let order_node_address = order_index_address + btree_header_size
  let order_header = encode_writer_dense_btree_header(
    9,
    order_node_size,
    13,
    order_node_address,
    order_records.length(),
    offset_size~,
    length_size~,
  )
  let order_node = encode_writer_dense_btree_node(
    9, order_node_size, 13, order_records,
  )
  let order_bytes = @buffer.Buffer::Buffer(
    size_hint=order_header.length() + order_node.length(),
  )
  writer_write_bytes(order_bytes, order_header)
  writer_write_bytes(order_bytes, order_node)
  (heap_bytes.to_bytes(), index_bytes.to_bytes(), order_bytes.to_bytes())
}

///|
fn encode_edit_message(
  version : Int32,
  order_size : Int32,
  msg_type : Int32,
  flags : Byte,
  data : Bytes,
  creation_order : UInt64,
) -> Bytes raise Hdf5Error {
  let padding = if version == 1 {
    round_up_to_eight(data.length()) - data.length()
  } else {
    0
  }
  let encoded_size = data.length() + padding
  guard encoded_size <= 65535 else {
    raise InvalidField("object header message is too large")
  }
  let buffer = @buffer.Buffer::Buffer(
    size_hint=(if version == 1 { 8 } else { 4 + order_size }) + encoded_size,
  )
  if version == 1 {
    buffer.write_uint16_le(msg_type.to_uint16())
    buffer.write_uint16_le(encoded_size.to_uint16())
    buffer.write_byte(flags)
    writer_write_zeros(buffer, 3)
  } else {
    buffer.write_byte(msg_type.to_byte())
    buffer.write_uint16_le(encoded_size.to_uint16())
    buffer.write_byte(flags)
    if order_size == 2 {
      guard creation_order <= 65535UL else {
        raise InvalidField("attribute creation order exceeds its header field")
      }
      buffer.write_uint16_le(creation_order.to_uint16())
    }
  }
  writer_write_bytes(buffer, data)
  if padding > 0 {
    writer_write_zeros(buffer, padding)
  }
  buffer.to_bytes()
}

///|
fn object_header_rewrite_plan(
  bytes : Bytes,
  offset : Int32,
  offset_size : Int32,
  length_size : Int32,
) -> ObjectHeaderRewritePlan raise Hdf5Error {
  let reader = Reader::at(bytes, offset)
  let first = reader.read_u8()
  if first == b'\x01' {
    let _reserved = reader.read_u8()
    let _total_messages = reader.read_u16_le()
    let _reference_count = reader.read_u32_le()
    let header_size = reader.read_u32_le().reinterpret_as_int()
    let _padding = reader.read_u32_le()
    guard header_size >= 0 else {
      raise InvalidField("negative object header size")
    }
    let body_start = reader.position()
    guard body_start.to_int64() + header_size.to_int64() + 4L <=
      bytes.length().to_int64() else {
      raise UnexpectedEof(body_start.to_int64())
    }
    guard header_size >= 8 + offset_size + length_size else {
      raise UnsupportedFeature(
        "object header has no room for a continuation message",
      )
    }
    {
      version: 1,
      offset,
      flags: 0,
      body_start,
      body_size: header_size,
      checksum_offset: -1,
      track_order: false,
    }
  } else if first == b'O' {
    guard reader.read_bytes(3) == b"HDR" else {
      raise InvalidField("expected OHDR signature")
    }
    guard reader.read_u8() == b'\x02' else {
      raise InvalidField("expected object header version 2")
    }
    let flags = reader.read_u8().to_int()
    if (flags & 0x20) != 0 {
      reader.skip(16)
    }
    if (flags & 0x10) != 0 {
      reader.skip(4)
    }
    let chunk_size = reader.read_uint_le(1 << (flags & 0x03))
    guard chunk_size <= 0x7FFFFFFFUL else {
      raise InvalidField("object header chunk size is too large")
    }
    let body_start = reader.position()
    let body_size = chunk_size.to_int()
    let checksum_offset = body_start + body_size
    guard checksum_offset.to_int64() + metadata_checksum_size.to_int64() <=
      bytes.length().to_int64() else {
      raise UnexpectedEof(checksum_offset.to_int64())
    }
    let tracked = (flags & 0x04) != 0
    let order_size = if tracked { 2 } else { 0 }
    guard body_size >= 4 + order_size + offset_size + length_size else {
      raise UnsupportedFeature(
        "object header has no room for a continuation message",
      )
    }
    {
      version: 2,
      offset,
      flags,
      body_start,
      body_size,
      checksum_offset,
      track_order: tracked,
    }
  } else {
    raise InvalidField("unknown object header version")
  }
}

///|
fn append_nil_padding(
  body : @buffer.Buffer,
  remaining : Int32,
  version : Int32,
  order_size : Int32,
) -> Int32 {
  let header_size = if version == 1 { 8 } else { 4 + order_size }
  let mut left = remaining
  let mut count = 0
  while left >= header_size {
    let raw_data = if left - header_size <= 65535 {
      left - header_size
    } else {
      65535
    }
    let data = if version == 1 { raw_data - raw_data % 8 } else { raw_data }
    if version == 1 {
      body.write_uint16_le(0U.to_uint16())
      body.write_uint16_le(data.to_uint16())
      body.write_byte(b'\x00')
      writer_write_zeros(body, 3)
    } else {
      body.write_byte(b'\x00')
      body.write_uint16_le(data.to_uint16())
      body.write_byte(b'\x00')
      if order_size == 2 {
        body.write_uint16_le(0U.to_uint16())
      }
    }
    writer_write_zeros(body, data)
    left = left - header_size - data
    count = count + 1
  }
  if left > 0 {
    writer_write_zeros(body, left)
  }
  count
}

///|
fn rewrite_initial_object_header(
  output : FixedArray[Byte],
  plan : ObjectHeaderRewritePlan,
  offset_size : Int32,
  length_size : Int32,
  continuation_address : UInt64,
  continuation_length : UInt64,
  tracking : Bool,
  new_message_count : Int32,
) -> Unit raise Hdf5Error {
  let order_size = if plan.version == 2 && tracking { 2 } else { 0 }
  let header_size = if plan.version == 1 { 8 } else { 4 + order_size }
  guard plan.body_size >= header_size + offset_size + length_size else {
    raise UnsupportedFeature(
      "object header has no room for a continuation message",
    )
  }
  guard editable_uint_fits_width(continuation_length, length_size) else {
    raise UnsupportedFeature("object header continuation is too large")
  }
  let body = @buffer.Buffer::Buffer(size_hint=plan.body_size)
  if plan.version == 1 {
    body.write_uint16_le(msg_object_continuation.to_uint16())
    body.write_uint16_le((offset_size + length_size).to_uint16())
    body.write_byte(b'\x00')
    writer_write_zeros(body, 3)
  } else {
    body.write_byte(msg_object_continuation.to_byte())
    body.write_uint16_le((offset_size + length_size).to_uint16())
    body.write_byte(b'\x00')
    if order_size == 2 {
      body.write_uint16_le(0U.to_uint16())
    }
  }
  write_uint_width(body, continuation_address, offset_size)
  write_uint_width(body, continuation_length, length_size)
  let nil_count = append_nil_padding(
    body,
    plan.body_size - header_size - offset_size - length_size,
    plan.version,
    order_size,
  )
  let encoded = body.to_bytes()
  guard encoded.length() == plan.body_size else {
    raise InvalidField("object header rewrite size mismatch")
  }
  if plan.version == 2 {
    let flags = if tracking { plan.flags | 0x0C } else { plan.flags & 0xF3 }
    patch_little_endian_uint(
      output,
      plan.offset + 5,
      1,
      flags.to_uint64(),
      "object header flags",
    )
    patch_heap_output_bytes(output, plan.body_start, encoded)
    patch_metadata_checksum_in_place(output, plan.offset, plan.checksum_offset)
  } else {
    patch_heap_output_bytes(output, plan.body_start, encoded)
    patch_little_endian_uint(
      output,
      plan.offset + 2,
      2,
      (new_message_count + nil_count + 1).to_uint64(),
      "object header message count",
    )
  }
}

///|
fn append_contiguous_dense_metadata(
  tail : @buffer.Buffer,
  file_length : Int32,
  heap_bytes : Bytes,
  index_bytes : Bytes,
  order_bytes : Bytes,
) -> UInt64 raise Hdf5Error {
  let start = append_aligned_bytes_to_tail(
    tail, file_length, heap_bytes, "append dense attribute metadata",
  )
  tail.write_bytes(index_bytes.get_view().unwrap())
  tail.write_bytes(order_bytes.get_view().unwrap())
  start
}

///|
/// Rebuilds all attributes of an object into compact or dense storage and
/// replaces the object's initial header chunk with a continuation. Existing
/// non-attribute messages are copied into the new continuation chunk.
fn Hdf5File::write_rebuilt_attributes(
  self : Hdf5File,
  address : UInt64,
  header : ObjectHeader,
  attributes : Array[EditableAttribute],
  tail : @buffer.Buffer,
) -> Bytes raise Hdf5Error {
  let offset_size = self.superblock.offset_size
  let length_size = self.superblock.length_size
  let header_offset = address_to_offset(address, self.superblock.base_address)
  let plan = object_header_rewrite_plan(
    self.bytes,
    header_offset,
    offset_size,
    length_size,
  )
  guard attributes.length() <= 65535 else {
    raise InvalidArgument("too many attributes for one object")
  }
  let mut max_creation_order = 0UL
  for attribute in attributes {
    if attribute.creation_order > max_creation_order {
      max_creation_order = attribute.creation_order
    }
  }
  guard max_creation_order <= 65535UL else {
    raise InvalidField("attribute creation order exceeds the file format")
  }
  let mut dense = attributes.length() > writer_dense_threshold
  if dense {
    for attribute in attributes {
      if attribute.data.length() > writer_dense_max_managed_size {
        dense = false
      }
    }
  }
  if !dense {
    for attribute in attributes {
      let encoded_size = if plan.version == 1 {
        round_up_to_eight(attribute.data.length())
      } else {
        attribute.data.length()
      }
      guard encoded_size <= 65535 else {
        raise UnsupportedFeature(
          "attribute message is too large for compact storage",
        )
      }
    }
  }
  let mut dense_addresses : (UInt64, UInt64, UInt64)? = None
  if dense {
    let entries : Array[WritableDenseHeapEntry] = []
    for attribute in attributes {
      entries.push({
        name: attribute.name,
        hash: writer_name_hash(attribute.name),
        data: attribute.data,
        creation_order: attribute.creation_order,
        flags: attribute.flags,
        shared_id: None,
      })
    }
    let heap_offset = next_aligned_tail_start(
      self.bytes.length(),
      tail.length(),
      "append dense attribute metadata",
    )
    let heap_address = file_offset_to_stored_address(
      heap_offset,
      self.superblock.base_address,
    )
    let (heap_bytes, index_bytes, order_bytes) = encode_editable_dense_storage(
      entries, heap_address, offset_size, length_size,
    )
    let start = append_contiguous_dense_metadata(
      tail,
      self.bytes.length(),
      heap_bytes,
      index_bytes,
      order_bytes,
    )
    guard start == heap_offset else {
      raise InvalidField("dense attribute metadata address changed")
    }
    let index_address = heap_address + heap_bytes.length().to_uint64()
    let order_address = index_address + index_bytes.length().to_uint64()
    dense_addresses = Some((heap_address, index_address, order_address))
  }
  let tracking = plan.track_order || dense
  let order_size = if plan.version == 2 && tracking { 2 } else { 0 }
  let new_messages : Array[Bytes] = []
  for message in header.messages {
    if message.msg_type == _msg_nil ||
      message.msg_type == msg_object_continuation ||
      message.msg_type == msg_attribute_info ||
      message.msg_type == msg_attribute {
      continue
    }
    new_messages.push(
      encode_edit_message(
        plan.version,
        order_size,
        message.msg_type,
        message.flags,
        message.data,
        0UL,
      ),
    )
  }
  if attributes.length() > 0 {
    match dense_addresses {
      Some((heap_address, index_address, order_address)) =>
        new_messages.push(
          encode_edit_message(
            plan.version,
            order_size,
            msg_attribute_info,
            b'\x00',
            encode_dense_attribute_info(
              offset_size, max_creation_order, heap_address, index_address, order_address,
            ),
            0UL,
          ),
        )
      None => {
        new_messages.push(
          encode_edit_message(
            plan.version,
            order_size,
            msg_attribute_info,
            b'\x00',
            encode_editable_compact_attribute_info(
              offset_size, tracking, max_creation_order,
            ),
            0UL,
          ),
        )
        for attribute in attributes {
          new_messages.push(
            encode_edit_message(
              plan.version,
              order_size,
              msg_attribute,
              attribute.flags,
              attribute.data,
              attribute.creation_order,
            ),
          )
        }
      }
    }
  }
  let mut message_bytes = 0
  for message in new_messages {
    message_bytes = message_bytes + message.length()
  }
  let chunk_offset = next_aligned_tail_start(
    self.bytes.length(),
    tail.length(),
    "append object header continuation",
  )
  let chunk = @buffer.Buffer::Buffer(
    size_hint=message_bytes + (if plan.version == 1 { 0 } else { 8 }),
  )
  if plan.version == 1 {
    for message in new_messages {
      writer_write_bytes(chunk, message)
    }
  } else {
    writer_write_bytes(chunk, b"OCHK")
    for message in new_messages {
      writer_write_bytes(chunk, message)
    }
    chunk.write_uint_le(metadata_checksum(chunk.to_bytes(), 0, chunk.length()))
  }
  let chunk_bytes = chunk.to_bytes()
  guard editable_uint_fits_width(chunk_bytes.length().to_uint64(), length_size) else {
    raise UnsupportedFeature("object header continuation is too large")
  }
  let start = append_aligned_bytes_to_tail(
    tail,
    self.bytes.length(),
    chunk_bytes,
    "append object header continuation",
  )
  guard start == chunk_offset else {
    raise InvalidField("object header continuation address changed")
  }
  let continuation_address = file_offset_to_stored_address(
    start,
    self.superblock.base_address,
  )
  let output = self.bytes.to_fixedarray()
  rewrite_initial_object_header(
    output,
    plan,
    offset_size,
    length_size,
    continuation_address,
    chunk_bytes.length().to_uint64(),
    tracking,
    new_messages.length(),
  )
  self.finish_appended_image(output, tail)
}

///|
/// Adds a new attribute to an existing group, dataset or committed datatype.
/// Fixed-size values and variable-length UTF-8 strings or sequences are
/// supported. Storage automatically switches between compact and dense forms
/// at the same threshold used by the writer.
pub fn Hdf5File::add_attribute(
  self : Hdf5File,
  path : String,
  name : String,
  dimensions : Array[UInt64],
  values : Hdf5WriteValues,
) -> Bytes raise Hdf5Error {
  guard self.source is None else {
    raise InvalidArgument("attribute editing requires a bytes-backed Hdf5File")
  }
  validate_editable_attribute_name(name)
  let node = self.resolve(path)
  let header = self.object_header_at(node.address)
  let attributes = self.editable_attributes(header)
  guard editable_attribute_index(attributes, name) is None else {
    raise InvalidArgument("attribute name already exists: " + name)
  }
  let mut creation_order = 0UL
  for attribute in attributes {
    if attribute.creation_order >= creation_order {
      creation_order = attribute.creation_order + 1UL
    }
  }
  guard creation_order <= 65535UL else {
    raise InvalidField("attribute creation order exceeds the file format")
  }
  let tail = @buffer.Buffer::Buffer()
  let reference_address_size = self.superblock.offset_size
  let has_references = attribute_values_contain_reference(values)
  let objects = if has_references {
    self.editable_reference_objects()
  } else {
    []
  }
  let reference_attributes : Array[WritableAttribute] = []
  if has_references {
    match values {
      StructuredValues(datatype, items) =>
        for item in items {
          self.collect_editable_attribute_references(
            datatype, item, reference_attributes,
          )
        }
      _ => ()
    }
  }
  let initial_heap = editable_values_heap(
    values, objects, reference_address_size,
  )
  if has_references {
    match values {
      StructuredValues(datatype, items) =>
        for item in items {
          collect_heap_regions(
            initial_heap.regions,
            datatype,
            item,
            objects,
            reference_attributes,
            reference_address_size~,
          )
        }
      _ => ()
    }
  }
  let heap_address = if initial_heap.strings.length() > 0 ||
    initial_heap.sequences.length() > 0 {
    let heap_payload = encode_global_heap(initial_heap)
    let heap_offset = append_aligned_bytes_to_tail(
      tail,
      self.bytes.length(),
      heap_payload,
      "append an attribute global heap",
    )
    file_offset_to_stored_address(heap_offset, self.superblock.base_address)
  } else {
    0UL
  }
  let region_address = if initial_heap.regions.length() > 0 {
    let heap_payload = encode_region_heap(initial_heap)
    let heap_offset = append_aligned_bytes_to_tail(
      tail,
      self.bytes.length(),
      heap_payload,
      "append an attribute region-reference heap",
    )
    file_offset_to_stored_address(heap_offset, self.superblock.base_address)
  } else {
    0UL
  }
  let heap = {
    address: heap_address,
    region_address,
    strings: initial_heap.strings,
    regions: initial_heap.regions,
    sequences: initial_heap.sequences,
  }
  let data = encode_new_attribute_message(
    name, dimensions, values, heap, objects, reference_address_size,
  )
  attributes.push({ name, creation_order, flags: b'\x00', data, })
  self.write_rebuilt_attributes(node.address, header, attributes, tail)
}

///|
/// Deletes an attribute and rebuilds the object's compact or dense storage.
pub fn Hdf5File::delete_attribute(
  self : Hdf5File,
  path : String,
  name : String,
) -> Bytes raise Hdf5Error {
  guard self.source is None else {
    raise InvalidArgument("attribute editing requires a bytes-backed Hdf5File")
  }
  let node = self.resolve(path)
  let header = self.object_header_at(node.address)
  let attributes = self.editable_attributes(header)
  let index = match editable_attribute_index(attributes, name) {
    Some(index) => index
    None => raise InvalidArgument("attribute does not exist: " + name)
  }
  let rebuilt : Array[EditableAttribute] = []
  for position in 0..<attributes.length() {
    if position != index {
      rebuilt.push(attributes[position])
    }
  }
  self.write_rebuilt_attributes(
    node.address,
    header,
    rebuilt,
    @buffer.Buffer::Buffer(),
  )
}

///|
/// Renames an attribute while preserving its datatype, dataspace, value and
/// creation order. Compact/dense storage is rebuilt consistently.
pub fn Hdf5File::rename_attribute(
  self : Hdf5File,
  path : String,
  old_name : String,
  new_name : String,
) -> Bytes raise Hdf5Error {
  guard self.source is None else {
    raise InvalidArgument("attribute editing requires a bytes-backed Hdf5File")
  }
  validate_editable_attribute_name(new_name)
  let node = self.resolve(path)
  let header = self.object_header_at(node.address)
  let attributes = self.editable_attributes(header)
  let index = match editable_attribute_index(attributes, old_name) {
    Some(index) => index
    None => raise InvalidArgument("attribute does not exist: " + old_name)
  }
  if old_name == new_name {
    return self.bytes
  }
  guard editable_attribute_index(attributes, new_name) is None else {
    raise InvalidArgument("attribute name already exists: " + new_name)
  }
  attributes[index] = {
    name: new_name,
    creation_order: attributes[index].creation_order,
    flags: attributes[index].flags,
    data: rewrite_attribute_name(attributes[index].data, new_name),
  }
  self.write_rebuilt_attributes(
    node.address,
    header,
    attributes,
    @buffer.Buffer::Buffer(),
  )
}
