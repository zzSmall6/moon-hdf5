///|
/// Encodes a version 2 object header for a newly created object. The initial
/// chunk is padded so later edits have room for a continuation message without
/// relocating the object.
fn encode_created_object_header(
  messages : Array[RebuiltObjectMessage],
) -> Bytes raise Hdf5Error {
  let body = @buffer.Buffer::Buffer(size_hint=128)
  for message in messages {
    writer_write_bytes(
      body,
      encode_edit_message(
        2,
        0,
        message.msg_type,
        message.flags,
        message.data,
        message.creation_order,
      ),
    )
  }
  let minimum_body_size : Int32 = 128
  if minimum_body_size - body.length() >= 4 {
    let nil_data_size = minimum_body_size - body.length() - 4
    body.write_byte(b'\x00')
    body.write_uint16_le(nil_data_size.to_uint16())
    body.write_byte(b'\x00')
    writer_write_zeros(body, nil_data_size)
  }
  guard body.length() <= 65535 else {
    raise UnsupportedFeature("new group object header is too large")
  }
  let header = @buffer.Buffer::Buffer(size_hint=body.length() + 12)
  writer_write_bytes(header, b"OHDR")
  header.write_byte(b'\x02')
  header.write_byte(b'\x01')
  header.write_uint16_le(body.length().to_uint16())
  writer_write_bytes(header, body.to_bytes())
  header.write_uint_le(metadata_checksum(header.to_bytes(), 0, header.length()))
  header.to_bytes()
}

///|
fn encode_created_group_header(offset_size : Int32) -> Bytes raise Hdf5Error {
  encode_created_object_header([
    {
      msg_type: msg_link_info,
      flags: b'\x00',
      data: encode_editable_compact_link_info(offset_size, false, 0UL),
      creation_order: 0UL,
    },
  ])
}

///|
fn encode_created_contiguous_layout(
  address : UInt64,
  size : Int32,
  offset_size : Int32,
  length_size : Int32,
) -> Bytes raise Hdf5Error {
  guard editable_uint_fits_width(address, offset_size) else {
    raise UnsupportedFeature("dataset address exceeds the file address width")
  }
  guard editable_uint_fits_width(size.to_uint64(), length_size) else {
    raise UnsupportedFeature("dataset size exceeds the file length width")
  }
  let buffer = @buffer.Buffer::Buffer(size_hint=2 + offset_size + length_size)
  buffer.write_byte(b'\x03')
  buffer.write_byte(b'\x01')
  write_uint_width(buffer, address, offset_size)
  write_uint_width(buffer, size.to_uint64(), length_size)
  buffer.to_bytes()
}

///|
fn created_dataset_values_supported(values : Hdf5WriteValues) -> Bool {
  match values {
    VariableStringValues(_) | ObjectReferenceValues(_) => true
    StructuredValues(datatype, _) =>
      !datatype_contains_vlen(datatype) &&
      !datatype_contains_reference(datatype)
    _ => true
  }
}

///|
fn Hdf5File::created_reference_objects(
  self : Hdf5File,
) -> Array[WritableObject] raise Hdf5Error {
  let objects : Array[WritableObject] = [
    {
      path: "/",
      kind: WritableGroup,
      address: self.root_address(),
      raw_address: 0UL,
      dataspace: None,
    },
  ]
  for node in self.walk() {
    objects.push({
      path: node.path,
      kind: WritableGroup,
      address: node.address,
      raw_address: 0UL,
      dataspace: None,
    })
  }
  objects
}

///|
/// Appends a newly encoded object header to the current output tail and
/// returns its stored file address.
fn Hdf5File::append_created_object(
  self : Hdf5File,
  tail : @buffer.Buffer,
  header_bytes : Bytes,
) -> UInt64 raise Hdf5Error {
  let start = append_aligned_bytes_to_tail(
    tail,
    self.bytes.length(),
    header_bytes,
    "append a new object",
  )
  let address = file_offset_to_stored_address(
    start,
    self.superblock.base_address,
  )
  guard editable_uint_fits_width(address, self.superblock.offset_size) else {
    raise UnsupportedFeature(
      "new object address exceeds the file address width",
    )
  }
  address
}

///|
/// Adds a new empty group below `parent_path` and returns the updated file.
///
/// The new object header is appended first, then the parent group's complete
/// link collection is rebuilt with a hard link to it. Existing symbol-table
/// parents are converted to link-message storage by the normal link rebuild
/// path. Old metadata remains unreferenced because free-space reclamation is not
/// implemented.
pub fn Hdf5File::add_group(
  self : Hdf5File,
  parent_path : String,
  name : String,
) -> Bytes raise Hdf5Error {
  validate_editable_link_name(name)
  let (parent_address, header, links) = self.editable_group(parent_path)
  guard editable_link_index(links, name) is None else {
    raise InvalidArgument("link name already exists: " + name)
  }
  let tail = @buffer.Buffer::Buffer()
  let object_address = self.append_created_object(
    tail,
    encode_created_group_header(self.superblock.offset_size),
  )
  let mut creation_order = 0UL
  for link in links {
    if link.creation_order >= creation_order {
      creation_order = link.creation_order + 1UL
    }
  }
  links.push({
    link: {
      name,
      object_header_address: object_address,
      cache_type: 0U,
      link_target: None,
      external_path: None,
    },
    creation_order,
  })
  self.write_rebuilt_links(parent_address, header, links, tail)
}

///|
/// Adds a fixed-size contiguous dataset below `parent_path` and returns the
/// updated file.
///
/// Numeric, fixed-string, complex, fixed-size structured, variable-length
/// string and object-reference values are supported. The payload is appended
/// first and the new dataset object header links it through a contiguous layout
/// message using the file's address and length widths. Variable-length string
/// descriptors use the file address width, and object references use the file
/// address width rather than a host-sized reference slot. Nested variable-length
/// or reference members in structured values remain unsupported.
pub fn Hdf5File::add_dataset(
  self : Hdf5File,
  parent_path : String,
  name : String,
  dimensions : Array[UInt64],
  values : Hdf5WriteValues,
) -> Bytes raise Hdf5Error {
  validate_editable_link_name(name)
  let (parent_address, header, links) = self.editable_group(parent_path)
  guard editable_link_index(links, name) is None else {
    raise InvalidArgument("link name already exists: " + name)
  }
  guard created_dataset_values_supported(values) else {
    raise UnsupportedFeature(
      "dataset creation does not support nested variable-length or reference values",
    )
  }
  let offset_size = self.superblock.offset_size
  let length_size = self.superblock.length_size
  validate_write_values(
    dimensions,
    values,
    reference_address_size=Some(offset_size),
  )
  let tail = @buffer.Buffer::Buffer()
  let strings : Array[String] = []
  collect_heap_strings(strings, values)
  let mut heap = empty_writable_heap()
  if strings.length() > 0 {
    let heap_offset = next_aligned_tail_start(
      self.bytes.length(),
      tail.length(),
      "append dataset heap",
    )
    let heap_address = file_offset_to_stored_address(
      heap_offset,
      self.superblock.base_address,
    )
    guard editable_uint_fits_width(heap_address, offset_size) else {
      raise UnsupportedFeature(
        "dataset heap address exceeds the file address width",
      )
    }
    heap = {
      address: heap_address,
      region_address: 0UL,
      strings,
      regions: [],
      sequences: [],
    }
    ignore(
      append_aligned_bytes_to_tail(
        tail,
        self.bytes.length(),
        encode_global_heap(heap),
        "append dataset heap",
      ),
    )
  }
  let objects = if values is ObjectReferenceValues(_) {
    self.created_reference_objects()
  } else {
    []
  }
  let encoded = encode_write_values(
    values,
    heap~,
    objects~,
    reference_address_size=offset_size,
    vlen_address_size=offset_size,
  )
  let data_address = if encoded.payload.length() == 0 {
    heap_undefined_address(offset_size)
  } else {
    let start = append_aligned_bytes_to_tail(
      tail,
      self.bytes.length(),
      encoded.payload,
      "append dataset storage",
    )
    file_offset_to_stored_address(start, self.superblock.base_address)
  }
  let object_address = self.append_created_object(
    tail,
    encode_created_object_header([
      {
        msg_type: msg_dataspace,
        flags: b'\x00',
        data: encode_dataspace(dimensions),
        creation_order: 0UL,
      },
      {
        msg_type: msg_datatype,
        flags: b'\x00',
        data: encoded.datatype,
        creation_order: 0UL,
      },
      {
        msg_type: msg_data_layout,
        flags: b'\x00',
        data: encode_created_contiguous_layout(
          data_address,
          encoded.payload.length(),
          offset_size,
          length_size,
        ),
        creation_order: 0UL,
      },
    ]),
  )
  let mut creation_order = 0UL
  for link in links {
    if link.creation_order >= creation_order {
      creation_order = link.creation_order + 1UL
    }
  }
  links.push({
    link: {
      name,
      object_header_address: object_address,
      cache_type: 0U,
      link_target: None,
      external_path: None,
    },
    creation_order,
  })
  self.write_rebuilt_links(parent_address, header, links, tail)
}
