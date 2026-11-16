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
    raise UnsupportedFeature("new object header is too large")
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
/// Encodes a newly created version 2 object header at `start`, extending large
/// message collections through checksummed continuation chunks.
fn encode_created_object_header_with_continuation(
  messages : Array[RebuiltObjectMessage],
  start : UInt64,
  offset_size : Int32,
  length_size : Int32,
) -> Bytes raise Hdf5Error {
  let payload = @buffer.Buffer::Buffer()
  for message in messages {
    writer_write_bytes(
      payload,
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
  encode_object_header(payload.to_bytes(), start, offset_size~, length_size~)
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
fn validate_created_fill_layout(
  fill : Hdf5WriteValues?,
  address_size : Int32,
) -> Unit raise Hdf5Error {
  match fill {
    Some(StructuredValues(datatype, _)) =>
      if address_size == 4 && datatype_contains_vlen(datatype) {
        raise UnsupportedFeature(
          "custom fills for structured variable-length values require eight-byte file addresses",
        )
      }
    _ => ()
  }
}

///|
fn Hdf5File::collect_created_heap_values(
  self : Hdf5File,
  strings : Array[String],
  regions : Array[Bytes],
  reference_attributes : Array[WritableAttribute],
  values : Hdf5WriteValues,
  objects : Array[WritableObject],
  reference_address_size : Int32,
) -> Unit raise Hdf5Error {
  collect_heap_strings(strings, values)
  match values {
    StructuredValues(datatype, items) =>
      for item in items {
        collect_heap_structured_strings(strings, datatype, item)
        self.collect_editable_attribute_references(
          datatype, item, reference_attributes,
        )
        collect_heap_regions(
          regions,
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

///|
fn Hdf5File::append_created_values_heap(
  self : Hdf5File,
  tail : @buffer.Buffer,
  values : Hdf5WriteValues,
  fill_value : Hdf5WriteValues?,
  objects : Array[WritableObject],
  address_size : Int32,
) -> WritableGlobalHeap raise Hdf5Error {
  let strings : Array[String] = []
  let regions : Array[Bytes] = []
  let reference_attributes : Array[WritableAttribute] = []
  self.collect_created_heap_values(
    strings, regions, reference_attributes, values, objects, address_size,
  )
  match fill_value {
    Some(fill) =>
      self.collect_created_heap_values(
        strings, regions, reference_attributes, fill, objects, address_size,
      )
    None => ()
  }
  let region_address = if regions.length() > 0 {
    let start = append_aligned_bytes_to_tail(
      tail,
      self.bytes.length(),
      encode_heap_objects(regions, length_size=self.superblock.length_size),
      "append dataset region-reference heap",
    )
    file_offset_to_stored_address(start, self.superblock.base_address)
  } else {
    0UL
  }
  let base_heap : WritableGlobalHeap = {
    address: 0UL,
    region_address,
    strings,
    regions,
    sequences: [],
    external_references: [],
  }
  let sequences : Array[Bytes] = []
  let collect_sequences = fn(values : Hdf5WriteValues) -> Unit raise Hdf5Error {
    match values {
      StructuredValues(datatype, items) =>
        for item in items {
          collect_heap_sequences(
            sequences,
            datatype,
            item,
            heap=base_heap,
            objects~,
            reference_address_size=address_size,
            vlen_address_size=address_size,
          )
        }
      _ => ()
    }
  }
  collect_sequences(values)
  if fill_value is Some(fill) {
    collect_sequences(fill)
  }
  let heap_address = if strings.length() > 0 || sequences.length() > 0 {
    let start = append_aligned_bytes_to_tail(
      tail,
      self.bytes.length(),
      encode_global_heap(
        {
          address: 0UL,
          region_address,
          strings,
          regions,
          sequences,
          external_references: [],
        },
        length_size=self.superblock.length_size,
      ),
      "append dataset global heap",
    )
    file_offset_to_stored_address(start, self.superblock.base_address)
  } else {
    0UL
  }
  {
    address: heap_address,
    region_address,
    strings,
    regions,
    sequences,
    external_references: [],
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
    let (kind, dataspace) = match node.kind {
      Dataset => {
        let metadata = self.read_dataset_metadata(node.path)
        (WritableDatasetObject(0), Some(metadata.dataspace))
      }
      _ => (WritableGroup, None)
    }
    objects.push({
      path: node.path,
      kind,
      address: node.address,
      raw_address: 0UL,
      dataspace,
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
fn Hdf5File::append_created_object_with_messages(
  self : Hdf5File,
  tail : @buffer.Buffer,
  messages : Array[RebuiltObjectMessage],
) -> UInt64 raise Hdf5Error {
  let start = next_aligned_tail_start(
    self.bytes.length(),
    tail.length(),
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
  self.append_created_object(
    tail,
    encode_created_object_header_with_continuation(
      messages,
      address,
      self.superblock.offset_size,
      self.superblock.length_size,
    ),
  )
}

///|
fn Hdf5File::append_created_link(
  self : Hdf5File,
  parent_address : UInt64,
  header : ObjectHeader,
  links : Array[EditableLink],
  object_address : UInt64,
  name : String,
  tail : @buffer.Buffer,
) -> Bytes raise Hdf5Error {
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
  self.append_created_link(
    parent_address, header, links, object_address, name, tail,
  )
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
/// address width rather than a host-sized reference slot. Nested
/// variable-length and reference members in compound, fixed-array and sequence
/// values are encoded recursively. An optional fill value is written through
/// the normal fill-value message.
pub fn Hdf5File::add_dataset(
  self : Hdf5File,
  parent_path : String,
  name : String,
  dimensions : Array[UInt64],
  values : Hdf5WriteValues,
  fill_value? : Hdf5WriteValues,
) -> Bytes raise Hdf5Error {
  validate_editable_link_name(name)
  let (parent_address, header, links) = self.editable_group(parent_path)
  guard editable_link_index(links, name) is None else {
    raise InvalidArgument("link name already exists: " + name)
  }
  let offset_size = self.superblock.offset_size
  let length_size = self.superblock.length_size
  validate_write_values(
    dimensions,
    values,
    reference_address_size=Some(offset_size),
  )
  let copied_fill = validated_write_fill(
    values,
    fill_value,
    reference_address_size=Some(offset_size),
  )
  validate_created_fill_layout(copied_fill, offset_size)
  let tail = @buffer.Buffer::Buffer()
  let objects = if created_values_need_objects(values) ||
    (copied_fill is Some(fill) && created_values_need_objects(fill)) {
    self.created_reference_objects()
  } else {
    []
  }
  let heap = self.append_created_values_heap(
    tail, values, copied_fill, objects, offset_size,
  )
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
    encode_created_object_header(
      {
        let messages : Array[RebuiltObjectMessage] = [
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
        ]
        if copied_fill is Some(fill) {
          messages.push({
            msg_type: msg_fill_value,
            flags: b'\x00',
            data: encode_fill_value(
              fill,
              heap~,
              objects~,
              reference_address_size=offset_size,
              vlen_address_size=offset_size,
            ),
            creation_order: 0UL,
          })
        }
        messages.push({
          msg_type: msg_data_layout,
          flags: b'\x00',
          data: encode_created_contiguous_layout(
            data_address,
            encoded.payload.length(),
            offset_size,
            length_size,
          ),
          creation_order: 0UL,
        })
        messages
      },
    ),
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
/// Adds a compact dataset below `parent_path` and returns the updated file.
///
/// Values are embedded in the new object's data-layout message. Variable-length
/// and reference payloads reuse the normal appended global-heap path. The
/// payload is capped at HDF5 1.14.6's 65,524-byte compact limit; large object
/// headers use continuation chunks. An optional fill value is stored through
/// the normal fill-value message. Compact storage is not extendable.
pub fn Hdf5File::add_compact_dataset(
  self : Hdf5File,
  parent_path : String,
  name : String,
  dimensions : Array[UInt64],
  values : Hdf5WriteValues,
  fill_value? : Hdf5WriteValues,
) -> Bytes raise Hdf5Error {
  validate_editable_link_name(name)
  let (parent_address, header, links) = self.editable_group(parent_path)
  guard editable_link_index(links, name) is None else {
    raise InvalidArgument("link name already exists: " + name)
  }
  let offset_size = self.superblock.offset_size
  validate_write_values(
    dimensions,
    values,
    reference_address_size=Some(offset_size),
  )
  let copied_fill = validated_write_fill(
    values,
    fill_value,
    reference_address_size=Some(offset_size),
  )
  validate_created_fill_layout(copied_fill, offset_size)
  let tail = @buffer.Buffer::Buffer()
  let objects = if created_values_need_objects(values) ||
    (copied_fill is Some(fill) && created_values_need_objects(fill)) {
    self.created_reference_objects()
  } else {
    []
  }
  let heap = self.append_created_values_heap(
    tail, values, copied_fill, objects, offset_size,
  )
  let encoded = encode_write_values(
    values,
    heap~,
    objects~,
    reference_address_size=offset_size,
    vlen_address_size=offset_size,
  )
  guard encoded.payload.length() <= 65524 else {
    raise InvalidArgument("compact dataset values exceed 65524 bytes")
  }
  let messages : Array[RebuiltObjectMessage] = [
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
  ]
  if copied_fill is Some(fill) {
    messages.push({
      msg_type: msg_fill_value,
      flags: b'\x00',
      data: encode_fill_value(
        fill,
        heap~,
        objects~,
        reference_address_size=offset_size,
        vlen_address_size=offset_size,
      ),
      creation_order: 0UL,
    })
  }
  messages.push({
    msg_type: msg_data_layout,
    flags: b'\x00',
    data: encode_compact_layout(encoded.payload),
    creation_order: 0UL,
  })
  let object_address = self.append_created_object_with_messages(tail, messages)
  self.append_created_link(
    parent_address, header, links, object_address, name, tail,
  )
}

///|
/// Adds a named committed datatype below `parent_path` and returns the updated
/// file.
///
/// The datatype message uses the file's address width for legacy and version-4
/// reference members. The new object can receive attributes and hard links
/// through the normal editing APIs, and the complete image remains readable by
/// HDF5 1.14.6.
pub fn Hdf5File::add_committed_datatype(
  self : Hdf5File,
  parent_path : String,
  name : String,
  datatype : Datatype,
) -> Bytes raise Hdf5Error {
  validate_editable_link_name(name)
  let (parent_address, header, links) = self.editable_group(parent_path)
  guard editable_link_index(links, name) is None else {
    raise InvalidArgument("link name already exists: " + name)
  }
  let offset_size = self.superblock.offset_size
  validate_writer_datatype(
    datatype,
    0,
    reference_address_size=Some(offset_size),
  )
  let data = encode_writer_datatype(
    datatype,
    reference_address_size=offset_size,
  )
  guard data.length() <= 65535 else {
    raise UnsupportedFeature("committed datatype message is too large")
  }
  let tail = @buffer.Buffer::Buffer()
  let object_address = self.append_created_object_with_messages(tail, [
    { msg_type: msg_datatype, flags: b'\x00', data, creation_order: 0UL, },
  ])
  self.append_created_link(
    parent_address, header, links, object_address, name, tail,
  )
}

///|
fn created_values_need_objects(values : Hdf5WriteValues) -> Bool {
  match values {
    ObjectReferenceValues(_) => true
    StructuredValues(datatype, _) => datatype_contains_reference(datatype)
    _ => false
  }
}

///|
/// Adds a finite chunked dataset below `parent_path` and returns the updated
/// file.
///
/// The fixed-array chunk index and chunk payloads are appended at aligned EOF
/// offsets. Built-in Shuffle, deflate, Fletcher32 and scale-offset filters are
/// supported. Caller-defined filters are supplied through `custom_filters` and
/// referenced by `Custom(id, parameters)` pipeline entries; IDs must be unique
/// and cannot replace built-in filters. When `maximum_dimensions` contains
/// `None`, the dataset uses an extensible-array index for one unlimited axis or
/// a v2 B-tree for multiple unlimited axes.
pub fn Hdf5File::add_chunked_dataset(
  self : Hdf5File,
  parent_path : String,
  name : String,
  dimensions : Array[UInt64],
  chunk_dimensions : Array[UInt32],
  values : Hdf5WriteValues,
  fill_value? : Hdf5WriteValues,
  maximum_dimensions? : Array[UInt64?],
  filters~ : Array[Hdf5WriteFilter] = [],
  custom_filters~ : Array[&Hdf5CustomFilter] = [],
) -> Bytes raise Hdf5Error {
  validate_editable_link_name(name)
  let (parent_address, header, links) = self.editable_group(parent_path)
  guard editable_link_index(links, name) is None else {
    raise InvalidArgument("link name already exists: " + name)
  }
  let offset_size = self.superblock.offset_size
  let length_size = self.superblock.length_size
  validate_write_values(
    dimensions,
    values,
    reference_address_size=Some(offset_size),
  )
  guard dimensions.length() > 0 &&
    chunk_dimensions.length() == dimensions.length() else {
    raise InvalidArgument("chunk dimensions must match a simple dataspace")
  }
  for axis in 0..<chunk_dimensions.length() {
    guard dimensions[axis] > 0UL &&
      chunk_dimensions[axis] > 0U &&
      chunk_dimensions[axis] <= 0x7FFFFFFFU else {
      raise InvalidArgument("chunk dimension is invalid")
    }
  }
  let (encoded_maximum, unlimited_axes) = match maximum_dimensions {
    Some(maximum) =>
      validate_extensible_dimensions(dimensions, maximum, chunk_dimensions)
    None => (dimensions.copy(), [])
  }
  let grid_dimensions = dimensions.copy()
  for axis in 0..<grid_dimensions.length() {
    if !unlimited_axes.contains(axis) {
      grid_dimensions[axis] = encoded_maximum[axis]
    }
  }
  chunk_grid(grid_dimensions, chunk_dimensions) |> ignore
  validate_writer_chunk_size(
    chunk_dimensions,
    writer_value_element_size(values),
  )
  let registered_filters : Array[&Hdf5CustomFilter] = []
  for filter in custom_filters {
    register_custom_filter(registered_filters, filter)
  }
  validate_write_filters(filters, registered_filters)
  validate_writer_filters_for_values(values, filters)
  let copied_fill = validated_write_fill(
    values,
    fill_value,
    reference_address_size=Some(offset_size),
  )
  validate_created_fill_layout(copied_fill, offset_size)
  let tail = @buffer.Buffer::Buffer()
  let objects = if created_values_need_objects(values) ||
    (copied_fill is Some(fill) && created_values_need_objects(fill)) {
    self.created_reference_objects()
  } else {
    []
  }
  let heap = self.append_created_values_heap(
    tail, values, copied_fill, objects, offset_size,
  )
  let encoded = encode_write_values(
    values,
    heap~,
    objects~,
    reference_address_size=offset_size,
    vlen_address_size=offset_size,
  )
  let fill_payload = match copied_fill {
    Some(fill) =>
      encode_write_values(
        fill,
        heap~,
        objects~,
        reference_address_size=offset_size,
        vlen_address_size=offset_size,
      ).payload
    None => Bytes::make(encoded.element_size, b'\x00')
  }
  let chunk_dataset : WritableDataset = {
    path: "/",
    dimensions: dimensions.copy(),
    maximum_dimensions: encoded_maximum,
    values: copy_write_values(values),
    storage: if unlimited_axes.length() == 0 {
      WritableChunked(chunk_dimensions.copy(), copy_write_filters(filters))
    } else {
      WritableExtensible(
        chunk_dimensions.copy(),
        copy_write_filters(filters),
        unlimited_axes.copy(),
      )
    },
    fill_value: copied_fill,
    chunk_filter_masks: [],
  }
  let storage_start = next_aligned_tail_start(
    self.bytes.length(),
    tail.length(),
    "append chunked dataset storage",
  )
  let storage = if unlimited_axes.length() == 0 {
    encode_fixed_array_storage(
      storage_start,
      chunk_dataset,
      heap,
      objects,
      registered_filters,
      offset_size~,
      length_size~,
      base_address=self.superblock.base_address,
      reference_address_size=offset_size,
      vlen_address_size=offset_size,
    )
  } else if unlimited_axes.length() == 1 {
    encode_extensible_array_storage(
      storage_start,
      chunk_dataset,
      heap,
      objects,
      registered_filters,
      offset_size~,
      length_size~,
      base_address=self.superblock.base_address,
      reference_address_size=offset_size,
      vlen_address_size=offset_size,
    )
  } else {
    encode_btree_v2_chunk_storage(
      storage_start,
      chunk_dataset,
      heap,
      objects,
      registered_filters,
      offset_size~,
      length_size~,
      base_address=self.superblock.base_address,
      reference_address_size=offset_size,
      vlen_address_size=offset_size,
    )
  }
  ignore(
    append_aligned_bytes_to_tail(
      tail,
      self.bytes.length(),
      storage,
      "append chunked dataset storage",
    ),
  )
  let storage_address = file_offset_to_stored_address(
    storage_start,
    self.superblock.base_address,
  )
  guard editable_uint_fits_width(storage_address, offset_size) else {
    raise UnsupportedFeature(
      "chunk index address exceeds the file address width",
    )
  }
  let dataspace = if unlimited_axes.length() > 0 {
    encode_dataspace(dimensions, maximum_dimensions=encoded_maximum)
  } else {
    encode_dataspace(dimensions)
  }
  let messages : Array[RebuiltObjectMessage] = [
    {
      msg_type: msg_dataspace,
      flags: b'\x00',
      data: dataspace,
      creation_order: 0UL,
    },
    {
      msg_type: msg_datatype,
      flags: b'\x00',
      data: encoded.datatype,
      creation_order: 0UL,
    },
  ]
  if copied_fill is Some(fill) {
    messages.push({
      msg_type: msg_fill_value,
      flags: b'\x00',
      data: encode_fill_value(
        fill,
        heap~,
        objects~,
        reference_address_size=offset_size,
        vlen_address_size=offset_size,
      ),
      creation_order: 0UL,
    })
  }
  if filters.length() > 0 {
    messages.push({
      msg_type: msg_filter_pipeline,
      flags: b'\x00',
      data: encode_filter_pipeline(
        filters,
        encoded.element_size,
        writer_chunk_element_count(chunk_dimensions),
        values,
        fill_payload,
        registered_filters,
      ),
      creation_order: 0UL,
    })
  }
  messages.push({
    msg_type: msg_data_layout,
    flags: b'\x00',
    data: encode_chunked_layout(
      storage_address,
      chunk_dimensions,
      encoded.element_size,
      extensible=unlimited_axes.length() == 1,
      btree_v2=unlimited_axes.length() > 1,
      offset_size~,
    ),
    creation_order: 0UL,
  })
  let object_address = self.append_created_object(
    tail,
    encode_created_object_header(messages),
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
/// Adds an unfiltered dataset whose raw bytes live in external files below
/// `parent_path` and returns the updated file.
///
/// Slots are concatenated in caller order to form the row-major element stream
/// and must add up exactly to the dataset byte size. The external-data heap is
/// appended to the HDF5 file with the file's address and length widths.
pub fn Hdf5File::add_external_dataset(
  self : Hdf5File,
  parent_path : String,
  name : String,
  dimensions : Array[UInt64],
  values : Hdf5WriteValues,
  slots : Array[Hdf5ExternalSlot],
  fill_value? : Hdf5WriteValues,
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
  guard !writer_values_contain_vlen(values) else {
    raise UnsupportedFeature(
      "external storage cannot hold variable-length values",
    )
  }
  let offset_size = self.superblock.offset_size
  let length_size = self.superblock.length_size
  validate_write_values(
    dimensions,
    values,
    reference_address_size=Some(offset_size),
  )
  let copied_fill = validated_write_fill(
    values,
    fill_value,
    reference_address_size=Some(offset_size),
  )
  validate_created_fill_layout(copied_fill, offset_size)
  if copied_fill is Some(fill) &&
    (
      !created_dataset_values_supported(fill) ||
      writer_values_contain_vlen(fill)
    ) {
    raise UnsupportedFeature(
      "external storage cannot hold variable-length fills",
    )
  }
  let objects = if created_values_need_objects(values) ||
    (copied_fill is Some(fill) && created_values_need_objects(fill)) {
    self.created_reference_objects()
  } else {
    []
  }
  let encoded = encode_write_values(
    values,
    objects~,
    reference_address_size=offset_size,
    vlen_address_size=offset_size,
  )
  let expected_size = writer_shape_count(dimensions).to_int64() *
    encoded.element_size.to_int64()
  guard expected_size >= 0 else {
    raise InvalidArgument("external dataset size is too large")
  }
  validate_external_slots(slots, expected_size.reinterpret_as_uint64())

  let tail = @buffer.Buffer::Buffer()
  let empty_heap = empty_writable_heap()
  let heap_offset = next_aligned_tail_start(
    self.bytes.length(),
    tail.length(),
    "append external data heap",
  )
  let heap_address = file_offset_to_stored_address(
    heap_offset,
    self.superblock.base_address,
  )
  let external_heap = encode_external_local_heap(
    slots,
    heap_address,
    offset_size~,
    length_size~,
  )
  ignore(
    append_aligned_bytes_to_tail(
      tail,
      self.bytes.length(),
      external_heap,
      "append external data heap",
    ),
  )
  let messages : Array[RebuiltObjectMessage] = [
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
  ]
  if copied_fill is Some(fill) {
    messages.push({
      msg_type: msg_fill_value,
      flags: b'\x00',
      data: encode_fill_value(
        fill,
        heap=empty_heap,
        objects~,
        reference_address_size=offset_size,
        vlen_address_size=offset_size,
      ),
      creation_order: 0UL,
    })
  }
  messages.push({
    msg_type: msg_external_data_files,
    flags: b'\x00',
    data: encode_external_data_message(
      slots,
      heap_address,
      offset_size~,
      length_size~,
    ),
    creation_order: 0UL,
  })
  messages.push({
    msg_type: msg_data_layout,
    flags: b'\x00',
    data: encode_external_data_layout(
      expected_size.reinterpret_as_uint64(),
      offset_size~,
      length_size~,
    ),
    creation_order: 0UL,
  })
  let object_address = self.append_created_object(
    tail,
    encode_created_object_header(messages),
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
