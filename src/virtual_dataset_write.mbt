///|
fn validate_virtual_dataset_dimensions(
  dimensions : Array[UInt64],
) -> Unit raise Hdf5Error {
  guard dimensions.length() <= 32 else {
    raise InvalidArgument("virtual dataset rank exceeds 32")
  }
}

///|
fn virtual_dataset_maximum_dimensions(
  dimensions : Array[UInt64],
  maximum_dimensions : Array[UInt64?]?,
) -> Array[UInt64] raise Hdf5Error {
  match maximum_dimensions {
    None => dimensions.copy()
    Some(values) => {
      guard values.length() == dimensions.length() else {
        raise InvalidArgument(
          "virtual dataset maximum dimensions must match its rank",
        )
      }
      let encoded : Array[UInt64] = []
      for axis in 0..<values.length() {
        match values[axis] {
          None => encoded.push(virtual_unlimited)
          Some(value) => {
            guard value >= dimensions[axis] else {
              raise InvalidArgument(
                "virtual dataset dimensions exceed the maximum dimensions",
              )
            }
            encoded.push(value)
          }
        }
      }
      encoded
    }
  }
}

///|
fn virtual_dataset_needs_extended_dataspace(
  dimensions : Array[UInt64],
  maximum_dimensions : Array[UInt64],
) -> Bool {
  if dimensions.length() != maximum_dimensions.length() {
    return true
  }
  for axis in 0..<dimensions.length() {
    if dimensions[axis] != maximum_dimensions[axis] {
      return true
    }
  }
  false
}

///|
fn virtual_dataset_values(
  datatype : Datatype,
  fill_value : DatasetValue?,
  reference_address_size : Int32,
) -> (Hdf5WriteValues, Hdf5WriteValues?) raise Hdf5Error {
  validate_writer_datatype(
    datatype,
    0,
    reference_address_size=Some(reference_address_size),
  )
  let empty = StructuredValues(datatype, [])
  match fill_value {
    Some(value) => {
      validate_structured_value(value, datatype, 0)
      (empty, Some(StructuredValues(datatype, [value])))
    }
    None => (empty, None)
  }
}

///|
/// Adds a virtual dataset backed by one or more source mappings.
///
/// `source_file` uses `.` for the current file and otherwise names a file that
/// a resolver or caller will provide when reading. Source and virtual selections
/// may be all, empty or hyperslab selections. HDF5 1.14.6 does not support point
/// or block virtual mappings, so those selections are rejected. When both
/// selections have countable elements, their counts must match. A fill value is
/// one value of the dataset datatype and is used for unmapped virtual elements.
/// A `%b` in either source name expands to the block number of an unlimited
/// virtual hyperslab; `%%` expands to a literal percent sign. A `None` entry in
/// `maximum_dimensions` marks the corresponding axis as unlimited.
pub fn Hdf5Writer::add_virtual_dataset(
  self : Hdf5Writer,
  path : String,
  dimensions : Array[UInt64],
  datatype : Datatype,
  mappings : Array[Hdf5VirtualMapping],
  fill_value? : DatasetValue,
  maximum_dimensions? : Array[UInt64?],
) -> Unit raise Hdf5Error {
  let normalized = canonical_writer_path(path)
  guard normalized != "/" else {
    raise InvalidArgument("a dataset cannot replace the root group")
  }
  guard !self.has_entry(normalized) else {
    raise InvalidArgument("writer object path already exists")
  }
  validate_virtual_dataset_dimensions(dimensions)
  let maximum = virtual_dataset_maximum_dimensions(
    dimensions, maximum_dimensions,
  )
  let (values, copied_fill) = virtual_dataset_values(datatype, fill_value, 8)
  for mapping in mappings {
    validate_virtual_mapping(dimensions, maximum, mapping)
  }
  self.ensure_groups(writer_parent_path(normalized))
  self.record_link(normalized)
  self.datasets.push({
    path: normalized,
    dimensions: dimensions.copy(),
    maximum_dimensions: maximum,
    values,
    chunk_source: None,
    value_source: None,
    block_dimensions: [],
    storage: WritableVirtual(copy_virtual_mappings(mappings)),
    fill_value: copied_fill,
    chunk_filter_masks: [],
  })
}

///|
/// Adds a virtual dataset below `parent_path` and returns the updated file.
///
/// The mapping block and any fill-value heap payloads are appended at aligned
/// EOF offsets. The dataset object has no raw storage of its own. Source files
/// are not opened or validated during creation; readers resolve them through
/// their normal same-file or registered-source paths.
pub fn Hdf5File::add_virtual_dataset(
  self : Hdf5File,
  parent_path : String,
  name : String,
  dimensions : Array[UInt64],
  datatype : Datatype,
  mappings : Array[Hdf5VirtualMapping],
  fill_value? : DatasetValue,
  maximum_dimensions? : Array[UInt64?],
) -> Bytes raise Hdf5Error {
  validate_editable_link_name(name)
  let (parent_address, header, links) = self.editable_group(parent_path)
  guard editable_link_index(links, name) is None else {
    raise InvalidArgument("link name already exists: " + name)
  }
  validate_virtual_dataset_dimensions(dimensions)
  let maximum = virtual_dataset_maximum_dimensions(
    dimensions, maximum_dimensions,
  )
  let (values, copied_fill) = virtual_dataset_values(
    datatype,
    fill_value,
    self.superblock.offset_size,
  )
  for mapping in mappings {
    validate_virtual_mapping(dimensions, maximum, mapping)
  }
  validate_created_fill_layout(copied_fill, self.superblock.offset_size)

  let offset_size = self.superblock.offset_size
  let length_size = self.superblock.length_size
  let tail = @buffer.Buffer::Buffer()
  let objects = if copied_fill is Some(fill) &&
    created_values_need_objects(fill) {
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
  let mapping_block = encode_virtual_mapping_block(
    dimensions, maximum, mappings, length_size,
  )
  let mapping_heap = encode_heap_objects([mapping_block], length_size~)
  let mapping_start = append_aligned_bytes_to_tail(
    tail,
    self.bytes.length(),
    mapping_heap,
    "append virtual mapping heap",
  )
  let mapping_address = file_offset_to_stored_address(
    mapping_start,
    self.superblock.base_address,
  )
  guard editable_uint_fits_width(mapping_address, offset_size) else {
    raise UnsupportedFeature(
      "virtual mapping heap address exceeds the file address width",
    )
  }

  let messages : Array[RebuiltObjectMessage] = [
    {
      msg_type: msg_dataspace,
      flags: b'\x00',
      data: if virtual_dataset_needs_extended_dataspace(dimensions, maximum) {
        encode_dataspace(dimensions, maximum_dimensions=maximum)
      } else {
        encode_dataspace(dimensions)
      },
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
    data: encode_virtual_layout(mapping_address, 1, offset_size~),
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
