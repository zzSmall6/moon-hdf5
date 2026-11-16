///|
fn encode_resized_dataspace(
  dimensions : Array[UInt64],
  maximum_dimensions : Array[UInt64],
  length_size : Int32,
) -> Bytes raise Hdf5Error {
  guard length_size == 2 || length_size == 4 || length_size == 8 else {
    raise UnsupportedAddressSize(length_size)
  }
  guard dimensions.length() > 0 &&
    dimensions.length() == maximum_dimensions.length() else {
    raise InvalidArgument("resized dataspace rank is invalid")
  }
  let buffer = @buffer.Buffer::Buffer(
    size_hint=4 + dimensions.length() * length_size * 2,
  )
  buffer.write_byte(b'\x02')
  buffer.write_byte(dimensions.length().to_byte())
  buffer.write_byte(b'\x01')
  buffer.write_byte(b'\x01')
  for dimension in dimensions {
    write_uint_width(buffer, dimension, length_size)
  }
  for dimension in maximum_dimensions {
    write_uint_width(buffer, dimension, length_size)
  }
  buffer.to_bytes()
}

///|
/// Changes the shape of an existing resizable chunked dataset and returns a
/// new byte image.
///
/// The object header address, links, attributes, datatype, filters and chunk
/// index are preserved. The dataspace is rewritten through the normal
/// append-only object-header continuation path. Existing chunks keep their
/// coordinates and values; newly exposed elements are read as the dataset fill
/// value. Shrinking only changes the visible extent, so hidden chunks remain
/// allocated until a later storage rewrite or free-space reclamation.
///
/// Only bytes-backed files and chunked datasets with an extensible-array,
/// version-2 B-tree or legacy version-1 B-tree index are supported. Each new
/// dimension must not exceed the dataspace maximum. Fixed-array, compact,
/// contiguous, external, virtual and scalar or null dataspaces are rejected.
pub fn Hdf5File::resize_dataset(
  self : Hdf5File,
  path : String,
  dimensions : Array[UInt64],
) -> Bytes raise Hdf5Error {
  guard self.source is None else {
    raise InvalidArgument("dataset resize requires a bytes-backed Hdf5File")
  }
  let node = self.resolve(path)
  guard node.kind == Dataset else {
    raise InvalidField("object is not a dataset: " + path)
  }
  let header = self.object_header_at(node.address)
  let metadata = read_dataset_metadata(
    header,
    self.superblock.offset_size,
    self.superblock.length_size,
  )
  guard metadata.dataspace.kind == Simple else {
    raise UnsupportedFeature("dataset resize requires a simple dataspace")
  }
  guard metadata.layout.kind() == Chunked else {
    raise UnsupportedFeature("dataset resize requires chunked storage")
  }
  guard dimensions.length() == metadata.dataspace.dimensions.length() else {
    raise InvalidArgument("resized dataset rank does not match")
  }
  let resizable = match metadata.layout.chunk_index {
    ExtensibleArray(_) | BTreeV2(_) | BTreeV1 => true
    Single(_, _) | Implicit | FixedArray(_) => false
  }
  guard resizable else {
    raise UnsupportedFeature("dataset chunk index does not support resizing")
  }
  for axis in 0..<dimensions.length() {
    let maximum = metadata.dataspace.maximum_dimensions[axis]
    guard dimensions[axis] <= maximum || maximum == 0xFFFFFFFFFFFFFFFFUL else {
      raise InvalidArgument("resized dataset exceeds its maximum dimensions")
    }
  }
  if dimensions == metadata.dataspace.dimensions {
    return self.bytes
  }
  let header_offset = address_to_offset(
    node.address,
    self.superblock.base_address,
  )
  let plan = object_header_rewrite_plan(
    self.bytes,
    header_offset,
    self.superblock.offset_size,
    self.superblock.length_size,
  )
  let message : RebuiltObjectMessage = {
    msg_type: msg_dataspace,
    flags: b'\x00',
    data: encode_resized_dataspace(
      dimensions,
      metadata.dataspace.maximum_dimensions,
      self.superblock.length_size,
    ),
    creation_order: 0UL,
  }
  let tail = @buffer.Buffer::Buffer()
  self.write_rebuilt_object_messages(
    node.address,
    header,
    [msg_dataspace],
    [message],
    plan.track_order,
    tail,
  )
}

///|
/// Appends a block at the trailing corner of an existing resizable dataset and
/// returns a new byte image.
///
/// An axis is growable when its maximum extent exceeds the current extent or
/// is unlimited. Growable axes advance by the appended block extent, while
/// fixed axes require the block extent to equal the current extent. The new
/// block is placed at the old shape's trailing corner; elements not covered by
/// the block use the dataset fill value.
///
/// Existing chunks are preserved and only the new chunk grid is written. The
/// appended values may use any datatype supported by value patching, including
/// variable-length, reference and nested structured values. External, virtual,
/// compact, contiguous, scalar and null datasets are rejected.
pub fn Hdf5File::append_dataset(
  self : Hdf5File,
  path : String,
  dimensions : Array[UInt64],
  values : Hdf5WriteValues,
) -> Bytes raise Hdf5Error {
  guard self.source is None else {
    raise InvalidArgument("dataset append requires a bytes-backed Hdf5File")
  }
  let node = self.resolve(path)
  guard node.kind == Dataset else {
    raise InvalidField("object is not a dataset: " + path)
  }
  let header = self.object_header_at(node.address)
  let metadata = read_dataset_metadata(
    header,
    self.superblock.offset_size,
    self.superblock.length_size,
  )
  guard metadata.dataspace.kind == Simple else {
    raise UnsupportedFeature("dataset append requires a simple dataspace")
  }
  guard metadata.layout.kind() == Chunked else {
    raise UnsupportedFeature("dataset append requires chunked storage")
  }
  let resizable = match metadata.layout.chunk_index {
    ExtensibleArray(_) | BTreeV2(_) | BTreeV1 => true
    Single(_, _) | Implicit | FixedArray(_) => false
  }
  guard resizable else {
    raise UnsupportedFeature("dataset chunk index does not support resizing")
  }
  let rank = metadata.dataspace.dimensions.length()
  guard dimensions.length() == rank else {
    raise InvalidArgument("appended dataset rank does not match")
  }
  guard patch_values_match_datatype(values, metadata.datatype) else {
    raise InvalidArgument("appended values must match the dataset type")
  }
  validate_patch_write_values(dimensions, values)
  let new_dimensions = metadata.dataspace.dimensions.copy()
  let offsets = Array::make(rank, 0UL)
  let mut grows = false
  for axis in 0..<rank {
    let old_dimension = metadata.dataspace.dimensions[axis]
    let maximum = metadata.dataspace.maximum_dimensions[axis]
    let growable = maximum > old_dimension || maximum == 0xFFFFFFFFFFFFFFFFUL
    if !growable {
      guard dimensions[axis] == old_dimension else {
        raise InvalidArgument(
          "appended block must match fixed dataset dimensions",
        )
      }
    } else {
      guard old_dimension <= 0x7FFFFFFFUL - dimensions[axis] else {
        raise InvalidArgument("appended dataset dimension is too large")
      }
      let next = old_dimension + dimensions[axis]
      guard next <= maximum || maximum == 0xFFFFFFFFFFFFFFFFUL else {
        raise InvalidArgument("appended dataset exceeds its maximum dimensions")
      }
      offsets[axis] = old_dimension
      new_dimensions[axis] = next
      if next > old_dimension {
        grows = true
      }
    }
  }
  if !grows {
    return self.bytes
  }
  let resized_bytes = self.resize_dataset(path, new_dimensions)
  let resized_file = Hdf5File::parse(resized_bytes)
  let resized_node = resized_file.resolve(path)
  let resized_header = resized_file.object_header_at(resized_node.address)
  let resized_metadata = read_dataset_metadata(
    resized_header,
    resized_file.superblock.offset_size,
    resized_file.superblock.length_size,
  )
  let filters = match resized_header.first(msg_filter_pipeline) {
    Some(message) =>
      parse_filters(message, custom_filters=resized_file.custom_filters)
    None => []
  }
  let fill = dataset_fill_value(resized_header, resized_metadata.datatype)
  let full_payload = read_chunked_with_filters(
    resized_file.bytes,
    resized_metadata.layout,
    resized_metadata.dataspace,
    resized_metadata.datatype,
    resized_file.superblock.offset_size,
    resized_file.superblock.base_address,
    fill,
    filters,
    resized_file.superblock.length_size,
  ).to_fixedarray()
  let prefix_tail = @buffer.Buffer::Buffer()
  let appended_payload = resized_file.encode_dataset_patch_payload(
    values,
    resized_metadata.datatype,
    resized_header,
    true,
    prefix_tail,
  )
  let element_size = resized_metadata.datatype.size
  let block_count = writer_shape_count(dimensions)
  guard appended_payload.length() == block_count * element_size else {
    raise InvalidArgument("appended values do not match the block shape")
  }
  for index in 0..<block_count {
    let destination = match
      writer_linear_destination(index, dimensions, new_dimensions, offsets) {
      Some(destination) => destination
      None =>
        raise InvalidArgument("appended value is outside the dataset shape")
    }
    full_payload.blit_from_bytes(
      destination * element_size,
      appended_payload,
      index * element_size,
      element_size,
    )
  }
  resized_file.patch_chunked_dataset_payload(
    path,
    full_payload.unsafe_reinterpret_as_bytes(),
    values,
    prefix_tail.to_bytes(),
  )
}
