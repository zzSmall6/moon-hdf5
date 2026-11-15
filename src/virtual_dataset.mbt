///|
/// One serialized mapping in a virtual dataset layout.
priv struct VirtualDatasetMapping {
  source_file : String
  source_dataset : String
  source_selection : Hdf5RegionSelection
  virtual_selection : Hdf5RegionSelection
}

///|
priv struct VirtualMappingLocation {
  virtual_index : UInt64
  source_ordinal : Int32
}

///|
fn virtual_selection_shape(
  dataspace : Dataspace,
  selection : Hdf5RegionSelection,
) -> Array[UInt64] raise Hdf5Error {
  let rank = dataspace.dimensions.length()
  match selection {
    RegionEmpty => Array::make(rank, 0UL)
    RegionAll => dataspace.dimensions.copy()
    RegionHyperslab(_, _, count, block) => {
      let dimensions : Array[UInt64] = []
      for axis in 0..<rank {
        guard block[axis] == 0UL ||
          count[axis] <= 0xFFFFFFFFFFFFFFFFUL / block[axis] else {
          raise InvalidField("virtual region selection shape overflow")
        }
        dimensions.push(count[axis] * block[axis])
      }
      dimensions
    }
    _ => Array::make(rank, 0UL)
  }
}

///|
/// Returns the number of elements in a selection without expanding it.
fn virtual_selection_count(
  dataspace : Dataspace,
  selection : Hdf5RegionSelection,
) -> Int64 raise Hdf5Error {
  match selection {
    RegionEmpty => 0L
    RegionAll => dataspace.element_count()
    RegionHyperslab(_, _, _, _) =>
      selection_element_count(virtual_selection_shape(dataspace, selection))
    _ =>
      raise InvalidField(
        "virtual mapping selection is not directly addressable",
      )
  }
}

///|
/// Reports whether a selection can be addressed in generation order without
/// expanding it. All and non-overlapping hyperslab selections meet this
/// condition; point and irregular block selections retain the materialized
/// fallback path.
fn virtual_selection_is_direct(selection : Hdf5RegionSelection) -> Bool {
  match selection {
    RegionEmpty | RegionAll => true
    RegionHyperslab(_, stride, count, block) => {
      for axis in 0..<stride.length() {
        if count[axis] == 0UL || block[axis] == 0UL {
          continue
        }
        if stride[axis] < block[axis] {
          return false
        }
      }
      true
    }
    _ => false
  }
}

///|
fn virtual_selection_index_at(
  dataspace : Dataspace,
  selection : Hdf5RegionSelection,
  ordinal : Int64,
) -> UInt64 raise Hdf5Error {
  let dimensions = virtual_selection_shape(dataspace, selection)
  let total = virtual_selection_count(dataspace, selection)
  guard ordinal >= 0L && ordinal < total else {
    raise InvalidField("virtual mapping ordinal is outside the selection")
  }
  match selection {
    RegionEmpty =>
      raise InvalidField("empty virtual mapping selection has no elements")
    RegionAll => ordinal.reinterpret_as_uint64()
    RegionHyperslab(start, stride, _count, block) => {
      let rank = dimensions.length()
      let positions = Array::make(rank, 0UL)
      let mut remaining = ordinal
      let mut axis = rank - 1
      while axis >= 0 {
        let width = dimensions[axis].reinterpret_as_int64()
        positions[axis] = (remaining % width).reinterpret_as_uint64()
        remaining = remaining / width
        axis = axis - 1
      }
      let coordinate = Array::make(rank, 0UL)
      for dimension in 0..<rank {
        coordinate[dimension] = start[dimension] +
          positions[dimension] / block[dimension] * stride[dimension] +
          positions[dimension] % block[dimension]
      }
      coordinate_linear_index(coordinate, dataspace.dimensions)
    }
    _ =>
      raise InvalidField(
        "virtual mapping selection is not directly addressable",
      )
  }
}

///|
fn virtual_selection_ordinal_of(
  dataspace : Dataspace,
  selection : Hdf5RegionSelection,
  target : UInt64,
) -> Int64? raise Hdf5Error {
  match selection {
    RegionEmpty => None
    RegionAll =>
      if target < dataspace.element_count().reinterpret_as_uint64() {
        Some(target.reinterpret_as_int64())
      } else {
        None
      }
    RegionHyperslab(start, stride, count, block) => {
      let dimensions = virtual_selection_shape(dataspace, selection)
      let rank = dimensions.length()
      let coordinate = Array::make(rank, 0UL)
      let mut remaining = target
      let mut axis = rank - 1
      while axis >= 0 {
        let width = dataspace.dimensions[axis]
        coordinate[axis] = remaining % width
        remaining = remaining / width
        axis = axis - 1
      }
      let positions = Array::make(rank, 0UL)
      for axis in 0..<rank {
        if count[axis] == 0UL ||
          block[axis] == 0UL ||
          coordinate[axis] < start[axis] {
          return None
        }
        let delta = coordinate[axis] - start[axis]
        let stride_index = delta / stride[axis]
        let block_position = delta % stride[axis]
        if stride_index >= count[axis] || block_position >= block[axis] {
          return None
        }
        positions[axis] = stride_index * block[axis] + block_position
      }
      let mut ordinal = 0L
      for axis in 0..<rank {
        guard ordinal <=
          0x7FFFFFFFFFFFFFFFL / dimensions[axis].reinterpret_as_int64() else {
          raise InvalidField("virtual mapping ordinal overflow")
        }
        ordinal = ordinal * dimensions[axis].reinterpret_as_int64() +
          positions[axis].reinterpret_as_int64()
      }
      Some(ordinal)
    }
    _ =>
      raise InvalidField(
        "virtual mapping selection is not directly addressable",
      )
  }
}

///|
fn read_virtual_mappings(
  file : Hdf5File,
  layout : DataLayout,
) -> Array[VirtualDatasetMapping] raise Hdf5Error {
  let undefined = heap_undefined_address(file.superblock.offset_size)
  if layout.virtual_heap_address == undefined {
    return []
  }
  guard layout.virtual_heap_index > 0 else {
    raise InvalidField("invalid virtual mapping heap index")
  }
  let payload = file.read_reference_heap_object(
    layout.virtual_heap_address,
    layout.virtual_heap_index,
  )
  let length_size = file.superblock.length_size
  guard payload.length().to_int64() >= 1L + length_size.to_int64() + 4L else {
    raise InvalidField("virtual mapping block is too short")
  }
  let checksum_offset = payload.length() - 4
  let stored_checksum = stored_metadata_checksum(payload, checksum_offset)
  let computed_checksum = metadata_checksum(payload, 0, checksum_offset)
  guard stored_checksum == computed_checksum else {
    raise InvalidField("virtual mapping block checksum mismatch")
  }

  let reader = Reader::new(payload)
  guard reader.read_u8().to_int() == 0 else {
    raise UnsupportedFeature("unsupported virtual mapping block version")
  }
  let entry_count = reader.read_uint_le(length_size)
  guard entry_count <= 0x7FFFFFFFUL else {
    raise InvalidField("virtual mapping block contains too many entries")
  }
  guard entry_count == 0UL ||
    entry_count <= (reader.remaining() - 4).to_uint64() / 18UL else {
    raise InvalidField("virtual mapping block entry count is too large")
  }

  let mappings : Array[VirtualDatasetMapping] = []
  for _ in 0..<entry_count.to_int() {
    let source_file = read_link_c_string(
      reader, "virtual dataset source file name",
    )
    let source_dataset = read_link_c_string(
      reader, "virtual dataset source dataset name",
    )
    let source_selection = parse_region_selection_body(reader)
    let virtual_selection = parse_region_selection_body(reader)
    mappings.push({
      source_file,
      source_dataset,
      source_selection,
      virtual_selection,
    })
  }
  guard reader.remaining() == 4 else {
    raise InvalidField("virtual mapping block has trailing bytes")
  }
  guard reader.read_u32_le() == stored_checksum else {
    raise InvalidField("virtual mapping block checksum mismatch")
  }
  mappings
}

///|
fn Hdf5File::registered_source(self : Hdf5File, name : String) -> &Hdf5Source? {
  for item in self.external_sources {
    if item.name == name {
      return Some(item.source)
    }
  }
  None
}

///|
fn Hdf5File::source_from_registered_file(
  self : Hdf5File,
  target_file : String,
) -> (String, Hdf5File) raise Hdf5Error {
  if target_file == "" || target_file == "." {
    return ("", self)
  }
  let normalized = resolver_file_name(target_file)
  let matched = match self.registered_source(normalized) {
    Some(source) => Some((normalized, source))
    None => {
      let basename = resolver_basename(normalized)
      if basename != normalized {
        match self.registered_source(basename) {
          Some(source) => Some((basename, source))
          None => None
        }
      } else {
        None
      }
    }
  }
  let (name, source) = match matched {
    Some(value) => value
    None =>
      raise InvalidField(
        "virtual dataset source file is not registered: " + target_file,
      )
  }
  let mut file = Hdf5File::parse_source(source)
  for filter in self.custom_filters {
    file = file.with_custom_filter(filter)
  }
  for external in self.external_sources {
    if external.name != name {
      file = file.with_external_source(external.name, external.source)
    }
  }
  (name, file)
}

///|
fn Hdf5File::virtual_source_file(
  self : Hdf5File,
  target_file : String,
  resolver : Hdf5Resolver?,
  referring_file : String,
) -> (String, Hdf5File) raise Hdf5Error {
  match resolver {
    Some(value) =>
      if target_file == "" || target_file == "." {
        (referring_file, value.open(referring_file))
      } else {
        let name = value.external_source_name(referring_file, target_file)
        (name, value.open(name))
      }
    None => self.source_from_registered_file(target_file)
  }
}

///|
fn virtual_datatype_compatible(source : Datatype, target : Datatype) -> Bool {
  source.size == target.size &&
  source.class == target.class &&
  source.signed == target.signed &&
  source.little_endian == target.little_endian &&
  source.vlen_is_string == target.vlen_is_string &&
  source.bit_offset == target.bit_offset &&
  source.bit_precision == target.bit_precision &&
  source.float_sign_position == target.float_sign_position &&
  source.float_exponent_position == target.float_exponent_position &&
  source.float_exponent_size == target.float_exponent_size &&
  source.float_mantissa_position == target.float_mantissa_position &&
  source.float_mantissa_size == target.float_mantissa_size &&
  source.float_exponent_bias == target.float_exponent_bias &&
  source.float_normalization == target.float_normalization &&
  source.string_padding == target.string_padding &&
  source.string_charset == target.string_charset &&
  source.opaque_tag == target.opaque_tag &&
  source.reference_kind == target.reference_kind &&
  source.array_dimensions == target.array_dimensions &&
  virtual_enum_members_equal(source.enum_members, target.enum_members) &&
  virtual_compound_members_equal(
    source.compound_members,
    target.compound_members,
  ) &&
  virtual_base_types_equal(source.base_types, target.base_types)
}

///|
fn virtual_enum_members_equal(
  source : Array[EnumMember],
  target : Array[EnumMember],
) -> Bool {
  if source.length() != target.length() {
    return false
  }
  for index in 0..<source.length() {
    if source[index].name != target[index].name ||
      source[index].value != target[index].value {
      return false
    }
  }
  true
}

///|
fn virtual_compound_members_equal(
  source : Array[CompoundMember],
  target : Array[CompoundMember],
) -> Bool {
  if source.length() != target.length() {
    return false
  }
  for index in 0..<source.length() {
    if source[index].name != target[index].name ||
      source[index].offset != target[index].offset ||
      !virtual_datatype_compatible(
        source[index].datatype,
        target[index].datatype,
      ) {
      return false
    }
  }
  true
}

///|
fn virtual_base_types_equal(
  source : Array[Datatype],
  target : Array[Datatype],
) -> Bool {
  if source.length() != target.length() {
    return false
  }
  for index in 0..<source.length() {
    if !virtual_datatype_compatible(source[index], target[index]) {
      return false
    }
  }
  true
}

///|
fn Hdf5File::read_dataset_indices_payload(
  self : Hdf5File,
  path : String,
  header : ObjectHeader,
  metadata : DatasetMetadata,
  indices : Array[UInt64],
  resolver : Hdf5Resolver?,
  referring_file : String,
  depth : Int32,
) -> Bytes raise Hdf5Error {
  guard depth < 32 else {
    raise InvalidField("virtual dataset nesting is too deep")
  }
  if metadata.layout.layout_class == 3 {
    return self.read_virtual_dataset_indices_payload(
      path, header, metadata, indices, resolver, referring_file, depth,
    )
  }
  if header.first(msg_external_data_files) is Some(_) {
    let value = match resolver {
      Some(value) => value
      None =>
        raise UnsupportedFeature(
          "external dataset storage requires Hdf5Resolver",
        )
    }
    let slots = value.external_data_slots(referring_file, self, header)
    return external_selection_payload(
      slots,
      metadata.datatype.size,
      indices,
      metadata.dataspace.element_count(),
    )
  }

  match metadata.layout.layout_class {
    0 =>
      selected_payload(
        b"",
        header,
        metadata,
        indices,
        self.superblock.offset_size,
        self.superblock.length_size,
        self.superblock.base_address,
        custom_filters=self.custom_filters,
      )
    1 =>
      match self.source {
        Some(source) =>
          source_contiguous_payload(
            source,
            header,
            metadata,
            indices,
            self.superblock.offset_size,
            self.superblock.base_address,
          )
        None =>
          selected_payload(
            self.bytes,
            header,
            metadata,
            indices,
            self.superblock.offset_size,
            self.superblock.length_size,
            self.superblock.base_address,
            custom_filters=self.custom_filters,
          )
      }
    2 => {
      let filters = match header.first(msg_filter_pipeline) {
        Some(message) =>
          parse_filters(message, custom_filters=self.custom_filters)
        None => []
      }
      match self.source {
        Some(source) =>
          read_source_chunked_with_filters(
            source,
            metadata.layout,
            metadata.dataspace,
            metadata.datatype,
            self.superblock.offset_size,
            self.superblock.length_size,
            self.superblock.base_address,
            dataset_fill_value(header, metadata.datatype),
            filters,
            indices,
          )
        None =>
          read_chunked_with_filters(
            self.bytes,
            metadata.layout,
            metadata.dataspace,
            metadata.datatype,
            self.superblock.offset_size,
            self.superblock.base_address,
            dataset_fill_value(header, metadata.datatype),
            filters,
            self.superblock.length_size,
            selection=Some(indices),
          )
      }
    }
    _ => raise InvalidField("unsupported dataset storage layout")
  }
}

///|
fn virtual_locations(
  virtual_indices : Array[UInt64],
) -> Array[VirtualMappingLocation] {
  let locations : Array[VirtualMappingLocation] = []
  for ordinal in 0..<virtual_indices.length() {
    locations.push({
      virtual_index: virtual_indices[ordinal],
      source_ordinal: ordinal,
    })
  }
  locations.sort_by((left, right) => {
    if left.virtual_index < right.virtual_index {
      -1
    } else if left.virtual_index > right.virtual_index {
      1
    } else {
      0
    }
  })
  locations
}

///|
fn find_virtual_location(
  locations : Array[VirtualMappingLocation],
  target : UInt64,
) -> Int32? {
  let mut low = 0
  let mut high = locations.length()
  while low < high {
    let middle = (low + high) / 2
    if locations[middle].virtual_index < target {
      low = middle + 1
    } else {
      high = middle
    }
  }
  if low < locations.length() && locations[low].virtual_index == target {
    Some(locations[low].source_ordinal)
  } else {
    None
  }
}

///|
fn Hdf5File::read_virtual_dataset_indices_payload(
  self : Hdf5File,
  _path : String,
  header : ObjectHeader,
  metadata : DatasetMetadata,
  indices : Array[UInt64],
  resolver : Hdf5Resolver?,
  referring_file : String,
  depth : Int32,
) -> Bytes raise Hdf5Error {
  guard metadata.layout.layout_class == 3 else {
    raise InvalidField("expected virtual dataset storage")
  }
  let datatype = metadata.datatype
  guard datatype.size > 0 else { raise InvalidField("invalid datatype size") }
  let output = filled_payload(
    indices.length().to_int64(),
    datatype.size,
    dataset_fill_value(header, datatype),
  )
  if indices.length() == 0 {
    return output.unsafe_reinterpret_as_bytes()
  }

  for mapping in read_virtual_mappings(self, metadata.layout) {
    let (source_name, source_file) = self.virtual_source_file(
      mapping.source_file,
      resolver,
      referring_file,
    )
    let source_path = resolver_canonical_object_path(
      "/",
      mapping.source_dataset,
    )
    let source_node = source_file.resolve(source_path)
    guard source_node.kind == Dataset else {
      raise InvalidField(
        "virtual dataset source is not a dataset: " + mapping.source_dataset,
      )
    }
    let source_header = source_file.object_header_at(source_node.address)
    let source_metadata = read_dataset_metadata(
      source_header,
      source_file.superblock.offset_size,
      source_file.superblock.length_size,
    )
    guard virtual_datatype_compatible(source_metadata.datatype, datatype) else {
      raise UnsupportedFeature(
        "virtual dataset source datatype conversion is not supported",
      )
    }
    validate_region_selection(
      source_metadata.dataspace,
      mapping.source_selection,
    )
    validate_region_selection(metadata.dataspace, mapping.virtual_selection)
    if virtual_selection_is_direct(mapping.source_selection) &&
      virtual_selection_is_direct(mapping.virtual_selection) {
      let source_count = virtual_selection_count(
        source_metadata.dataspace,
        mapping.source_selection,
      )
      let virtual_count = virtual_selection_count(
        metadata.dataspace,
        mapping.virtual_selection,
      )
      guard source_count == virtual_count else {
        raise InvalidField(
          "virtual and source selections have different element counts",
        )
      }
      let destinations : Array[Int32] = []
      let source_indices : Array[UInt64] = []
      for destination in 0..<indices.length() {
        match
          virtual_selection_ordinal_of(
            metadata.dataspace,
            mapping.virtual_selection,
            indices[destination],
          ) {
          Some(source_ordinal) => {
            guard source_ordinal >= 0L && source_ordinal < source_count else {
              raise InvalidField(
                "virtual mapping ordinal is outside the source selection",
              )
            }
            source_indices.push(
              virtual_selection_index_at(
                source_metadata.dataspace,
                mapping.source_selection,
                source_ordinal,
              ),
            )
            destinations.push(destination)
          }
          None => ()
        }
      }
      if destinations.length() == 0 {
        continue
      }
      let source_payload = source_file.read_dataset_indices_payload(
        source_path,
        source_header,
        source_metadata,
        source_indices,
        resolver,
        source_name,
        depth + 1,
      )
      let expected = source_indices.length() * datatype.size
      guard source_payload.length() == expected else {
        raise InvalidField("virtual dataset source payload has an invalid size")
      }
      for match_index in 0..<destinations.length() {
        output.blit_from_bytes(
          destinations[match_index] * datatype.size,
          source_payload,
          match_index * datatype.size,
          datatype.size,
        )
      }
      continue
    }
    let (_, source_indices) = region_selection_indices(
      source_metadata.dataspace,
      mapping.source_selection,
    )
    let (_, virtual_indices) = region_selection_indices(
      metadata.dataspace,
      mapping.virtual_selection,
    )
    guard source_indices.length() == virtual_indices.length() else {
      raise InvalidField(
        "virtual and source selections have different element counts",
      )
    }
    if source_indices.length() == 0 {
      continue
    }
    let source_payload = source_file.read_dataset_indices_payload(
      source_path,
      source_header,
      source_metadata,
      source_indices,
      resolver,
      source_name,
      depth + 1,
    )
    let expected = source_indices.length() * datatype.size
    guard source_payload.length() == expected else {
      raise InvalidField("virtual dataset source payload has an invalid size")
    }
    let locations = virtual_locations(virtual_indices)
    for destination in 0..<indices.length() {
      match find_virtual_location(locations, indices[destination]) {
        Some(source_ordinal) =>
          output.blit_from_bytes(
            destination * datatype.size,
            source_payload,
            source_ordinal * datatype.size,
            datatype.size,
          )
        None => ()
      }
    }
  }
  output.unsafe_reinterpret_as_bytes()
}

///|
fn Hdf5File::read_virtual_dataset_indices(
  self : Hdf5File,
  header : ObjectHeader,
  metadata : DatasetMetadata,
  dimensions : Array[UInt64],
  indices : Array[UInt64],
  resolver : Hdf5Resolver?,
  referring_file : String,
) -> DatasetSelectionResult raise Hdf5Error {
  let payload = self.read_virtual_dataset_indices_payload(
    "", header, metadata, indices, resolver, referring_file, 0,
  )
  let values = decode_values(
    self.bytes,
    payload,
    indices.length().to_int64(),
    metadata.datatype,
    self.superblock.offset_size,
    self.superblock.length_size,
    self.superblock.base_address,
    source=self.source,
  )
  { dimensions, datatype: metadata.datatype, values, }
}

///|
fn full_dataset_selection(dataspace : Dataspace) -> DatasetSelection {
  let rank = dataspace.dimensions.length()
  Hyperslab(
    Array::make(rank, 0UL),
    dataspace.dimensions.copy(),
    Array::make(rank, 1UL),
  )
}

///|
fn Hdf5File::read_dataset_selection_with_context(
  self : Hdf5File,
  path : String,
  selection : DatasetSelection,
  resolver : Hdf5Resolver?,
  referring_file : String,
) -> DatasetSelectionResult raise Hdf5Error {
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
  if metadata.layout.layout_class == 3 {
    let (dimensions, indices) = selection_indices(metadata.dataspace, selection)
    return self.read_virtual_dataset_indices(
      header, metadata, dimensions, indices, resolver, referring_file,
    )
  }
  if header.first(msg_external_data_files) is Some(_) {
    match resolver {
      Some(value) =>
        return value.read_external_dataset_selection(
          referring_file, self, header, metadata, selection,
        )
      None => ()
    }
  }
  match self.source {
    Some(source) =>
      read_source_dataset_selection(
        source,
        header,
        metadata,
        selection,
        self.superblock.offset_size,
        self.superblock.length_size,
        self.superblock.base_address,
        custom_filters=self.custom_filters,
      )
    None =>
      read_dataset_selection(
        self.bytes,
        header,
        metadata,
        selection,
        self.superblock.offset_size,
        self.superblock.length_size,
        self.superblock.base_address,
        custom_filters=self.custom_filters,
      )
  }
}
