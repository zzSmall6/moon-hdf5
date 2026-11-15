///|
/// An object reference resolved across the root file and registered external
/// HDF5 sources.
#derive(@debug.Debug)
pub(all) struct Hdf5ResolvedObjectReference {
  file : String
  target : Hdf5ReferenceTarget
}

///|
pub extend Hdf5ResolvedObjectReference with @debug.Debug::{to_repr}

///|
/// A dataset-region reference resolved across registered HDF5 sources.
#derive(@debug.Debug)
pub(all) struct Hdf5ResolvedRegionReference {
  file : String
  target : Hdf5RegionReferenceTarget
}

///|
pub extend Hdf5ResolvedRegionReference with @debug.Debug::{to_repr}

///|
/// An attribute reference resolved across registered HDF5 sources.
#derive(@debug.Debug)
pub(all) struct Hdf5ResolvedAttributeReference {
  file : String
  target : Hdf5AttributeReferenceTarget
}

///|
pub extend Hdf5ResolvedAttributeReference with @debug.Debug::{to_repr}

///|
fn Hdf5Resolver::reference_source(
  self : Hdf5Resolver,
  source_name : String,
) -> (String, Hdf5File) raise Hdf5Error {
  let normalized = if source_name == resolver_root_name() {
    resolver_root_name()
  } else {
    resolver_file_name(source_name)
  }
  let file = self.open(normalized)
  (normalized, file)
}

///|
fn Hdf5Resolver::external_reference_source(
  self : Hdf5Resolver,
  referring_file : String,
  file : Hdf5File,
  body : Version4ReferenceBody,
) -> (String, Hdf5File) raise Hdf5Error {
  match body.external_file {
    None => (referring_file, file)
    Some(name) => {
      let normalized = external_storage_file_name(name)
      let target_file = self.external_source_name(referring_file, normalized)
      let target = self.open(target_file)
      (target_file, target)
    }
  }
}

///|
fn reference_object_at(
  file_name : String,
  file : Hdf5File,
  address : UInt64,
) -> Hdf5ResolvedObjectReference raise Hdf5Error {
  let header = file.object_header_at(address) catch {
    error =>
      raise InvalidField(
        "object reference target is not a valid object header: " +
        error.message(),
      )
  }
  {
    file: file_name,
    target: {
      address,
      kind: object_kind(header),
      path: file.find_reference_path(address),
    },
  }
}

///|
fn Hdf5Resolver::resolve_object_reference_in(
  self : Hdf5Resolver,
  source_name : String,
  reference : Bytes,
) -> Hdf5ResolvedObjectReference? raise Hdf5Error {
  let (file_name, file) = self.reference_source(source_name)
  if !is_version4_reference(reference, file.superblock.offset_size) {
    let target = file.resolve_object_reference(reference)
    return target.map(target => { file: file_name, target, })
  }
  let (kind, flags, reader) = version4_reference_header(reference)
  if kind == 0 {
    guard flags == 0 && reference_is_null(reference) else {
      raise InvalidField("invalid null version-4 reference")
    }
    return None
  }
  guard kind == 2 else {
    if kind == 3 {
      raise InvalidArgument(
        "region references require resolve_region_reference",
      )
    } else if kind == 4 {
      raise InvalidArgument(
        "attribute references require resolve_attribute_reference",
      )
    }
    raise InvalidField("invalid version-4 reference type")
  }
  if flags == 0 {
    let target = file.resolve_object_reference(reference)
    return target.map(target => { file: file_name, target, })
  }
  let body = file.version4_reference_body(kind, flags, reader)
  let (target_file_name, target_file) = self.external_reference_source(
    file_name, file, body,
  )
  guard body.token_size == target_file.superblock.offset_size else {
    raise UnsupportedFeature(
      "version-4 reference token size " +
      body.token_size.to_string() +
      " does not match the external file address size",
    )
  }
  Some(reference_object_at(target_file_name, target_file, body.token_address))
}

///|
/// Resolves an object reference stored in the root source, including
/// version-4 references to registered external HDF5 files.
pub fn Hdf5Resolver::resolve_object_reference(
  self : Hdf5Resolver,
  reference : Bytes,
) -> Hdf5ResolvedObjectReference? raise Hdf5Error {
  self.resolve_object_reference_in(resolver_root_name(), reference)
}

///|
fn Hdf5Resolver::resolve_version4_region_in(
  self : Hdf5Resolver,
  file_name : String,
  file : Hdf5File,
  kind : Int32,
  flags : Int32,
  reader : Reader,
) -> Hdf5ResolvedRegionReference? raise Hdf5Error {
  let body = file.version4_reference_body(kind, flags, reader)
  let (target_file_name, target_file) = self.external_reference_source(
    file_name, file, body,
  )
  guard body.token_size == target_file.superblock.offset_size else {
    raise UnsupportedFeature(
      "version-4 reference token size " +
      body.token_size.to_string() +
      " does not match the external file address size",
    )
  }
  let address = body.token_address
  let header = target_file.object_header_at(address) catch {
    error =>
      raise InvalidField(
        "version-4 region reference target is not a valid object header: " +
        error.message(),
      )
  }
  guard object_kind(header) == Dataset else {
    raise InvalidField("region-reference target is not a dataset")
  }
  let metadata = read_dataset_metadata(
    header,
    target_file.superblock.offset_size,
    target_file.superblock.length_size,
  )
  guard body.rank == metadata.dataspace.dimensions.length() else {
    raise InvalidField("version-4 region selection rank does not match")
  }
  let selection_bytes = match body.selection {
    Some(bytes) => bytes
    None => raise InvalidField("version-4 region reference has no selection")
  }
  let selection = parse_region_selection(
    Reader::new(selection_bytes),
    metadata.dataspace,
  )
  Some({
    file: target_file_name,
    target: {
      address,
      kind: Dataset,
      path: target_file.find_reference_path(address),
      dataspace: metadata.dataspace,
      selection,
    },
  })
}

///|
fn Hdf5Resolver::resolve_region_reference_in(
  self : Hdf5Resolver,
  source_name : String,
  reference : Bytes,
) -> Hdf5ResolvedRegionReference? raise Hdf5Error {
  let (file_name, file) = self.reference_source(source_name)
  let offset_size = file.superblock.offset_size
  if !is_version4_reference(reference, offset_size) {
    let target = file.resolve_region_reference(reference)
    return target.map(target => { file: file_name, target, })
  }
  let (kind, flags, reader) = version4_reference_header(reference)
  if kind == 0 {
    guard flags == 0 && reference_is_null(reference) else {
      raise InvalidField("invalid null version-4 reference")
    }
    return None
  }
  guard kind == 3 else {
    if kind == 2 {
      raise InvalidArgument(
        "object references require resolve_object_reference",
      )
    } else if kind == 4 {
      raise InvalidArgument(
        "attribute references require resolve_attribute_reference",
      )
    }
    raise InvalidField("invalid version-4 reference type")
  }
  self.resolve_version4_region_in(file_name, file, kind, flags, reader)
}

///|
/// Resolves a region reference stored in the root source, including
/// version-4 references to registered external HDF5 files.
pub fn Hdf5Resolver::resolve_region_reference(
  self : Hdf5Resolver,
  reference : Bytes,
) -> Hdf5ResolvedRegionReference? raise Hdf5Error {
  self.resolve_region_reference_in(resolver_root_name(), reference)
}

///|
fn Hdf5Resolver::resolve_attribute_reference_in(
  self : Hdf5Resolver,
  source_name : String,
  reference : Bytes,
) -> Hdf5ResolvedAttributeReference? raise Hdf5Error {
  let (file_name, file) = self.reference_source(source_name)
  guard is_version4_reference(reference, file.superblock.offset_size) else {
    raise InvalidField("attribute references require version-4 encoding")
  }
  let (kind, flags, reader) = version4_reference_header(reference)
  if kind == 0 {
    guard flags == 0 && reference_is_null(reference) else {
      raise InvalidField("invalid null version-4 reference")
    }
    return None
  }
  guard kind == 4 else {
    if kind == 2 {
      raise InvalidArgument(
        "object references require resolve_object_reference",
      )
    } else if kind == 3 {
      raise InvalidArgument(
        "region references require resolve_region_reference",
      )
    }
    raise InvalidField("invalid version-4 reference type")
  }
  let body = file.version4_reference_body(kind, flags, reader)
  let (target_file_name, target_file) = self.external_reference_source(
    file_name, file, body,
  )
  guard body.token_size == target_file.superblock.offset_size else {
    raise UnsupportedFeature(
      "version-4 reference token size " +
      body.token_size.to_string() +
      " does not match the external file address size",
    )
  }
  let address = body.token_address
  let header = target_file.object_header_at(address) catch {
    error =>
      raise InvalidField(
        "attribute reference target is not a valid object header: " +
        error.message(),
      )
  }
  let attribute_name = match body.attribute_name {
    Some(name) => name
    None => raise InvalidField("version-4 attribute reference has no name")
  }
  Some({
    file: target_file_name,
    target: {
      address,
      kind: object_kind(header),
      path: target_file.find_reference_path(address),
      attribute_name,
    },
  })
}

///|
/// Resolves an attribute reference stored in the root source, including
/// version-4 references to registered external HDF5 files.
pub fn Hdf5Resolver::resolve_attribute_reference(
  self : Hdf5Resolver,
  reference : Bytes,
) -> Hdf5ResolvedAttributeReference? raise Hdf5Error {
  self.resolve_attribute_reference_in(resolver_root_name(), reference)
}

///|
fn Hdf5Resolver::read_attribute_reference_in(
  self : Hdf5Resolver,
  source_name : String,
  reference : Bytes,
) -> Attribute? raise Hdf5Error {
  let resolved = match
    self.resolve_attribute_reference_in(source_name, reference) {
    Some(resolved) => resolved
    None => return None
  }
  let path = match resolved.target.path {
    Some(path) => path
    None =>
      raise InvalidField("attribute-reference target has no named hard link")
  }
  let file = self.open(resolved.file)
  let attributes = file.attributes(path)
  match
    attributes.search_by(attribute => {
      attribute.name == resolved.target.attribute_name
    }) {
    Some(index) => Some(attributes[index])
    None =>
      raise InvalidField(
        "attribute-reference target does not exist: " +
        resolved.target.attribute_name,
      )
  }
}

///|
/// Reads the attribute selected by a reference stored in the root source,
/// including version-4 references to registered external HDF5 files.
pub fn Hdf5Resolver::read_attribute_reference(
  self : Hdf5Resolver,
  reference : Bytes,
) -> Attribute? raise Hdf5Error {
  self.read_attribute_reference_in(resolver_root_name(), reference)
}

///|
fn Hdf5Resolver::read_region_reference_in(
  self : Hdf5Resolver,
  source_name : String,
  reference : Bytes,
) -> DatasetSelectionResult? raise Hdf5Error {
  let resolved = match
    self.resolve_region_reference_in(source_name, reference) {
    Some(resolved) => resolved
    None => return None
  }
  let file = self.open(resolved.file)
  let target = resolved.target
  let header = file.object_header_at(target.address)
  let metadata = read_dataset_metadata(
    header,
    file.superblock.offset_size,
    file.superblock.length_size,
  )
  let (dimensions, indices) = region_selection_indices(
    metadata.dataspace,
    target.selection,
  )
  let result = match file.source {
    Some(source) =>
      read_source_dataset_indices(
        source,
        header,
        metadata,
        dimensions,
        indices,
        file.superblock.offset_size,
        file.superblock.length_size,
        file.superblock.base_address,
        custom_filters=file.custom_filters,
      )
    None =>
      read_dataset_indices(
        file.bytes,
        header,
        metadata,
        dimensions,
        indices,
        file.superblock.offset_size,
        file.superblock.length_size,
        file.superblock.base_address,
        custom_filters=file.custom_filters,
      )
  }
  Some(result)
}

///|
/// Reads the values selected by a region reference stored in the root source,
/// including version-4 references to registered external HDF5 files.
pub fn Hdf5Resolver::read_region_reference(
  self : Hdf5Resolver,
  reference : Bytes,
) -> DatasetSelectionResult? raise Hdf5Error {
  self.read_region_reference_in(resolver_root_name(), reference)
}
