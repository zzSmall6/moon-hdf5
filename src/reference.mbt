///|
/// The object reached by a non-null HDF5 object reference.
///
/// `path` is the first hard-link path found by a bounded hierarchy walk. It is
/// `None` when the target is valid but has no named hard link.
#derive(Eq, @debug.Debug)
pub(all) struct Hdf5ReferenceTarget {
  address : UInt64
  kind : Hdf5ObjectKind
  path : String?
}

///|
pub extend Hdf5ReferenceTarget with Eq::{not_equal, equal}

///|
pub extend Hdf5ReferenceTarget with @debug.Debug::{to_repr}

///|
/// The object and attribute reached by a version-4 attribute reference.
#derive(Eq, @debug.Debug)
pub(all) struct Hdf5AttributeReferenceTarget {
  address : UInt64
  kind : Hdf5ObjectKind
  path : String?
  attribute_name : String
}

///|
pub extend Hdf5AttributeReferenceTarget with Eq::{not_equal, equal}

///|
pub extend Hdf5AttributeReferenceTarget with @debug.Debug::{to_repr}

///|
fn reference_is_null(reference : Bytes) -> Bool {
  let mut index = 0
  while index < reference.length() {
    if reference.unsafe_get(index) != b'\x00' {
      return false
    }
    index = index + 1
  }
  true
}

///|
fn is_version4_reference(reference : Bytes, offset_size : Int32) -> Bool {
  reference.length() == offset_size + 10 && reference.length() > 0
}

///|
fn version4_reference_header(
  reference : Bytes,
) -> (Int32, Int32, Reader) raise Hdf5Error {
  let reader = Reader::new(reference)
  let kind = reader.read_u8().to_int()
  let flags = reader.read_u8().to_int()
  guard (flags & 0xFE) == 0 else {
    raise InvalidField("version-4 reference has unsupported flags")
  }
  (kind, flags, reader)
}

///|
fn Hdf5File::version4_indirect_payload(
  self : Hdf5File,
  flags : Int32,
  reader : Reader,
) -> Bytes raise Hdf5Error {
  guard flags == 0 else {
    raise UnsupportedFeature(
      "version-4 references to external files are not supported",
    )
  }
  let offset_size = self.superblock.offset_size
  let blob_size = reader.read_u32_le()
  guard blob_size <= 0x7FFFFFFFU else {
    raise InvalidField("version-4 reference blob is too large")
  }
  let heap_address = reader.read_uint_le(offset_size)
  let object_index = reader.read_u32_le()
  guard heap_address != 0UL &&
    heap_address != undefined_address &&
    object_index > 0U &&
    object_index <= 0xFFFFU else {
    raise InvalidField("invalid version-4 reference heap identifier")
  }
  let payload = self.read_reference_heap_object(
    heap_address,
    object_index.reinterpret_as_int(),
  )
  guard payload.length() == blob_size.reinterpret_as_int() else {
    raise InvalidField("version-4 reference blob size does not match the heap")
  }
  payload
}

///|
fn Hdf5File::version4_object_reference_target(
  self : Hdf5File,
  reference : Bytes,
) -> Hdf5ReferenceTarget? raise Hdf5Error {
  let offset_size = self.superblock.offset_size
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
  guard flags == 0 else {
    raise UnsupportedFeature(
      "version-4 references to external files are not supported",
    )
  }
  let token_size = reader.read_u8().to_int()
  guard token_size == offset_size else {
    raise UnsupportedFeature(
      "version-4 reference token size " +
      token_size.to_string() +
      " does not match the file address size",
    )
  }
  let address = reader.read_uint_le(token_size)
  let header = self.object_header_at(address) catch {
    error =>
      raise InvalidField(
        "object reference target is not a valid object header: " +
        error.message(),
      )
  }
  let kind = object_kind(header)
  let path = self.find_reference_path(address)
  Some({ address, kind, path, })
}

///|
fn Hdf5File::find_reference_path_in_group(
  self : Hdf5File,
  path : String,
  address : UInt64,
  target : UInt64,
  visited : Array[UInt64],
  depth : Int32,
) -> String? raise Hdf5Error {
  guard depth < 128 else { raise InvalidField("group hierarchy is too deep") }
  if visited.contains(address) {
    return None
  }
  visited.push(address)
  let header = self.object_header_at(address)
  for link in self.links(header) {
    match link.link_target {
      Some(_) => ()
      None => {
        let child_path = join_path(path, link.name)
        if link.object_header_address == target {
          return Some(child_path)
        }
        let child_kind = object_kind(
          self.object_header_at(link.object_header_address),
        ) catch {
          _ => Unknown
        }
        if child_kind == Group {
          let found = self.find_reference_path_in_group(
            child_path,
            link.object_header_address,
            target,
            visited,
            depth + 1,
          )
          if found is Some(found_path) {
            return Some(found_path)
          }
        }
      }
    }
  }
  None
}

///|
fn Hdf5File::find_reference_path(
  self : Hdf5File,
  target : UInt64,
) -> String? raise Hdf5Error {
  if target == self.root_address() {
    return Some("/")
  }
  let visited : Array[UInt64] = []
  self.find_reference_path_in_group(
    "/",
    self.root_address(),
    target,
    visited,
    0,
  )
}

///|
/// Resolves a legacy HDF5 object reference to its target object.
///
/// The all-zero null reference returns `None`. Region references must be
/// resolved with `resolve_region_reference`; malformed references and dangling
/// addresses raise an error. References are local to this file; use
/// `Hdf5Resolver` to obtain a file that contains a remotely linked dataset
/// before resolving its stored references.
pub fn Hdf5File::resolve_object_reference(
  self : Hdf5File,
  reference : Bytes,
) -> Hdf5ReferenceTarget? raise Hdf5Error {
  let offset_size = self.superblock.offset_size
  if is_version4_reference(reference, offset_size) {
    return self.version4_object_reference_target(reference)
  }
  guard reference.length() == offset_size else {
    if reference.length() == offset_size + 4 {
      raise InvalidArgument(
        "region references require resolve_region_reference",
      )
    }
    raise InvalidField(
      "object reference must be " +
      offset_size.to_string() +
      " bytes, found " +
      reference.length().to_string(),
    )
  }
  if reference_is_null(reference) {
    return None
  }
  let reader = Reader::new(reference)
  let address = reader.read_uint_le(offset_size)
  let header = self.object_header_at(address) catch {
    error =>
      raise InvalidField(
        "object reference target is not a valid object header: " +
        error.message(),
      )
  }
  let kind = object_kind(header)
  let path = self.find_reference_path(address)
  Some({ address, kind, path, })
}

///|
/// Resolves a version-4 attribute reference to its object and attribute name.
///
/// Null references return `None`. External-file and non-native tokens are
/// outside the supported profile. Use `read_attribute_reference` to load the
/// referenced attribute value.
pub fn Hdf5File::resolve_attribute_reference(
  self : Hdf5File,
  reference : Bytes,
) -> Hdf5AttributeReferenceTarget? raise Hdf5Error {
  let offset_size = self.superblock.offset_size
  guard is_version4_reference(reference, offset_size) else {
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
  let payload = self.version4_indirect_payload(flags, reader)
  let blob = Reader::new(payload)
  let token_size = blob.read_u8().to_int()
  guard token_size == offset_size else {
    raise UnsupportedFeature(
      "version-4 reference token size " +
      token_size.to_string() +
      " does not match the file address size",
    )
  }
  let address = blob.read_uint_le(token_size)
  let name_length = blob.read_u16_le().to_int()
  guard name_length <= blob.remaining() else {
    raise InvalidField("truncated version-4 attribute reference name")
  }
  let name_bytes = blob.read_bytes(name_length)
  guard blob.remaining() == 0 else {
    raise InvalidField("version-4 attribute reference has trailing bytes")
  }
  let attribute_name = @utf8.decode(name_bytes.get_view().unwrap()) catch {
    _ => raise InvalidField("invalid UTF-8 attribute reference name")
  }
  let header = self.object_header_at(address) catch {
    error =>
      raise InvalidField(
        "attribute reference target is not a valid object header: " +
        error.message(),
      )
  }
  let kind = object_kind(header)
  let path = self.find_reference_path(address)
  Some({ address, kind, path, attribute_name, })
}

///|
/// Reads the attribute selected by a version-4 attribute reference.
///
/// A null reference returns `None`. The target object must have a named hard
/// link so its attributes can be enumerated, and the named attribute must
/// still exist.
pub fn Hdf5File::read_attribute_reference(
  self : Hdf5File,
  reference : Bytes,
) -> Attribute? raise Hdf5Error {
  let target = match self.resolve_attribute_reference(reference) {
    Some(target) => target
    None => return None
  }
  let path = match target.path {
    Some(path) => path
    None =>
      raise InvalidField("attribute-reference target has no named hard link")
  }
  let attributes = self.attributes(path)
  match
    attributes.search_by(attribute => attribute.name == target.attribute_name) {
    Some(index) => Some(attributes[index])
    None =>
      raise InvalidField(
        "attribute-reference target does not exist: " + target.attribute_name,
      )
  }
}
