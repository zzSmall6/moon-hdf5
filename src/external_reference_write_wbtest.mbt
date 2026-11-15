///|
fn external_reference_target_bytes() -> Bytes raise Hdf5Error {
  let writer = Hdf5Writer::new()
  writer.add_dataset(
    "/matrix",
    [2UL, 3UL],
    Int64Values([0L, 1L, 2L, 3L, 4L, 5L]),
  )
  writer.add_group("/group")
  writer.add_dataset("/group/child", [1UL], Int64Values([9L]))
  writer.add_attribute("/matrix", "unit", [], Int64Values([7L]))
  writer.encode()
}

///|
fn external_reference_resolver_for(
  bytes : Bytes,
  target : Bytes,
) -> Hdf5Resolver raise Hdf5Error {
  let resolver = Hdf5Resolver::new(bytes)
  resolver.add_source("external_target.h5", target)
  resolver
}

///|
test "writer emits external version-4 object, region and attribute references" {
  let target = external_reference_target_bytes()
  let writer = Hdf5Writer::new()
  writer.add_external_source("external_target.h5", target)
  let object_type = Datatype::write_version4_object_reference()
  let region_type = Datatype::write_version4_region_reference()
  let attribute_type = Datatype::write_version4_attribute_reference()
  writer.add_dataset(
    "/objects",
    [3UL],
    StructuredValues(object_type, [
      ExternalObjectReference("external_target.h5", "/matrix"),
      ExternalObjectReference("external_target.h5", "/group"),
      ExternalObjectReference("external_target.h5", "/group/child"),
    ]),
  )
  writer.add_dataset(
    "/regions",
    [2UL],
    StructuredValues(region_type, [
      ExternalRegionReference(
        "external_target.h5",
        "/matrix",
        RegionPoints([[1UL, 2UL]]),
      ),
      ExternalRegionReference("external_target.h5", "/matrix", RegionAll),
    ]),
  )
  writer.add_dataset(
    "/attributes",
    [1UL],
    StructuredValues(attribute_type, [
      ExternalAttributeReference("external_target.h5", "/matrix", "unit"),
    ]),
  )
  let source = writer.encode()
  let resolver = external_reference_resolver_for(source, target)

  let objects = reference_value_bytes(
    Hdf5File::parse(source).read_dataset("/objects"),
  )
  assert_eq(objects[0].length(), 18)
  assert_eq(
    resolver.resolve_object_reference(objects[0]).unwrap().target.path,
    Some("/matrix"),
  )
  assert_eq(
    resolver.resolve_object_reference(objects[1]).unwrap().target.kind,
    Group,
  )
  assert_eq(
    resolver.resolve_object_reference(objects[2]).unwrap().file,
    "external_target.h5",
  )

  let regions = reference_value_bytes(
    Hdf5File::parse(source).read_dataset("/regions"),
  )
  let point = resolver.resolve_region_reference(regions[0]).unwrap()
  assert_eq(point.file, "external_target.h5")
  match point.target.selection {
    RegionPoints(points) => assert_eq(points, [[1UL, 2UL]])
    _ => fail("expected an external point selection")
  }
  assert_eq(
    resolver.resolve_region_reference(regions[1]).unwrap().target.selection,
    RegionAll,
  )

  let attributes = reference_value_bytes(
    Hdf5File::parse(source).read_dataset("/attributes"),
  )
  let attribute = resolver.resolve_attribute_reference(attributes[0]).unwrap()
  assert_eq(attribute.file, "external_target.h5")
  assert_eq(attribute.target.path, Some("/matrix"))
  assert_eq(attribute.target.attribute_name, "unit")
  assert_eq(
    version4_attribute_integer_values(
      resolver.read_attribute_reference(attributes[0]).unwrap(),
    ),
    [7L],
  )
}

///|
test "writer rejects unregistered and duplicate external sources" {
  let writer = Hdf5Writer::new()
  let target = external_reference_target_bytes()
  writer.add_external_source("external_target.h5", target)
  let duplicate = try {
    writer.add_external_source("external_target.h5", target)
    None
  } catch {
    error => Some(error)
  }
  assert_true(
    duplicate
    is Some(
      InvalidArgument(
        "external source is already registered: external_target.h5"
      )
    ),
  )

  let object_type = Datatype::write_version4_object_reference()
  let encoded = try {
    writer.add_dataset(
      "/objects",
      [1UL],
      StructuredValues(object_type, [
        ExternalObjectReference("missing.h5", "/matrix"),
      ]),
    )
    writer.encode() |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_true(
    encoded
    is Some(
      InvalidArgument("external reference file is not registered: missing.h5")
    ),
  )
}

///|
test "writer uses the external file's four-byte address width" {
  let target = @fs.read_file_to_bytes("testdata/nested_reference_32.h5")
  let writer = Hdf5Writer::new()
  writer.add_external_source("nested_reference_32.h5", target)
  let object_type = Datatype::write_version4_object_reference()
  let region_type = Datatype::write_version4_region_reference()
  let attribute_type = Datatype::write_version4_attribute_reference()
  writer.add_dataset(
    "/objects",
    [1UL],
    StructuredValues(object_type, [
      ExternalObjectReference("nested_reference_32.h5", "/target"),
    ]),
  )
  writer.add_dataset(
    "/regions",
    [1UL],
    StructuredValues(region_type, [
      ExternalRegionReference(
        "nested_reference_32.h5",
        "/target",
        RegionPoints([[2UL]]),
      ),
    ]),
  )
  writer.add_dataset(
    "/attributes",
    [1UL],
    StructuredValues(attribute_type, [
      ExternalAttributeReference("nested_reference_32.h5", "/target", "unit"),
    ]),
  )
  let source = writer.encode()
  let resolver = Hdf5Resolver::new(source)
  resolver.add_source("nested_reference_32.h5", target)
  let objects = reference_value_bytes(
    Hdf5File::parse(source).read_dataset("/objects"),
  )
  assert_eq(
    resolver.resolve_object_reference(objects[0]).unwrap().target.path,
    Some("/target"),
  )
  let regions = reference_value_bytes(
    Hdf5File::parse(source).read_dataset("/regions"),
  )
  assert_eq(
    resolver.resolve_region_reference(regions[0]).unwrap().target.selection,
    RegionPoints([[2UL]]),
  )
  let attributes = reference_value_bytes(
    Hdf5File::parse(source).read_dataset("/attributes"),
  )
  assert_eq(
    resolver.resolve_attribute_reference(attributes[0]).unwrap().target.attribute_name,
    "unit",
  )
}

///|
test "chunked and nested external references resolve after writer encoding" {
  let target = external_reference_target_bytes()
  let writer = Hdf5Writer::new()
  writer.add_external_source("external_target.h5", target)
  let object_type = Datatype::write_version4_object_reference()
  let region_type = Datatype::write_version4_region_reference()
  let attribute_type = Datatype::write_version4_attribute_reference()
  let compound = Datatype::write_compound([
    ("obj", object_type),
    ("region", region_type),
    ("attr", attribute_type),
  ])
  let value = Compound([
    {
      name: "obj",
      value: ExternalObjectReference("external_target.h5", "/group/child"),
    },
    {
      name: "region",
      value: ExternalRegionReference(
        "external_target.h5",
        "/matrix",
        RegionPoints([[0UL, 1UL]]),
      ),
    },
    {
      name: "attr",
      value: ExternalAttributeReference("external_target.h5", "/matrix", "unit"),
    },
  ])
  writer.add_chunked_dataset(
    "/chunked",
    [1UL],
    [1U],
    StructuredValues(compound, [value]),
  )
  writer.add_attribute(
    "/chunked",
    "nested",
    [1UL],
    StructuredValues(compound, [value]),
  )
  let source = writer.encode()
  let resolver = external_reference_resolver_for(source, target)
  let file = Hdf5File::parse(source)
  let fields = match file.read_dataset("/chunked").values {
    Structured([Compound(fields)]) => fields
    _ => fail("expected one chunked compound value")
  }
  let object = match fields[0].value {
    Reference(bytes) => bytes
    _ => fail("expected a nested object reference")
  }
  assert_eq(
    resolver.resolve_object_reference(object).unwrap().target.path,
    Some("/group/child"),
  )
  let region = match fields[1].value {
    Reference(bytes) => bytes
    _ => fail("expected a nested region reference")
  }
  assert_eq(
    resolver.resolve_region_reference(region).unwrap().target.selection,
    RegionPoints([[0UL, 1UL]]),
  )
  let attribute = match fields[2].value {
    Reference(bytes) => bytes
    _ => fail("expected a nested attribute reference")
  }
  assert_eq(
    resolver.resolve_attribute_reference(attribute).unwrap().target.attribute_name,
    "unit",
  )

  let attributes = file.attributes("/chunked")
  let nested = match
    attributes.search_by(attribute => attribute.name == "nested") {
    Some(index) => attributes[index]
    None => fail("nested attribute is missing")
  }
  let attribute_fields = match nested.values {
    Some(Structured([Compound(fields)])) => fields
    _ => fail("expected a nested compound attribute")
  }
  let attribute_reference = match attribute_fields[2].value {
    Reference(bytes) => bytes
    _ => fail("expected a nested attribute reference")
  }
  assert_eq(
    resolver.resolve_attribute_reference(attribute_reference).unwrap().file,
    "external_target.h5",
  )
}

///|
test "existing datasets and attributes accept external version-4 references" {
  let target = external_reference_target_bytes()
  let object_type = Datatype::write_version4_object_reference()
  let region_type = Datatype::write_version4_region_reference()
  let attribute_type = Datatype::write_version4_attribute_reference()
  let writer = Hdf5Writer::new()
  writer.add_dataset("/local", [1UL], Int64Values([1L]))
  writer.add_dataset(
    "/objects",
    [1UL],
    StructuredValues(object_type, [ObjectReference(None)]),
  )
  writer.add_dataset(
    "/regions",
    [1UL],
    StructuredValues(region_type, [RegionReference(None, RegionEmpty)]),
  )
  writer.add_attribute(
    "/objects",
    "remote",
    [],
    StructuredValues(attribute_type, [AttributeReference(None, "")]),
  )
  let file = Hdf5File::parse(writer.encode()).with_external_source(
    "external_target.h5", target,
  )

  let with_object = file.patch_dataset_values(
    "/objects",
    StructuredValues(object_type, [
      ExternalObjectReference("external_target.h5", "/group/child"),
    ]),
  )
  let object_resolver = external_reference_resolver_for(with_object, target)
  let objects = reference_value_bytes(
    Hdf5File::parse(with_object).read_dataset("/objects"),
  )
  assert_eq(
    object_resolver.resolve_object_reference(objects[0]).unwrap().target.path,
    Some("/group/child"),
  )

  let with_region = file.patch_dataset_values(
    "/regions",
    StructuredValues(region_type, [
      ExternalRegionReference("external_target.h5", "/matrix", RegionAll),
    ]),
  )
  let region_resolver = external_reference_resolver_for(with_region, target)
  let regions = reference_value_bytes(
    Hdf5File::parse(with_region).read_dataset("/regions"),
  )
  assert_eq(
    region_resolver.resolve_region_reference(regions[0]).unwrap().target.selection,
    RegionAll,
  )

  let with_attribute = file.patch_attribute_values(
    "/objects",
    "remote",
    StructuredValues(attribute_type, [
      ExternalAttributeReference("external_target.h5", "/matrix", "unit"),
    ]),
  )
  let attribute_resolver = external_reference_resolver_for(
    with_attribute, target,
  )
  let attributes = Hdf5File::parse(with_attribute).attributes("/objects")
  let remote = match
    attributes.search_by(attribute => attribute.name == "remote") {
    Some(index) => attributes[index]
    None => fail("remote attribute is missing")
  }
  let reference = match remote.values {
    Some(Structured([Reference(bytes)])) => bytes
    _ => fail("expected an external attribute reference")
  }
  let resolved = attribute_resolver
    .resolve_attribute_reference(reference)
    .unwrap()
  assert_eq(resolved.file, "external_target.h5")
  assert_eq(resolved.target.attribute_name, "unit")
}
