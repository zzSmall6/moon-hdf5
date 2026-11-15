///|
fn external_reference_resolver() -> Hdf5Resolver raise Error {
  let resolver = Hdf5Resolver::new(
    @fs.read_file_to_bytes("testdata/external_references.h5"),
  )
  resolver.add_source(
    "external_reference_target.h5",
    @fs.read_file_to_bytes("testdata/external_reference_target.h5"),
  )
  resolver
}

///|
test "resolver follows version-4 object references into external files" {
  let bytes = @fs.read_file_to_bytes("testdata/external_references.h5")
  let file = Hdf5File::parse(bytes)
  let references = reference_value_bytes(file.read_dataset("/object_refs"))
  let resolver = external_reference_resolver()

  let target = resolver.resolve_object_reference(references[0]).unwrap()
  assert_eq(target.file, "external_reference_target.h5")
  assert_eq(target.target.path, Some("/target"))
  assert_eq(target.target.kind, Dataset)

  let group = resolver.resolve_object_reference(references[1]).unwrap()
  assert_eq(group.target.path, Some("/group"))
  assert_eq(group.target.kind, Group)

  let child = resolver.resolve_object_reference(references[2]).unwrap()
  assert_eq(child.target.path, Some("/group/child"))
  assert_eq(child.target.kind, Dataset)

  let local_target = resolver.resolve_object_reference(references[3]).unwrap()
  assert_eq(local_target.file, "<root>")
  assert_eq(local_target.target.path, Some("/local_target"))
  assert_true(resolver.resolve_object_reference(references[4]) is None)

  let direct_error = try {
    file.resolve_object_reference(references[0]) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_true(
    direct_error
    is Some(
      UnsupportedFeature(
        "version-4 references to external files require Hdf5Resolver"
      )
    ),
  )
}

///|
test "resolver reads version-4 region references in external files" {
  let file = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/external_references.h5"),
  )
  let references = reference_value_bytes(file.read_dataset("/region_refs"))
  let resolver = external_reference_resolver()

  let hyperslab = resolver.resolve_region_reference(references[0]).unwrap()
  assert_eq(hyperslab.file, "external_reference_target.h5")
  assert_eq(hyperslab.target.path, Some("/target"))
  assert_region_hyperslab(hyperslab.target.selection, [1UL], [2UL], [2UL], [1UL])
  assert_eq(
    selection_result_integer_values(
      resolver.read_region_reference(references[0]).unwrap(),
    ),
    [1L, 3L],
  )

  let points = resolver.resolve_region_reference(references[1]).unwrap()
  match points.target.selection {
    RegionPoints(values) => assert_eq(values, [[4UL], [0UL]])
    _ => fail("expected external point selection")
  }
  assert_eq(
    selection_result_integer_values(
      resolver.read_region_reference(references[1]).unwrap(),
    ),
    [4L, 0L],
  )

  let all = resolver.resolve_region_reference(references[2]).unwrap()
  assert_eq(all.target.selection, RegionAll)
  assert_eq(
    selection_result_integer_values(
      resolver.read_region_reference(references[2]).unwrap(),
    ),
    [0L, 1L, 2L, 3L, 4L, 5L],
  )

  let empty = resolver.resolve_region_reference(references[3]).unwrap()
  assert_eq(empty.target.selection, RegionEmpty)
  assert_eq(
    selection_result_integer_values(
      resolver.read_region_reference(references[3]).unwrap(),
    ),
    [],
  )
  assert_true(resolver.resolve_region_reference(references[4]) is None)
  assert_true(resolver.read_region_reference(references[4]) is None)
}

///|
test "resolver reads version-4 attribute references in external files" {
  let file = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/external_references.h5"),
  )
  let references = reference_value_bytes(file.read_dataset("/attribute_refs"))
  let resolver = external_reference_resolver()

  let target = resolver.resolve_attribute_reference(references[0]).unwrap()
  assert_eq(target.file, "external_reference_target.h5")
  assert_eq(target.target.path, Some("/target"))
  assert_eq(target.target.attribute_name, "unit")
  let attribute = resolver.read_attribute_reference(references[0]).unwrap()
  assert_eq(version4_attribute_integer_values(attribute), [7L])
  assert_true(resolver.resolve_attribute_reference(references[1]) is None)
  assert_true(resolver.read_attribute_reference(references[1]) is None)
}

///|
test "external reference resolution requires a registered target source" {
  let file = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/external_references.h5"),
  )
  let references = reference_value_bytes(file.read_dataset("/object_refs"))
  let resolver = Hdf5Resolver::new(
    @fs.read_file_to_bytes("testdata/external_references.h5"),
  )
  let error = try {
    resolver.resolve_object_reference(references[0]) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    error,
    Some(
      InvalidField(
        "external target file is not registered: external_reference_target.h5",
      ),
    ),
  )
}
