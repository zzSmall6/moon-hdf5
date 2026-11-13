///|
test "version-4 reference datasets and attributes can be patched" {
  let writer = Hdf5Writer::new()
  writer.add_dataset("/target", [6UL], Int64Values([0L, 1L, 2L, 3L, 4L, 5L]))
  writer.add_dataset(
    "/matrix",
    [2UL, 3UL],
    Int64Values([0L, 1L, 2L, 3L, 4L, 5L]),
  )
  writer.add_group("/group")
  writer.add_attribute("/target", "unit", [], Int64Values([7L]))
  writer.add_attribute("/matrix", "unit", [], Int64Values([11L]))
  writer.add_attribute("/group", "label", [], FixedStringValues(["group"], 8))

  let object_type = Datatype::write_version4_object_reference()
  let region_type = Datatype::write_version4_region_reference()
  let attribute_type = Datatype::write_version4_attribute_reference()
  writer.add_dataset(
    "/objects",
    [4UL],
    StructuredValues(object_type, [
      ObjectReference(Some("/target")),
      ObjectReference(Some("/group")),
      ObjectReference(Some("/matrix")),
      ObjectReference(None),
    ]),
  )
  writer.add_attribute(
    "/objects",
    "refs",
    [2UL],
    StructuredValues(object_type, [
      ObjectReference(Some("/target")),
      ObjectReference(None),
    ]),
  )
  writer.add_dataset(
    "/regions",
    [3UL],
    StructuredValues(region_type, [
      RegionReference(Some("/target"), RegionAll),
      RegionReference(Some("/matrix"), RegionPoints([[0UL, 1UL]])),
      RegionReference(None, RegionEmpty),
    ]),
  )
  writer.add_attribute(
    "/regions",
    "refs",
    [2UL],
    StructuredValues(region_type, [
      RegionReference(Some("/target"), RegionPoints([[2UL]])),
      RegionReference(None, RegionEmpty),
    ]),
  )
  writer.add_dataset(
    "/attributes",
    [3UL],
    StructuredValues(attribute_type, [
      AttributeReference(Some("/target"), "unit"),
      AttributeReference(Some("/matrix"), "unit"),
      AttributeReference(None, ""),
    ]),
  )
  writer.add_attribute(
    "/attributes",
    "refs",
    [1UL],
    StructuredValues(attribute_type, [
      AttributeReference(Some("/target"), "unit"),
    ]),
  )

  let file = Hdf5File::parse(writer.encode())
  let object_patched = file.patch_dataset_values(
    "/objects",
    ObjectReferenceValues([
      Some("/target"),
      Some("/matrix"),
      Some("/group"),
      None,
    ]),
  )
  let object_file = Hdf5File::parse(object_patched)
  let objects = reference_value_bytes(object_file.read_dataset("/objects"))
  let target = object_file.resolve("/target")
  let matrix = object_file.resolve("/matrix")
  let group = object_file.resolve("/group")
  assert_eq(objects[0].length(), 18)
  assert_eq(
    object_file.resolve_object_reference(objects[0]),
    Some({ address: target.address, kind: Dataset, path: Some("/target"), }),
  )
  assert_eq(
    object_file.resolve_object_reference(objects[1]),
    Some({ address: matrix.address, kind: Dataset, path: Some("/matrix"), }),
  )
  assert_eq(
    object_file.resolve_object_reference(objects[2]),
    Some({ address: group.address, kind: Group, path: Some("/group"), }),
  )
  assert_true(object_file.resolve_object_reference(objects[3]) is None)

  let object_attribute_patched = object_file.patch_attribute_values(
    "/objects",
    "refs",
    ObjectReferenceValues([Some("/matrix"), Some("/target")]),
  )
  let object_attribute_file = Hdf5File::parse(object_attribute_patched)
  match
    find_attribute(object_attribute_file.attributes("/objects"), "refs").values {
    Some(Structured([Reference(first), Reference(second)])) => {
      assert_eq(
        object_attribute_file.resolve_object_reference(first),
        Some({ address: matrix.address, kind: Dataset, path: Some("/matrix"), }),
      )
      assert_eq(
        object_attribute_file.resolve_object_reference(second),
        Some({ address: target.address, kind: Dataset, path: Some("/target"), }),
      )
    }
    _ => fail("expected patched version-4 object-reference attribute values")
  }

  let region_metadata = object_attribute_file.read_dataset_metadata("/regions").datatype
  let region_patched = object_attribute_file.patch_dataset_values(
    "/regions",
    StructuredValues(region_metadata, [
      RegionReference(
        Some("/matrix"),
        RegionHyperslab([0UL, 0UL], [1UL, 1UL], [2UL, 2UL], [1UL, 1UL]),
      ),
      RegionReference(None, RegionEmpty),
      RegionReference(
        Some("/target"),
        RegionBlocks([
          { start: [0UL], end: [1UL], },
          { start: [5UL], end: [5UL], },
        ]),
      ),
    ]),
  )
  let region_file = Hdf5File::parse(region_patched)
  let regions = reference_value_bytes(region_file.read_dataset("/regions"))
  assert_eq(
    patched_region_selection(region_file, regions[0]),
    (
      Some("/matrix"),
      RegionHyperslab([0UL, 0UL], [1UL, 1UL], [2UL, 2UL], [1UL, 1UL]),
    ),
  )
  assert_true(region_file.resolve_region_reference(regions[1]) is None)
  assert_eq(
    patched_region_selection(region_file, regions[2]),
    (
      Some("/target"),
      RegionBlocks([
        { start: [0UL], end: [1UL], },
        { start: [5UL], end: [5UL], },
      ]),
    ),
  )

  let region_attribute_patched = region_file.patch_attribute_values(
    "/regions",
    "refs",
    StructuredValues(region_metadata, [
      RegionReference(Some("/matrix"), RegionAll),
      RegionReference(Some("/target"), RegionPoints([[4UL]])),
    ]),
  )
  let region_attribute_file = Hdf5File::parse(region_attribute_patched)
  match
    find_attribute(region_attribute_file.attributes("/regions"), "refs").values {
    Some(Structured([Reference(first), Reference(second)])) => {
      assert_eq(
        patched_region_selection(region_attribute_file, first),
        (Some("/matrix"), RegionAll),
      )
      assert_eq(
        patched_region_selection(region_attribute_file, second),
        (Some("/target"), RegionPoints([[4UL]])),
      )
    }
    _ => fail("expected patched version-4 region-reference attribute values")
  }

  let attribute_metadata = region_attribute_file.read_dataset_metadata(
      "/attributes",
    ).datatype
  let attribute_patched = region_attribute_file.patch_dataset_values(
    "/attributes",
    StructuredValues(attribute_metadata, [
      AttributeReference(Some("/group"), "label"),
      AttributeReference(Some("/target"), "unit"),
      AttributeReference(None, ""),
    ]),
  )
  let attribute_file = Hdf5File::parse(attribute_patched)
  let attributes = reference_value_bytes(
    attribute_file.read_dataset("/attributes"),
  )
  assert_eq(
    attribute_file.resolve_attribute_reference(attributes[0]),
    Some({
      address: group.address,
      kind: Group,
      path: Some("/group"),
      attribute_name: "label",
    }),
  )
  assert_eq(
    attribute_file.resolve_attribute_reference(attributes[1]),
    Some({
      address: target.address,
      kind: Dataset,
      path: Some("/target"),
      attribute_name: "unit",
    }),
  )
  assert_true(attribute_file.resolve_attribute_reference(attributes[2]) is None)

  let attribute_reference_patched = attribute_file.patch_attribute_values(
    "/attributes",
    "refs",
    StructuredValues(attribute_metadata, [
      AttributeReference(Some("/matrix"), "unit"),
    ]),
  )
  let attribute_reference_file = Hdf5File::parse(attribute_reference_patched)
  match
    find_attribute(attribute_reference_file.attributes("/attributes"), "refs").values {
    Some(Structured([Reference(reference)])) =>
      assert_eq(
        attribute_reference_file.resolve_attribute_reference(reference),
        Some({
          address: matrix.address,
          kind: Dataset,
          path: Some("/matrix"),
          attribute_name: "unit",
        }),
      )
    _ => fail("expected a patched version-4 attribute-reference attribute")
  }
}

///|
test "compact and filtered chunked version-4 references can be patched" {
  let writer = Hdf5Writer::new()
  writer.add_dataset("/target", [5UL], Int64Values([0L, 1L, 2L, 3L, 4L]))
  writer.add_group("/group")
  let object_type = Datatype::write_version4_object_reference()
  let region_type = Datatype::write_version4_region_reference()
  writer.add_compact_dataset(
    "/objects",
    [2UL],
    StructuredValues(object_type, [
      ObjectReference(Some("/target")),
      ObjectReference(None),
    ]),
  )
  writer.add_attribute(
    "/objects",
    "refs",
    [1UL],
    StructuredValues(object_type, [ObjectReference(None)]),
  )
  writer.add_chunked_dataset(
    "/regions",
    [5UL],
    [2U],
    StructuredValues(region_type, [
      RegionReference(Some("/target"), RegionPoints([[0UL]])),
      RegionReference(None, RegionEmpty),
      RegionReference(Some("/target"), RegionAll),
      RegionReference(None, RegionEmpty),
      RegionReference(Some("/target"), RegionPoints([[4UL]])),
    ]),
    filters=[Shuffle, Deflate(6)],
  )
  let file = Hdf5File::parse(writer.encode())
  let patched = file.patch_dataset_values(
    "/objects",
    ObjectReferenceValues([Some("/group"), Some("/target")]),
  )
  let patched_file = Hdf5File::parse(patched)
  assert_eq(
    patched_file.read_dataset_metadata("/objects").layout.kind(),
    Compact,
  )
  let objects = reference_value_bytes(patched_file.read_dataset("/objects"))
  let group = patched_file.resolve("/group")
  let target = patched_file.resolve("/target")
  assert_eq(
    patched_file.resolve_object_reference(objects[0]),
    Some({ address: group.address, kind: Group, path: Some("/group"), }),
  )
  assert_eq(
    patched_file.resolve_object_reference(objects[1]),
    Some({ address: target.address, kind: Dataset, path: Some("/target"), }),
  )

  let attribute_metadata = patched_file.read_dataset_metadata("/objects").datatype
  let attribute_patched = patched_file.patch_attribute_values(
    "/objects",
    "refs",
    ObjectReferenceValues([Some("/target")]),
  )
  let attribute_file = Hdf5File::parse(attribute_patched)
  match find_attribute(attribute_file.attributes("/objects"), "refs").values {
    Some(Structured([Reference(reference)])) =>
      assert_eq(
        attribute_file.resolve_object_reference(reference),
        Some({ address: target.address, kind: Dataset, path: Some("/target"), }),
      )
    _ => fail("expected a patched compact object-reference attribute")
  }
  assert_eq(attribute_metadata.reference_kind, 2)

  let region_metadata = attribute_file.read_dataset_metadata("/regions").datatype
  let region_patched = attribute_file.patch_dataset_values(
    "/regions",
    StructuredValues(region_metadata, [
      RegionReference(Some("/target"), RegionAll),
      RegionReference(Some("/target"), RegionPoints([[1UL], [3UL]])),
      RegionReference(None, RegionEmpty),
      RegionReference(Some("/target"), RegionPoints([[4UL]])),
      RegionReference(None, RegionEmpty),
    ]),
  )
  let region_file = Hdf5File::parse(region_patched)
  assert_eq(
    region_file.read_dataset_metadata("/regions").layout.kind(),
    Chunked,
  )
  let regions = reference_value_bytes(region_file.read_dataset("/regions"))
  assert_eq(
    patched_region_selection(region_file, regions[0]),
    (Some("/target"), RegionAll),
  )
  assert_eq(
    patched_region_selection(region_file, regions[1]),
    (Some("/target"), RegionPoints([[1UL], [3UL]])),
  )
  assert_true(region_file.resolve_region_reference(regions[2]) is None)
  assert_eq(
    patched_region_selection(region_file, regions[3]),
    (Some("/target"), RegionPoints([[4UL]])),
  )
  assert_true(region_file.resolve_region_reference(regions[4]) is None)
}

///|
test "version-4 region and attribute reference patching validates targets" {
  let writer = Hdf5Writer::new()
  writer.add_group("/group")
  writer.add_dataset("/target", [2UL], Int64Values([1L, 2L]))
  writer.add_attribute("/target", "unit", [], Int64Values([7L]))
  let region_type = Datatype::write_version4_region_reference()
  let attribute_type = Datatype::write_version4_attribute_reference()
  writer.add_dataset(
    "/regions",
    [1UL],
    StructuredValues(region_type, [RegionReference(Some("/target"), RegionAll)]),
  )
  writer.add_dataset(
    "/attributes",
    [1UL],
    StructuredValues(attribute_type, [
      AttributeReference(Some("/target"), "unit"),
    ]),
  )
  let file = Hdf5File::parse(writer.encode())
  let non_dataset = try {
    file.patch_dataset_values(
      "/regions",
      StructuredValues(region_type, [RegionReference(Some("/group"), RegionAll)]),
    )
    |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    non_dataset,
    Some(InvalidArgument("region reference target is not a dataset: /group")),
  )
  let missing_attribute = try {
    file.patch_dataset_values(
      "/attributes",
      StructuredValues(attribute_type, [
        AttributeReference(Some("/target"), "missing"),
      ]),
    )
    |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    missing_attribute,
    Some(
      InvalidArgument(
        "attribute reference attribute does not exist: /target.missing",
      ),
    ),
  )
}
