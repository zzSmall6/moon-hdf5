///|
fn patched_region_selection(
  file : Hdf5File,
  reference : Bytes,
) -> (String?, Hdf5RegionSelection) raise Error {
  let resolved = file.resolve_region_reference(reference) catch {
    error =>
      fail("failed to resolve patched region reference: " + error.message())
  }
  let target = match resolved {
    Some(target) => target
    None => fail("expected a non-null region reference")
  }
  (target.path, target.selection)
}

///|
test "legacy region reference datasets and attributes can be patched" {
  let region_type = Datatype::write_region_reference()
  let writer = Hdf5Writer::new()
  writer.add_dataset("/source", [6UL], Int64Values([0L, 1L, 2L, 3L, 4L, 5L]))
  writer.add_dataset(
    "/matrix",
    [2UL, 3UL],
    Int64Values([0L, 1L, 2L, 3L, 4L, 5L]),
  )
  writer.add_dataset(
    "/regions",
    [3UL],
    StructuredValues(region_type, [
      RegionReference(None, RegionEmpty),
      RegionReference(Some("/source"), RegionPoints([[1UL], [4UL]])),
      RegionReference(
        Some("/source"),
        RegionHyperslab([1UL], [1UL], [2UL], [2UL]),
      ),
    ]),
  )
  writer.add_attribute(
    "/regions",
    "template",
    [2UL],
    StructuredValues(region_type, [
      RegionReference(Some("/source"), RegionAll),
      RegionReference(None, RegionEmpty),
    ]),
  )
  let file = Hdf5File::parse(writer.encode())
  let patched = file.patch_dataset_values(
    "/regions",
    StructuredValues(region_type, [
      RegionReference(
        Some("/matrix"),
        RegionHyperslab([0UL, 0UL], [1UL, 1UL], [2UL, 2UL], [1UL, 1UL]),
      ),
      RegionReference(None, RegionEmpty),
      RegionReference(
        Some("/source"),
        RegionBlocks([
          { start: [0UL], end: [1UL], },
          { start: [5UL], end: [5UL], },
        ]),
      ),
    ]),
  )
  let patched_file = Hdf5File::parse(patched)
  let references = reference_value_bytes(patched_file.read_dataset("/regions"))
  assert_eq(references.length(), 3)
  assert_eq(references[0].length(), 12)
  assert_eq(
    patched_region_selection(patched_file, references[0]),
    (
      Some("/matrix"),
      RegionHyperslab([0UL, 0UL], [1UL, 1UL], [2UL, 2UL], [1UL, 1UL]),
    ),
  )
  assert_true(patched_file.resolve_region_reference(references[1]) is None)
  assert_eq(
    patched_region_selection(patched_file, references[2]),
    (
      Some("/source"),
      RegionBlocks([
        { start: [0UL], end: [1UL], },
        { start: [5UL], end: [5UL], },
      ]),
    ),
  )
  assert_eq(integer_values(patched_file.read_dataset("/source")), [
    0L, 1L, 2L, 3L, 4L, 5L,
  ])

  let attribute_patched = patched_file.patch_attribute_values(
    "/regions",
    "template",
    StructuredValues(region_type, [
      RegionReference(Some("/matrix"), RegionAll),
      RegionReference(Some("/source"), RegionPoints([[2UL]])),
    ]),
  )
  let attribute_file = Hdf5File::parse(attribute_patched)
  match
    find_attribute(attribute_file.attributes("/regions"), "template").values {
    Some(Structured([Reference(first), Reference(second)])) => {
      assert_eq(
        patched_region_selection(attribute_file, first),
        (Some("/matrix"), RegionAll),
      )
      assert_eq(
        patched_region_selection(attribute_file, second),
        (Some("/source"), RegionPoints([[2UL]])),
      )
    }
    _ => fail("expected patched region-reference attribute values")
  }

  let repatched = attribute_file.patch_dataset_values(
    "/regions",
    StructuredValues(region_type, [
      RegionReference(None, RegionEmpty),
      RegionReference(None, RegionEmpty),
      RegionReference(Some("/source"), RegionAll),
    ]),
  )
  let repatched_file = Hdf5File::parse(repatched)
  let repatched_refs = reference_value_bytes(
    repatched_file.read_dataset("/regions"),
  )
  assert_true(
    repatched_file.resolve_region_reference(repatched_refs[0]) is None,
  )
  assert_true(
    repatched_file.resolve_region_reference(repatched_refs[1]) is None,
  )
  assert_eq(
    patched_region_selection(repatched_file, repatched_refs[2]),
    (Some("/source"), RegionAll),
  )
}

///|
test "filtered chunked legacy region references can be patched" {
  let region_type = Datatype::write_region_reference()
  let writer = Hdf5Writer::new()
  writer.add_dataset("/source", [5UL], Int64Values([0L, 1L, 2L, 3L, 4L]))
  writer.add_chunked_dataset(
    "/regions",
    [3UL],
    [2U],
    StructuredValues(region_type, [
      RegionReference(Some("/source"), RegionPoints([[0UL]])),
      RegionReference(None, RegionEmpty),
      RegionReference(Some("/source"), RegionAll),
    ]),
    filters=[Shuffle, Deflate(6)],
  )
  let file = Hdf5File::parse(writer.encode())
  let patched = file.patch_dataset_values(
    "/regions",
    StructuredValues(region_type, [
      RegionReference(Some("/source"), RegionAll),
      RegionReference(Some("/source"), RegionPoints([[4UL], [1UL]])),
      RegionReference(None, RegionEmpty),
    ]),
  )
  let patched_file = Hdf5File::parse(patched)
  assert_eq(
    patched_file.read_dataset_metadata("/regions").layout.kind(),
    Chunked,
  )
  let references = reference_value_bytes(patched_file.read_dataset("/regions"))
  assert_eq(
    patched_region_selection(patched_file, references[0]),
    (Some("/source"), RegionAll),
  )
  assert_eq(
    patched_region_selection(patched_file, references[1]),
    (Some("/source"), RegionPoints([[4UL], [1UL]])),
  )
  assert_true(patched_file.resolve_region_reference(references[2]) is None)
}

///|
test "four-byte legacy region references can be patched" {
  let file = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/references_layouts_32.h5"),
  )
  assert_eq(file.superblock.offset_size, 4)
  let datatype = file.read_dataset_metadata("/ref_compact").datatype
  assert_eq(datatype.reference_kind, 1)
  assert_eq(datatype.size, 8)
  let patched = file.patch_dataset_values(
    "/ref_compact",
    StructuredValues(datatype, [RegionReference(Some("/chunked"), RegionAll)]),
  )
  let patched_file = Hdf5File::parse(patched)
  let references = reference_value_bytes(
    patched_file.read_dataset("/ref_compact"),
  )
  assert_eq(references[0].length(), 8)
  assert_eq(
    patched_region_selection(patched_file, references[0]),
    (Some("/chunked"), RegionAll),
  )
}

///|
test "region patching rejects non-dataset targets" {
  let region_type = Datatype::write_region_reference()
  let writer = Hdf5Writer::new()
  writer.add_group("/group")
  writer.add_dataset("/source", [2UL], Int64Values([1L, 2L]))
  writer.add_dataset(
    "/regions",
    [1UL],
    StructuredValues(region_type, [RegionReference(Some("/source"), RegionAll)]),
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
}
