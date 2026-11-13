///|
test "legacy object reference datasets can be patched repeatedly" {
  let writer = Hdf5Writer::new()
  writer.add_group("/targets")
  writer.add_dataset("/targets/first", [1UL], Int64Values([11L]))
  writer.add_dataset("/targets/second", [2UL], Int64Values([21L, 22L]))
  writer.add_committed_datatype(
    "/named_integer",
    Datatype::write_integer(true, 4),
  )
  writer.add_dataset(
    "/refs",
    [3UL],
    ObjectReferenceValues([Some("/targets/first"), None, Some("/targets")]),
  )
  let file = Hdf5File::parse(writer.encode())
  let patched = file.patch_dataset_values(
    "/refs",
    ObjectReferenceValues([
      Some("/targets/second"),
      Some("/targets/first"),
      None,
    ]),
  )
  let patched_file = Hdf5File::parse(patched)
  let references = reference_value_bytes(patched_file.read_dataset("/refs"))
  assert_eq(references.length(), 3)
  assert_eq(references[0].length(), patched_file.superblock.offset_size)
  let first = patched_file.resolve("/targets/first")
  let second = patched_file.resolve("/targets/second")
  assert_eq(
    patched_file.resolve_object_reference(references[0]),
    Some({
      address: second.address,
      kind: Dataset,
      path: Some("/targets/second"),
    }),
  )
  assert_eq(
    patched_file.resolve_object_reference(references[1]),
    Some({
      address: first.address,
      kind: Dataset,
      path: Some("/targets/first"),
    }),
  )
  assert_eq(patched_file.resolve_object_reference(references[2]), None)
  assert_eq(integer_values(patched_file.read_dataset("/targets/second")), [
    21L, 22L,
  ])

  let repatched = patched_file.patch_dataset_values(
    "/refs",
    ObjectReferenceValues([None, Some("/targets"), Some("/named_integer")]),
  )
  let repatched_file = Hdf5File::parse(repatched)
  let repatched_refs = reference_value_bytes(
    repatched_file.read_dataset("/refs"),
  )
  assert_eq(repatched_file.resolve_object_reference(repatched_refs[0]), None)
  let targets = repatched_file.resolve("/targets")
  assert_eq(
    repatched_file.resolve_object_reference(repatched_refs[1]),
    Some({ address: targets.address, kind: Group, path: Some("/targets"), }),
  )
  let named_integer = repatched_file.resolve("/named_integer")
  assert_eq(
    repatched_file.resolve_object_reference(repatched_refs[2]),
    Some({
      address: named_integer.address,
      kind: Datatype,
      path: Some("/named_integer"),
    }),
  )

  let missing_target = try {
    patched_file.patch_dataset_values(
      "/refs",
      ObjectReferenceValues([Some("/missing"), None, None]),
    )
    |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_true(missing_target is Some(InvalidField(_)))
}

///|
test "legacy object reference attributes can be patched" {
  let writer = Hdf5Writer::new()
  writer.add_dataset("/source", [1UL], Int64Values([1L]))
  writer.add_dataset("/destination", [1UL], Int64Values([2L]))
  writer.add_group("/group")
  writer.add_attribute(
    "/source",
    "links",
    [2UL],
    ObjectReferenceValues([Some("/destination"), None]),
  )
  writer.add_attribute("/source", "keep", [], Int64Values([3L]))
  let file = Hdf5File::parse(writer.encode())
  let patched = file.patch_attribute_values(
    "/source",
    "links",
    ObjectReferenceValues([Some("/group"), Some("/destination")]),
  )
  let patched_file = Hdf5File::parse(patched)
  let attributes = patched_file.attributes("/source")
  let destination = patched_file.resolve("/destination")
  let group = patched_file.resolve("/group")
  match find_attribute(attributes, "links").values {
    Some(Structured([Reference(first), Reference(second)])) => {
      assert_eq(first.length(), patched_file.superblock.offset_size)
      assert_eq(
        patched_file.resolve_object_reference(first),
        Some({ address: group.address, kind: Group, path: Some("/group"), }),
      )
      assert_eq(
        patched_file.resolve_object_reference(second),
        Some({
          address: destination.address,
          kind: Dataset,
          path: Some("/destination"),
        }),
      )
    }
    _ => fail("expected patched object-reference attribute values")
  }
  match find_attribute(attributes, "keep").values {
    Some(Integers(values)) => assert_eq(values, [3L])
    _ => fail("expected the unrelated attribute to remain unchanged")
  }
}

///|
test "chunked and filtered legacy object references can be patched" {
  let writer = Hdf5Writer::new()
  writer.add_group("/group")
  writer.add_dataset("/first", [1UL], Int64Values([1L]))
  writer.add_dataset("/second", [1UL], Int64Values([2L]))
  writer.add_chunked_dataset(
    "/plain_refs",
    [3UL],
    [2U],
    ObjectReferenceValues([Some("/first"), None, Some("/group")]),
  )
  writer.add_chunked_dataset(
    "/filtered_refs",
    [3UL],
    [2U],
    ObjectReferenceValues([Some("/first"), None, Some("/group")]),
    filters=[Shuffle, Deflate(6)],
  )
  let file = Hdf5File::parse(writer.encode())
  let patched = file.patch_dataset_values(
    "/plain_refs",
    ObjectReferenceValues([Some("/second"), Some("/group"), None]),
  )
  let patched_file = Hdf5File::parse(patched)
  assert_eq(
    patched_file.read_dataset_metadata("/plain_refs").layout.kind(),
    Chunked,
  )
  let plain = reference_value_bytes(patched_file.read_dataset("/plain_refs"))
  let second = patched_file.resolve("/second")
  let group = patched_file.resolve("/group")
  assert_eq(
    patched_file.resolve_object_reference(plain[0]),
    Some({ address: second.address, kind: Dataset, path: Some("/second"), }),
  )
  assert_eq(
    patched_file.resolve_object_reference(plain[1]),
    Some({ address: group.address, kind: Group, path: Some("/group"), }),
  )
  assert_eq(patched_file.resolve_object_reference(plain[2]), None)

  let filtered = patched_file.patch_dataset_values(
    "/filtered_refs",
    ObjectReferenceValues([Some("/group"), None, Some("/second")]),
  )
  let filtered_file = Hdf5File::parse(filtered)
  assert_eq(
    filtered_file.read_dataset_metadata("/filtered_refs").layout.kind(),
    Chunked,
  )
  let filtered_refs = reference_value_bytes(
    filtered_file.read_dataset("/filtered_refs"),
  )
  let filtered_group = filtered_file.resolve("/group")
  let filtered_second = filtered_file.resolve("/second")
  assert_eq(
    filtered_file.resolve_object_reference(filtered_refs[0]),
    Some({ address: filtered_group.address, kind: Group, path: Some("/group"), }),
  )
  assert_eq(filtered_file.resolve_object_reference(filtered_refs[1]), None)
  assert_eq(
    filtered_file.resolve_object_reference(filtered_refs[2]),
    Some({
      address: filtered_second.address,
      kind: Dataset,
      path: Some("/second"),
    }),
  )
  assert_eq(integer_values(filtered_file.read_dataset("/first")), [1L])
}

///|
test "compact legacy object references can be patched" {
  let writer = Hdf5Writer::new()
  writer.add_group("/group")
  writer.add_dataset("/target", [1UL], Int64Values([9L]))
  writer.add_compact_dataset(
    "/compact_refs",
    [2UL],
    ObjectReferenceValues([Some("/group"), None]),
  )
  let file = Hdf5File::parse(writer.encode())
  assert_eq(file.read_dataset_metadata("/compact_refs").layout.kind(), Compact)
  let patched = file.patch_dataset_values(
    "/compact_refs",
    ObjectReferenceValues([None, Some("/target")]),
  )
  let patched_file = Hdf5File::parse(patched)
  assert_eq(
    patched_file.read_dataset_metadata("/compact_refs").layout.kind(),
    Compact,
  )
  let references = reference_value_bytes(
    patched_file.read_dataset("/compact_refs"),
  )
  let target = patched_file.resolve("/target")
  assert_eq(patched_file.resolve_object_reference(references[0]), None)
  assert_eq(
    patched_file.resolve_object_reference(references[1]),
    Some({ address: target.address, kind: Dataset, path: Some("/target"), }),
  )
}

///|
test "four-byte legacy object reference datasets and attributes can be patched" {
  let file = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/references_layouts_32.h5"),
  )
  assert_eq(file.superblock.offset_size, 4)
  assert_eq(
    reference_value_bytes(file.read_dataset("/object_ref_group"))[0].length(),
    4,
  )
  let patched = file.patch_dataset_values(
    "/object_ref_group",
    ObjectReferenceValues([Some("/chunked")]),
  )
  let patched_file = Hdf5File::parse(patched)
  let references = reference_value_bytes(
    patched_file.read_dataset("/object_ref_group"),
  )
  assert_eq(references[0].length(), 4)
  let chunked = patched_file.resolve("/chunked")
  assert_eq(
    patched_file.resolve_object_reference(references[0]),
    Some({ address: chunked.address, kind: Dataset, path: Some("/chunked"), }),
  )

  let attribute_patched = patched_file.patch_attribute_values(
    "/compact",
    "sibling_ref",
    ObjectReferenceValues([Some("/group")]),
  )
  let attribute_file = Hdf5File::parse(attribute_patched)
  let group = attribute_file.resolve("/group")
  match
    find_attribute(attribute_file.attributes("/compact"), "sibling_ref").values {
    Some(Structured([Reference(bytes)])) => {
      assert_eq(bytes.length(), 4)
      assert_eq(
        attribute_file.resolve_object_reference(bytes),
        Some({ address: group.address, kind: Group, path: Some("/group"), }),
      )
    }
    _ => fail("expected a patched four-byte object-reference attribute")
  }
}
