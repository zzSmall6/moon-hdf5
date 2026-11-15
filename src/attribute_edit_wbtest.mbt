///|
fn attribute_edit_error(operation : () -> Unit raise Hdf5Error) -> Hdf5Error? {
  try {
    operation()
    None
  } catch {
    error => Some(error)
  }
}

///|
fn attribute_edit_dense_records(
  file : Hdf5File,
  path : String,
) -> Array[MetadataBTreeRecord] raise Hdf5Error {
  let node = file.resolve(path)
  let header = file.object_header_at(node.address)
  let info = match header.first(msg_attribute_info) {
    Some(message) => message
    None => raise InvalidField("expected an attribute info message")
  }
  let reader = Reader::new(info.data)
  guard reader.read_u8() == b'\x00' else {
    raise InvalidField("expected attribute info version zero")
  }
  let flags = reader.read_u8().to_int()
  if (flags & 1) != 0 {
    reader.skip(2)
  }
  let offset_size = file.superblock.offset_size
  let heap_address = reader.read_uint_le(offset_size)
  let name_index_address = reader.read_uint_le(offset_size)
  let undefined = heap_undefined_address(offset_size)
  let order_index_address = if (flags & 2) != 0 {
    reader.read_uint_le(offset_size)
  } else {
    undefined
  }
  guard heap_address != undefined else {
    raise InvalidField("expected dense attribute storage")
  }
  let index_address = if (flags & 2) != 0 {
    order_index_address
  } else {
    name_index_address
  }
  guard index_address != undefined else {
    raise InvalidField("expected a dense attribute index")
  }
  read_metadata_btree_records(
    file.bytes,
    index_address,
    offset_size,
    file.superblock.length_size,
    file.superblock.base_address,
    [8, 9],
  )
}

///|
test "attributes can be added renamed and deleted on groups and datasets" {
  let writer = Hdf5Writer::new()
  writer.add_group("/group")
  writer.add_dataset("/group/data", [2UL], Int32Values([1, 2]))
  writer.add_attribute("/group", "existing", [], UInt64Values([7UL]))
  let file = Hdf5File::parse(writer.encode())

  let added = file.add_attribute(
    "/group/data",
    "units",
    [],
    FixedStringValues(["items"], 8),
  )
  let with_units = Hdf5File::parse(added)
  let attributes = with_units.attributes("/group/data")
  assert_eq(attributes.length(), 1)
  assert_eq(attributes[0].name, "units")
  match attributes[0].values {
    Some(Strings(values)) => assert_eq(values, ["items"])
    _ => fail("expected added string attribute")
  }
  assert_eq(integer_values(with_units.read_dataset("/group/data")), [1L, 2L])
  match with_units.attributes("/group")[0].values {
    Some(UnsignedIntegers(values)) => assert_eq(values, [7UL])
    _ => fail("expected unchanged group attribute")
  }

  let renamed = with_units.rename_attribute("/group/data", "units", "labels")
  let renamed_file = Hdf5File::parse(renamed)
  assert_eq(renamed_file.attributes("/group/data").map(a => a.name), ["labels"])
  match renamed_file.attributes("/group/data")[0].values {
    Some(Strings(values)) => assert_eq(values, ["items"])
    _ => fail("expected renamed attribute value")
  }

  let deleted = renamed_file.delete_attribute("/group/data", "labels")
  let deleted_file = Hdf5File::parse(deleted)
  assert_eq(deleted_file.attributes("/group/data").length(), 0)
  assert_eq(integer_values(deleted_file.read_dataset("/group/data")), [1L, 2L])
  match deleted_file.attributes("/group")[0].values {
    Some(UnsignedIntegers(values)) => assert_eq(values, [7UL])
    _ => fail("expected group attribute after delete")
  }
}

///|
test "adding variable-length strings appends a global heap" {
  let writer = Hdf5Writer::new()
  writer.add_dataset("/values", [2UL], Int32Values([3, 4]))
  let file = Hdf5File::parse(writer.encode())
  let added = file.add_attribute(
    "/values",
    "names",
    [2UL],
    VariableStringValues(["moon", "数据"]),
  )
  let added_file = Hdf5File::parse(added)
  match added_file.attributes("/values")[0].values {
    Some(Strings(values)) => assert_eq(values, ["moon", "数据"])
    _ => fail("expected variable-length string attribute")
  }
  assert_eq(integer_values(added_file.read_dataset("/values")), [3L, 4L])
}

///|
test "adding variable-length sequence attributes appends a global heap" {
  let writer = Hdf5Writer::new()
  writer.add_dataset("/values", [1UL], Int32Values([3]))
  let file = Hdf5File::parse(writer.encode())
  let sequence = Datatype::write_vlen_sequence(Datatype::write_integer(true, 4))
  let added = file.add_attribute(
    "/values",
    "history",
    [2UL],
    StructuredValues(sequence, [
      Sequence([SignedInteger(1L), SignedInteger(-2L)]),
      Sequence([]),
    ]),
  )
  let added_file = Hdf5File::parse(added)
  match find_attribute(added_file.attributes("/values"), "history").values {
    Some(Structured(values)) =>
      assert_eq(values, [
        Sequence([SignedInteger(1L), SignedInteger(-2L)]),
        Sequence([]),
      ])
    _ => fail("expected variable-length sequence attribute")
  }
  assert_eq(integer_values(added_file.read_dataset("/values")), [3L])
}

///|
test "adding nested variable-length attribute members appends one heap" {
  let writer = Hdf5Writer::new()
  writer.add_dataset("/values", [1UL], Int32Values([3]))
  let file = Hdf5File::parse(writer.encode())
  let text_type = Datatype::write_vlen_string()
  let sequence_type = Datatype::write_vlen_sequence(
    Datatype::write_integer(true, 4),
  )
  let aliases_type = Datatype::write_array(text_type, [2UL])
  let record_type = Datatype::write_compound([
    ("label", text_type),
    ("history", sequence_type),
    ("aliases", aliases_type),
  ])
  let first = Compound([
    { name: "label", value: Text("first"), },
    {
      name: "history",
      value: Sequence([SignedInteger(1L), SignedInteger(-2L)]),
    },
    { name: "aliases", value: Elements([Text("甲"), Text("")]), },
  ])
  let second = Compound([
    { name: "label", value: Text("第二个"), },
    { name: "history", value: Sequence([]), },
    { name: "aliases", value: Elements([Text(""), Text("alias")]), },
  ])
  let added = file.add_attribute(
    "/values",
    "records",
    [2UL],
    StructuredValues(record_type, [first, second]),
  ) catch {
    error => fail("add nested vlen attributes: " + error.message())
  }
  let added_file = Hdf5File::parse(added)
  match find_attribute(added_file.attributes("/values"), "records").values {
    Some(Structured(values)) => assert_eq(values, [first, second])
    _ => fail("expected nested variable-length attribute values")
  }
  assert_eq(integer_values(added_file.read_dataset("/values")), [3L])
}

///|
test "adding region references in variable-length sequence attributes" {
  let writer = Hdf5Writer::new()
  writer.add_dataset("/source", [6UL], Int64Values([0L, 1L, 2L, 3L, 4L, 5L]))
  let file = Hdf5File::parse(writer.encode())
  let legacy_sequence = Datatype::write_vlen_sequence(
    Datatype::write_region_reference(),
  )
  let legacy_values = [
    Sequence([
      RegionReference(Some("/source"), RegionPoints([[1UL], [4UL]])),
      RegionReference(Some("/source"), RegionAll),
    ]),
    Sequence([]),
  ]
  let added = file.add_attribute(
    "/source",
    "legacy_regions",
    [2UL],
    StructuredValues(legacy_sequence, legacy_values),
  ) catch {
    error => fail("add legacy region sequence attribute: " + error.message())
  }
  let legacy_file = Hdf5File::parse(added)
  match
    find_attribute(legacy_file.attributes("/source"), "legacy_regions").values {
    Some(
      Structured([Sequence([Reference(points), Reference(all)]), Sequence([])])
    ) => {
      assert_eq(writer_region_integer_values(legacy_file, points), [1L, 4L])
      assert_eq(writer_region_integer_values(legacy_file, all), [
        0L, 1L, 2L, 3L, 4L, 5L,
      ])
    }
    _ => fail("expected legacy region-reference sequence attribute")
  }

  let version4_sequence = Datatype::write_vlen_sequence(
    Datatype::write_version4_region_reference(),
  )
  let version4_added = legacy_file.add_attribute(
    "/source",
    "version4_regions",
    [1UL],
    StructuredValues(version4_sequence, [
      Sequence([
        RegionReference(
          Some("/source"),
          RegionHyperslab([1UL], [2UL], [2UL], [1UL]),
        ),
        RegionReference(None, RegionEmpty),
      ]),
    ]),
  ) catch {
    error => fail("add version-4 region sequence attribute: " + error.message())
  }
  let version4_file = Hdf5File::parse(version4_added)
  match
    find_attribute(version4_file.attributes("/source"), "version4_regions").values {
    Some(Structured([Sequence([Reference(slab), Reference(null_ref)])])) => {
      assert_eq(writer_region_integer_values(version4_file, slab), [1L, 3L])
      assert_true(version4_file.resolve_region_reference(null_ref) is None)
    }
    _ => fail("expected version-4 region-reference sequence attribute")
  }
}

///|
test "attribute storage converts between compact and dense forms" {
  let writer = Hdf5Writer::new()
  writer.add_group("/dense")
  for index in 0..<8 {
    writer.add_attribute(
      "/dense",
      "attr_0" + index.to_string(),
      [],
      Int64Values([index.to_int64()]),
    )
  }
  let file = Hdf5File::parse(writer.encode())
  assert_eq(file.attributes("/dense").length(), 8)

  let dense_bytes = file.add_attribute(
    "/dense",
    "attr_08",
    [],
    Int64Values([8L]),
  )
  let dense_file = Hdf5File::parse(dense_bytes)
  let dense = dense_file.attributes("/dense")
  assert_eq(dense.length(), 9)
  assert_eq(dense.map(attribute => attribute.name), [
    "attr_00", "attr_01", "attr_02", "attr_03", "attr_04", "attr_05", "attr_06",
    "attr_07", "attr_08",
  ])
  for index in 0..<dense.length() {
    assert_eq(
      values_integers(
        find_attribute(dense, "attr_0" + index.to_string()).values.unwrap(),
      ),
      [index.to_int64()],
    )
  }

  let compact_bytes = dense_file.delete_attribute("/dense", "attr_00")
  let compact_file = Hdf5File::parse(compact_bytes)
  let compact = compact_file.attributes("/dense")
  assert_eq(compact.length(), 8)
  assert_eq(compact.map(attribute => attribute.name), [
    "attr_01", "attr_02", "attr_03", "attr_04", "attr_05", "attr_06", "attr_07",
    "attr_08",
  ])

  let dense_again = compact_file.add_attribute(
    "/dense",
    "attr_00",
    [],
    Int64Values([100L]),
  )
  let dense_again_file = Hdf5File::parse(dense_again)
  let attributes = dense_again_file.attributes("/dense")
  assert_eq(attributes.length(), 9)
  assert_eq(attributes.map(attribute => attribute.name), [
    "attr_01", "attr_02", "attr_03", "attr_04", "attr_05", "attr_06", "attr_07",
    "attr_08", "attr_00",
  ])
  assert_eq(
    values_integers(find_attribute(attributes, "attr_00").values.unwrap()),
    [100L],
  )
}

///|
test "dense attribute rebuilding spans heap blocks and huge objects" {
  let writer = Hdf5Writer::new()
  let value = "r".repeat(60000)
  writer.add_group("/rebuild")
  for index in 0..<8 {
    writer.add_attribute(
      "/rebuild",
      "small_" + index.to_string(),
      [],
      UInt64Values([index.to_uint64()]),
    )
  }
  let first = Hdf5File::parse(writer.encode()).add_attribute(
    "/rebuild",
    "medium_0",
    [],
    FixedStringValues([value], 60000),
  )
  let second = Hdf5File::parse(first).add_attribute(
    "/rebuild",
    "medium_1",
    [],
    FixedStringValues([value], 60000),
  )
  let third = Hdf5File::parse(second).add_attribute(
    "/rebuild",
    "huge",
    [2UL],
    FixedStringValues([value, value], 60000),
  )

  let file = Hdf5File::parse(third)
  let attributes = file.attributes("/rebuild")
  assert_eq(attributes.length(), 11)
  for name in ["medium_0", "medium_1"] {
    match find_attribute(attributes, name).values {
      Some(Strings(values)) => assert_eq(values, [value])
      _ => fail("expected a rebuilt medium attribute")
    }
  }
  match find_attribute(attributes, "huge").values {
    Some(Strings(values)) => assert_eq(values, [value, value])
    _ => fail("expected a rebuilt huge attribute")
  }
  assert_eq(attribute_edit_dense_records(file, "/rebuild").length(), 11)
}

///|
test "renaming a dense ordered attribute preserves creation order" {
  let writer = Hdf5Writer::new()
  writer.add_group("/ordered")
  for index in 0..<9 {
    writer.add_attribute(
      "/ordered",
      "attr_0" + index.to_string(),
      [2UL],
      Int64Values([index.to_int64(), index.to_int64() + 100L]),
    )
  }
  let file = Hdf5File::parse(writer.encode())
  let renamed = file.rename_attribute("/ordered", "attr_04", "renamed")
  let renamed_file = Hdf5File::parse(renamed)
  let attributes = renamed_file.attributes("/ordered")
  assert_eq(attributes.length(), 9)
  assert_eq(attributes[4].name, "renamed")
  assert_eq(
    values_integers(find_attribute(attributes, "renamed").values.unwrap()),
    [4L, 104L],
  )
  assert_eq(attributes[8].name, "attr_08")
  assert_eq(renamed_file.attributes("/ordered")[4].name, "renamed")
}

///|
test "dense attributes from h5py fixtures can be edited and rebuilt" {
  let file = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/attribute_storage.h5"),
  )
  let renamed = file.rename_attribute("/dense", "attr_00", "renamed")
  let renamed_file = Hdf5File::parse(renamed)
  let attributes = renamed_file.attributes("/dense")
  assert_eq(attributes.length(), 12)
  assert_eq(attributes.map(attribute => attribute.name), [
    "renamed", "attr_01", "attr_02", "attr_03", "attr_04", "attr_05", "attr_06",
    "attr_07", "attr_08", "attr_09", "attr_10", "attr_11",
  ])
  assert_eq(
    values_integers(find_attribute(attributes, "attr_07").values.unwrap()),
    [7L],
  )

  let added = renamed_file.add_attribute(
    "/dense_ordered",
    "added",
    [],
    Int64Values([99L]),
  )
  let added_file = Hdf5File::parse(added)
  let ordered = added_file.attributes("/dense_ordered")
  assert_eq(ordered.length(), 13)
  assert_eq(ordered.map(attribute => attribute.name), [
    "attr_00", "attr_01", "attr_02", "attr_03", "attr_04", "attr_05", "attr_06",
    "attr_07", "attr_08", "attr_09", "attr_10", "attr_11", "added",
  ])
  assert_eq(values_integers(find_attribute(ordered, "added").values.unwrap()), [
    99L,
  ])

  let deleted = added_file.delete_attribute("/dense_ordered", "attr_10")
  let deleted_file = Hdf5File::parse(deleted)
  assert_eq(deleted_file.attributes("/dense_ordered").length(), 12)
  assert_eq(
    values_integers(
      find_attribute(deleted_file.attributes("/dense_ordered"), "attr_11").values.unwrap(),
    ),
    [11L],
  )
}

///|
test "dense attributes rebuild with four-byte addresses" {
  let bytes = @fs.read_file_to_bytes("testdata/interop_32.h5")
  let mut edited = bytes
  for index in 0..<9 {
    let file = Hdf5File::parse(edited) catch {
      error => fail("parse " + index.to_string() + ": " + error.message())
    }
    edited = file.add_attribute(
      "/raw",
      "attr_0" + index.to_string(),
      [],
      Int64Values([index.to_int64()]),
    ) catch {
      error => fail("add " + index.to_string() + ": " + error.message())
    }
  }
  let edited_file = Hdf5File::parse(edited) catch {
    error => fail("parse edited: " + error.message())
  }
  assert_eq(edited_file.superblock.offset_size, 4)
  let attributes = edited_file.attributes("/raw") catch {
    error => fail("attributes: " + error.message())
  }
  assert_eq(attributes.length(), 9)
  for index in 0..<9 {
    assert_eq(
      values_integers(
        find_attribute(attributes, "attr_0" + index.to_string()).values.unwrap(),
      ),
      [index.to_int64()],
    )
  }
  let records = attribute_edit_dense_records(edited_file, "/raw") catch {
    error => fail(error.message())
  }
  assert_eq(records.length(), 9)
  assert_true(
    !records.any(record => (record.flags.to_int() & shared_message_flag) != 0),
  )

  let deleted = edited_file.delete_attribute("/raw", "attr_04")
  let deleted_file = Hdf5File::parse(deleted)
  assert_eq(deleted_file.attributes("/raw").length(), 8)
  assert_eq(
    find_attribute(deleted_file.attributes("/raw"), "attr_08").name,
    "attr_08",
  )
}

///|
test "shared compact and dense attributes can be edited" {
  let file = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/shared_messages.h5"),
  )
  assert_eq(
    values_integers(
      find_attribute(file.attributes("/d1"), "common").values.unwrap(),
    ),
    [7L, 42L],
  )

  let compact = file.rename_attribute("/d1", "common", "common_renamed")
  let compact_file = Hdf5File::parse(compact)
  assert_eq(compact_file.attributes("/d1").map(attribute => attribute.name), [
    "common_renamed",
  ])
  assert_eq(
    values_integers(
      find_attribute(compact_file.attributes("/d1"), "common_renamed").values.unwrap(),
    ),
    [7L, 42L],
  )
  assert_eq(
    values_integers(
      find_attribute(compact_file.attributes("/d2"), "common").values.unwrap(),
    ),
    [7L, 42L],
  )

  let dense = compact_file.rename_attribute(
    "/dense0", "common", "common_renamed",
  )
  let dense_file = Hdf5File::parse(dense)
  let attributes = dense_file.attributes("/dense0")
  assert_eq(attributes.length(), 13)
  assert_eq(
    values_integers(
      find_attribute(attributes, "common_renamed").values.unwrap(),
    ),
    [
      7L, 8L, 9L, 10L, 11L, 12L, 13L, 14L, 15L, 16L, 17L, 18L, 19L, 20L, 21L, 22L,
    ],
  )
  assert_eq(
    values_integers(
      find_attribute(attributes, "dense0_unique00").values.unwrap(),
    ),
    [0L, 1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, 10L, 11L, 12L, 13L, 14L, 15L],
  )
  let records = attribute_edit_dense_records(dense_file, "/dense0")
  assert_eq(records.length(), 13)
  assert_true(
    !records.any(record => (record.flags.to_int() & shared_message_flag) != 0),
  )

  let deleted = dense_file.delete_attribute("/dense0", "dense0_unique00")
  let deleted_file = Hdf5File::parse(deleted)
  assert_eq(deleted_file.attributes("/dense0").length(), 12)
  assert_eq(
    values_integers(
      find_attribute(deleted_file.attributes("/dense1"), "common").values.unwrap(),
    ),
    [
      7L, 8L, 9L, 10L, 11L, 12L, 13L, 14L, 15L, 16L, 17L, 18L, 19L, 20L, 21L, 22L,
    ],
  )

  let added = deleted_file.add_attribute(
    "/dense0",
    "added",
    [],
    Int64Values([99L]),
  )
  let added_file = Hdf5File::parse(added)
  assert_eq(added_file.attributes("/dense0").length(), 13)
  assert_eq(
    values_integers(
      find_attribute(added_file.attributes("/dense0"), "added").values.unwrap(),
    ),
    [99L],
  )
}

///|
test "version-1 object headers support attribute add rename and delete" {
  let bytes = @fs.read_file_to_bytes("testdata/basic_v0.h5")
  let file = Hdf5File::parse(bytes)
  assert_eq(
    values_integers(
      find_attribute(file.attributes("/group"), "answer").values.unwrap(),
    ),
    [42L],
  )

  let added = file.add_attribute(
    "/group",
    "description",
    [],
    FixedStringValues(["v1 fixture"], 10),
  )
  let added_file = Hdf5File::parse(added)
  let added_attributes = added_file.attributes("/group")
  assert_eq(added_attributes.map(attribute => attribute.name), [
    "answer", "description",
  ])
  match find_attribute(added_attributes, "description").values {
    Some(Strings(values)) => assert_eq(values, ["v1 fixture"])
    _ => fail("expected added version-1 string attribute")
  }
  assert_eq(
    values_integers(find_attribute(added_attributes, "answer").values.unwrap()),
    [42L],
  )
  assert_eq(integer_values(added_file.read_dataset("/group/values")), [
    0L, 1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L,
  ])

  let renamed = added_file.rename_attribute(
    "/group", "answer", "answer_renamed",
  )
  let renamed_file = Hdf5File::parse(renamed)
  let renamed_attributes = renamed_file.attributes("/group")
  assert_eq(renamed_attributes.map(attribute => attribute.name), [
    "answer_renamed", "description",
  ])
  assert_eq(
    values_integers(
      find_attribute(renamed_attributes, "answer_renamed").values.unwrap(),
    ),
    [42L],
  )
  assert_eq(integer_values(renamed_file.read_dataset("/group/values")), [
    0L, 1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L,
  ])

  let deleted = renamed_file.delete_attribute("/group", "description")
  let deleted_file = Hdf5File::parse(deleted)
  let deleted_attributes = deleted_file.attributes("/group")
  assert_eq(deleted_attributes.map(attribute => attribute.name), [
    "answer_renamed",
  ])
  assert_eq(
    values_integers(
      find_attribute(deleted_attributes, "answer_renamed").values.unwrap(),
    ),
    [42L],
  )
  match deleted_file.attributes("/group/values")[0].values {
    Some(Strings(values)) => assert_eq(values, ["count"])
    _ => fail("expected unrelated dataset attribute to remain unchanged")
  }
  assert_eq(integer_values(deleted_file.read_dataset("/group/values")), [
    0L, 1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L,
  ])
}

///|
test "committed datatype attributes can be added renamed and deleted" {
  let writer = Hdf5Writer::new()
  writer.add_committed_datatype(
    "/types/value",
    Datatype::write_integer(true, 4),
  )
  let file = Hdf5File::parse(writer.encode())
  let added = file.add_attribute(
    "/types/value",
    "description",
    [],
    FixedStringValues(["integer"], 8),
  )
  let added_file = Hdf5File::parse(added)
  assert_eq(added_file.attributes("/types/value").map(a => a.name), [
    "description",
  ])
  let renamed = added_file.rename_attribute(
    "/types/value", "description", "label",
  )
  let renamed_file = Hdf5File::parse(renamed)
  assert_eq(renamed_file.attributes("/types/value").map(a => a.name), ["label"])
  let deleted = renamed_file.delete_attribute("/types/value", "label")
  let deleted_file = Hdf5File::parse(deleted)
  assert_eq(deleted_file.attributes("/types/value").length(), 0)
  assert_eq(deleted_file.resolve("/types/value").kind, Datatype)
}

///|
test "reference-valued attributes can be added to existing files" {
  let writer = Hdf5Writer::new()
  writer.add_group("/group")
  writer.add_dataset("/source", [1UL], Int64Values([5L]))
  writer.add_dataset("/target", [2UL, 2UL], Int64Values([0L, 1L, 2L, 3L]))
  writer.add_attribute("/target", "unit", [], FixedStringValues(["m"], 1))
  let file = Hdf5File::parse(writer.encode())

  let object_added = file.add_attribute(
    "/source",
    "objects",
    [2UL],
    ObjectReferenceValues([Some("/target"), None]),
  )
  let object_file = Hdf5File::parse(object_added)
  let target = object_file.resolve("/target")
  match find_attribute(object_file.attributes("/source"), "objects").values {
    Some(Structured([Reference(first), Reference(second)])) => {
      assert_eq(first.length(), 8)
      assert_eq(
        object_file.resolve_object_reference(first),
        Some({ address: target.address, kind: Dataset, path: Some("/target"), }),
      )
      assert_true(object_file.resolve_object_reference(second) is None)
    }
    _ => fail("expected added legacy object-reference values")
  }

  let object_type = Datatype::write_version4_object_reference()
  let version4_added = object_file.add_attribute(
    "/source",
    "v4_objects",
    [1UL],
    StructuredValues(object_type, [ObjectReference(Some("/group"))]),
  )
  let version4_file = Hdf5File::parse(version4_added)
  let group = version4_file.resolve("/group")
  match
    find_attribute(version4_file.attributes("/source"), "v4_objects").values {
    Some(Structured([Reference(bytes)])) =>
      assert_eq(
        version4_file.resolve_object_reference(bytes),
        Some({ address: group.address, kind: Group, path: Some("/group"), }),
      )
    _ => fail("expected an added version-4 object-reference value")
  }

  let region_type = Datatype::write_version4_region_reference()
  let region_added = version4_file.add_attribute(
    "/source",
    "v4_regions",
    [1UL],
    StructuredValues(region_type, [
      RegionReference(
        Some("/target"),
        RegionHyperslab([0UL, 0UL], [1UL, 1UL], [2UL, 2UL], [1UL, 1UL]),
      ),
    ]),
  )
  let region_file = Hdf5File::parse(region_added)
  match find_attribute(region_file.attributes("/source"), "v4_regions").values {
    Some(Structured([Reference(bytes)])) =>
      match region_file.resolve_region_reference(bytes) {
        Some(region) => {
          assert_eq(region.path, Some("/target"))
          assert_eq(
            region.selection,
            RegionHyperslab([0UL, 0UL], [1UL, 1UL], [2UL, 2UL], [1UL, 1UL]),
          )
        }
        None => fail("expected an added version-4 region reference")
      }
    _ => fail("expected added version-4 region-reference values")
  }

  let attribute_type = Datatype::write_version4_attribute_reference()
  let attribute_added = region_file.add_attribute(
    "/source",
    "v4_attributes",
    [1UL],
    StructuredValues(attribute_type, [
      AttributeReference(Some("/target"), "unit"),
    ]),
  )
  let attribute_file = Hdf5File::parse(attribute_added)
  match
    find_attribute(attribute_file.attributes("/source"), "v4_attributes").values {
    Some(Structured([Reference(bytes)])) =>
      assert_eq(
        attribute_file
        .resolve_attribute_reference(bytes)
        .map(target => (target.path, target.attribute_name)),
        Some((Some("/target"), "unit")),
      )
    _ => fail("expected added version-4 attribute-reference values")
  }
  assert_eq(integer_values(attribute_file.read_dataset("/source")), [5L])
}

///|
test "reference-valued attributes honor four-byte file addresses" {
  let file = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/references_layouts_32.h5"),
  )
  assert_eq(file.superblock.offset_size, 4)
  let added = file.add_attribute(
    "/compact",
    "added_ref",
    [1UL],
    ObjectReferenceValues([Some("/group")]),
  )
  let object_file = Hdf5File::parse(added)
  let group = object_file.resolve("/group")
  match find_attribute(object_file.attributes("/compact"), "added_ref").values {
    Some(Structured([Reference(bytes)])) => {
      assert_eq(bytes.length(), 4)
      assert_eq(
        object_file.resolve_object_reference(bytes),
        Some({ address: group.address, kind: Group, path: Some("/group"), }),
      )
    }
    _ => fail("expected a four-byte object-reference attribute")
  }

  let version4_type = Datatype::write_version4_object_reference_with_address_size(
    4,
  )
  let version4_added = object_file.add_attribute(
    "/compact",
    "added_v4_ref",
    [1UL],
    StructuredValues(version4_type, [ObjectReference(Some("/group"))]),
  )
  let version4_file = Hdf5File::parse(version4_added)
  match
    find_attribute(version4_file.attributes("/compact"), "added_v4_ref").values {
    Some(Structured([Reference(bytes)])) => {
      assert_eq(bytes.length(), 14)
      assert_eq(
        version4_file.resolve_object_reference(bytes),
        Some({ address: group.address, kind: Group, path: Some("/group"), }),
      )
    }
    _ => fail("expected a four-byte version-4 object-reference attribute")
  }
}

///|
test "attribute editing rejects invalid names and missing targets" {
  let writer = Hdf5Writer::new()
  writer.add_dataset("/dataset", [1UL], Int64Values([1L]))
  writer.add_attribute("/dataset", "existing", [], Int64Values([1L]))
  writer.add_attribute("/dataset", "other", [], Int64Values([2L]))
  let file = Hdf5File::parse(writer.encode())

  let duplicate = attribute_edit_error(() => {
    file.add_attribute("/dataset", "existing", [], Int64Values([3L])) |> ignore
  })
  assert_eq(
    duplicate,
    Some(InvalidArgument("attribute name already exists: existing")),
  )
  let missing_delete = attribute_edit_error(() => {
    file.delete_attribute("/dataset", "absent") |> ignore
  })
  assert_eq(
    missing_delete,
    Some(InvalidArgument("attribute does not exist: absent")),
  )
  let missing_rename = attribute_edit_error(() => {
    file.rename_attribute("/dataset", "absent", "renamed") |> ignore
  })
  assert_eq(
    missing_rename,
    Some(InvalidArgument("attribute does not exist: absent")),
  )
  let duplicate_rename = attribute_edit_error(() => {
    file.rename_attribute("/dataset", "existing", "other") |> ignore
  })
  assert_eq(
    duplicate_rename,
    Some(InvalidArgument("attribute name already exists: other")),
  )
  let empty_name = attribute_edit_error(() => {
    file.add_attribute("/dataset", "", [], Int64Values([3L])) |> ignore
  })
  assert_eq(
    empty_name,
    Some(InvalidArgument("attribute name is empty or too long")),
  )
  let nul_name = attribute_edit_error(() => {
    file.rename_attribute("/dataset", "existing", "bad\u{0}name") |> ignore
  })
  assert_eq(
    nul_name,
    Some(InvalidArgument("attribute names cannot contain NUL")),
  )

  let sources : Array[&Hdf5Source] = [writer.encode()]
  let sourced = Hdf5File::parse_source(sources[0])
  let source_error = attribute_edit_error(() => {
    sourced.delete_attribute("/dataset", "existing") |> ignore
  })
  assert_eq(
    source_error,
    Some(InvalidArgument("attribute editing requires a bytes-backed Hdf5File")),
  )
}
