///|
test "nested variable-length strings sequences and arrays can be patched" {
  let text_type = Datatype::write_vlen_string()
  let sequence_type = Datatype::write_vlen_sequence(
    Datatype::write_integer(true, 4),
  )
  let labels_type = Datatype::write_array(text_type, [2UL])
  let record_type = Datatype::write_compound([
    ("name", text_type),
    ("counts", sequence_type),
    ("labels", labels_type),
  ])
  let record = fn(
    name : String,
    counts : Array[DatasetValue],
    left : String,
    right : String,
  ) -> DatasetValue {
    Compound([
      { name: "name", value: Text(name), },
      { name: "counts", value: Sequence(counts), },
      { name: "labels", value: Elements([Text(left), Text(right)]), },
    ])
  }
  let original = StructuredValues(record_type, [
    record("old", [SignedInteger(1L)], "a", ""),
    record("", [], "", "b"),
  ])
  let writer = Hdf5Writer::new()
  writer.add_dataset("/records", [2UL], original)
  writer.add_attribute(
    "/records",
    "template",
    [1UL],
    StructuredValues(record_type, [record("template", [], "x", "y")]),
  )
  writer.add_chunked_dataset(
    "/chunked",
    [4UL],
    [2U],
    StructuredValues(record_type, [
      record("first", [SignedInteger(1L), SignedInteger(2L)], "p", ""),
      record("", [], "", "q"),
      record("third", [SignedInteger(3L)], "r", "s"),
      record("last", [], "t", ""),
    ]),
    filters=[Shuffle, Deflate(6)],
  )
  let file = Hdf5File::parse(writer.encode())
  let replacement = [
    record("new", [SignedInteger(4L), SignedInteger(5L)], "新", ""),
    record("", [], "", "终"),
  ]
  let patched = Hdf5File::parse(
    file.patch_dataset_values(
      "/records",
      StructuredValues(record_type, replacement),
    ),
  )
  match patched.read_dataset("/records").values {
    Structured(values) => assert_eq(values, replacement)
    _ => fail("expected patched nested variable-length records")
  }

  let attribute_patched = Hdf5File::parse(
    patched.patch_attribute_values(
      "/records",
      "template",
      StructuredValues(record_type, [
        record("attribute", [SignedInteger(6L)], "", "值"),
      ]),
    ),
  )
  match
    find_attribute(attribute_patched.attributes("/records"), "template").values {
    Some(Structured(values)) =>
      assert_eq(values, [record("attribute", [SignedInteger(6L)], "", "值")])
    _ => fail("expected a patched nested variable-length attribute")
  }

  let chunked = Hdf5File::parse(
    attribute_patched.patch_dataset_values(
      "/chunked",
      StructuredValues(record_type, [
        record("", [], "", ""),
        record("压缩", [SignedInteger(7L), SignedInteger(8L)], "一", "二"),
        record("", [], "三", ""),
        record("tail", [SignedInteger(9L)], "", "四"),
      ]),
    ),
  )
  match chunked.read_dataset("/chunked").values {
    Structured(values) =>
      assert_eq(values, [
        record("", [], "", ""),
        record("压缩", [SignedInteger(7L), SignedInteger(8L)], "一", "二"),
        record("", [], "三", ""),
        record("tail", [SignedInteger(9L)], "", "四"),
      ])
    _ => fail("expected patched filtered nested variable-length records")
  }
}

///|
test "h5py nested variable-length sequences are patched recursively" {
  let file = Hdf5File::parse(@fs.read_file_to_bytes("testdata/nested_vlen.h5"))
  let dataset = file.read_dataset("/nested_sequences")
  assert_eq(dataset.datatype.class, 9)
  assert_false(dataset.datatype.vlen_is_string)
  assert_eq(dataset.datatype.base_types[0].class, 9)
  assert_true(dataset.datatype.base_types[0].vlen_is_string)
  match dataset.values {
    Structured(values) =>
      assert_eq(values, [
        Sequence([]),
        Sequence([Text("alpha"), Text(""), Text("数据")]),
        Sequence([Text("重复")]),
      ])
    _ => fail("expected nested variable-length sequences")
  }

  let datatype = dataset.datatype
  let replacement = StructuredValues(datatype, [
    Sequence([Text("new"), Text("")]),
    Sequence([]),
    Sequence([Text("终")]),
  ])
  let replacement_items = match replacement {
    StructuredValues(_, items) => items
    _ => fail("expected structured replacement values")
  }
  let patched = Hdf5File::parse(
    file.patch_dataset_values("/nested_sequences", replacement),
  )
  match patched.read_dataset("/nested_sequences").values {
    Structured(values) => assert_eq(values, replacement_items)
    _ => fail("expected patched nested variable-length sequences")
  }

  let attribute = find_attribute(patched.attributes("/"), "nested_template")
  let attribute_replacement = StructuredValues(attribute.datatype, [
    Sequence([Text("attribute"), Text("值")]),
  ])
  let attribute_replacement_items = match attribute_replacement {
    StructuredValues(_, items) => items
    _ => fail("expected structured attribute replacement values")
  }
  let attribute_patched = Hdf5File::parse(
    patched.patch_attribute_values(
      "/", "nested_template", attribute_replacement,
    ),
  )
  match
    find_attribute(attribute_patched.attributes("/"), "nested_template").values {
    Some(Structured(values)) => assert_eq(values, attribute_replacement_items)
    _ => fail("expected patched nested variable-length attribute")
  }

  let chunked_replacement = StructuredValues(datatype, [
    Sequence([]),
    Sequence([Text("压缩"), Text("")]),
    Sequence([Text("一"), Text("二")]),
    Sequence([]),
  ])
  let chunked_replacement_items = match chunked_replacement {
    StructuredValues(_, items) => items
    _ => fail("expected structured chunked replacement values")
  }
  let chunked = Hdf5File::parse(
    attribute_patched.patch_dataset_values(
      "/nested_sequences_chunked", chunked_replacement,
    ),
  )
  match chunked.read_dataset("/nested_sequences_chunked").values {
    Structured(values) => assert_eq(values, chunked_replacement_items)
    _ => fail("expected patched chunked nested variable-length sequences")
  }
}
