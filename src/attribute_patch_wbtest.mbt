///|
test "inline attribute values are patched without changing other metadata" {
  let writer = Hdf5Writer::new()
  writer.add_group("/group")
  writer.add_dataset("/group/values", [4UL], Int64Values([1L, 2L, 3L, 4L]))
  writer.add_attribute(
    "/group/values",
    "units",
    [],
    FixedStringValues(["count"], 8),
  )
  writer.add_attribute(
    "/group/values",
    "scale",
    [2UL],
    Float64Values([1.5, 2.5]),
  )
  let file = Hdf5File::parse(writer.encode())
  let patched = file.patch_attribute_values(
    "/group/values",
    "units",
    FixedStringValues(["items"], 8),
  )
  let patched_file = Hdf5File::parse(patched)
  let attributes = patched_file.attributes("/group/values")
  assert_eq(attributes.length(), 2)
  match find_attribute(attributes, "units").values {
    Some(Strings(values)) => assert_eq(values, ["items"])
    _ => fail("expected patched string attribute")
  }
  match find_attribute(attributes, "scale").values {
    Some(Floats(values)) => assert_eq(values, [1.5, 2.5])
    _ => fail("expected unchanged numeric attribute")
  }
  assert_eq(integer_values(patched_file.read_dataset("/group/values")), [
    1L, 2L, 3L, 4L,
  ])

  let repatched = patched_file.patch_attribute_values(
    "/group/values",
    "scale",
    Float64Values([-3.5, 4.25]),
  )
  match
    find_attribute(
      Hdf5File::parse(repatched).attributes("/group/values"),
      "scale",
    ).values {
    Some(Floats(values)) => assert_eq(values, [-3.5, 4.25])
    _ => fail("expected second attribute patch")
  }
}

///|
test "dense and ordered attributes are patched through fractal heaps" {
  let file = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/attribute_storage.h5"),
  )
  let patched = file.patch_attribute_values(
    "/dense",
    "attr_05",
    Int64Values([505L]),
  )
  let patched_dense = Hdf5File::parse(patched)
  let dense = patched_dense.attributes("/dense")
  assert_eq(dense.length(), 12)
  let names = [
    "attr_00", "attr_01", "attr_02", "attr_03", "attr_04", "attr_05", "attr_06",
    "attr_07", "attr_08", "attr_09", "attr_10", "attr_11",
  ]
  for index in 0..<dense.length() {
    let expected = if index == 5 { 505L } else { index.to_int64() }
    assert_eq(
      values_integers(find_attribute(dense, names[index]).values.unwrap()),
      [expected],
    )
  }
  assert_eq(patched_dense.attributes("/dense_ordered").length(), 12)

  let ordered = patched_dense.patch_attribute_values(
    "/dense_ordered",
    "attr_11",
    Int64Values([-111L]),
  )
  let ordered_file = Hdf5File::parse(ordered)
  let ordered_attributes = ordered_file.attributes("/dense_ordered")
  assert_eq(ordered_attributes.map(attribute => attribute.name), names)
  assert_eq(
    values_integers(
      find_attribute(ordered_attributes, "attr_11").values.unwrap(),
    ),
    [-111L],
  )
  assert_eq(
    values_integers(
      find_attribute(ordered_attributes, "attr_05").values.unwrap(),
    ),
    [5L],
  )
}

///|
test "big-endian and compact attributes can be patched" {
  let writer = Hdf5Writer::new()
  writer.add_dataset("/dataset", [1UL], Int64Values([1L]))
  let big_endian = Datatype::write_integer_with_order(true, 4, false)
  writer.add_attribute(
    "/dataset",
    "custom",
    [3UL],
    StructuredValues(big_endian, [
      SignedInteger(7L),
      SignedInteger(-8L),
      SignedInteger(123456L),
    ]),
  )
  let numeric = Hdf5File::parse(writer.encode())
  let patched_be = numeric.patch_attribute_values(
    "/dataset",
    "custom",
    StructuredValues(big_endian, [
      SignedInteger(-1L),
      SignedInteger(2L),
      SignedInteger(3L),
    ]),
  )
  let be_file = Hdf5File::parse(patched_be)
  match find_attribute(be_file.attributes("/dataset"), "custom").values {
    Some(Integers(values)) => assert_eq(values, [-1L, 2L, 3L])
    _ => fail("expected patched big-endian attribute")
  }
  assert_eq(integer_values(be_file.read_dataset("/dataset")), [1L])

  let storage = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/attribute_storage.h5"),
  )
  let patched_compact = storage.patch_attribute_values(
    "/compact",
    "attr_01",
    Int64Values([901L]),
  )
  let compact_file = Hdf5File::parse(patched_compact)
  assert_eq(
    values_integers(
      find_attribute(compact_file.attributes("/compact"), "attr_01").values.unwrap(),
    ),
    [901L],
  )
}

///|
test "attribute patching rejects missing incompatible source and vlen values" {
  let file = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/attribute_storage.h5"),
  )
  let missing = try {
    file.patch_attribute_values("/dense", "absent", Int64Values([1L])) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(missing, Some(InvalidArgument("attribute does not exist: absent")))
  let wrong_type = try {
    file.patch_attribute_values("/dense", "attr_00", Int32Values([1])) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    wrong_type,
    Some(InvalidArgument("patched values must match the attribute datatype")),
  )

  let writer = Hdf5Writer::new()
  writer.add_dataset("/dataset", [1UL], Int64Values([1L]))
  writer.add_attribute(
    "/dataset",
    "text",
    [1UL],
    VariableStringValues(["moon"]),
  )
  let encoded = writer.encode()
  let vlen_file = Hdf5File::parse(encoded)
  let vlen = try {
    vlen_file.patch_attribute_values(
      "/dataset",
      "text",
      VariableStringValues(["moonbit"]),
    )
    |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    vlen,
    Some(
      UnsupportedFeature(
        "variable-length and reference attributes cannot be patched",
      ),
    ),
  )
  let sources : Array[&Hdf5Source] = [encoded]
  let sourced = Hdf5File::parse_source(sources[0])
  let source_error = try {
    sourced.patch_attribute_values(
      "/dataset",
      "text",
      VariableStringValues(["moonbit"]),
    )
    |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    source_error,
    Some(InvalidArgument("attribute patching requires a bytes-backed Hdf5File")),
  )
}
