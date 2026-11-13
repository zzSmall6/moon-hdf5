///|
test "compact variable-length string datasets can be patched repeatedly" {
  let writer = Hdf5Writer::new()
  writer.add_compact_dataset(
    "/labels",
    [4UL],
    VariableStringValues(["old", "", "旧值", "old"]),
  )
  writer.add_dataset("/counts", [2UL], Int64Values([7L, 8L]))
  writer.add_attribute(
    "/labels",
    "description",
    [],
    VariableStringValues(["原始"]),
  )
  let original = writer.encode()
  let file = Hdf5File::parse(original)
  let patched = file.patch_dataset_values(
    "/labels",
    VariableStringValues(["新值", "", "月亮", "新值"]),
  )
  let patched_file = Hdf5File::parse(patched)
  assert_eq(
    patched_file.read_dataset_metadata("/labels").layout.kind(),
    Compact,
  )
  assert_eq(string_values(patched_file.read_dataset("/labels")), [
    "新值", "", "月亮", "新值",
  ])
  assert_eq(integer_values(patched_file.read_dataset("/counts")), [7L, 8L])
  match
    find_attribute(patched_file.attributes("/labels"), "description").values {
    Some(Strings(values)) => assert_eq(values, ["原始"])
    _ => fail("expected the attribute to remain unchanged")
  }

  let repatched = patched_file.patch_dataset_values(
    "/labels",
    VariableStringValues(["second", "second", "", "第二"]),
  )
  let repatched_file = Hdf5File::parse(repatched)
  assert_eq(string_values(repatched_file.read_dataset("/labels")), [
    "second", "second", "", "第二",
  ])
  assert_eq(integer_values(repatched_file.read_dataset("/counts")), [7L, 8L])
}

///|
test "contiguous variable-length string datasets can be patched repeatedly" {
  let writer = Hdf5Writer::new()
  writer.add_dataset(
    "/names",
    [5UL],
    VariableStringValues(["alpha", "", "数据", "alpha", ""]),
  )
  writer.add_dataset("/numbers", [2UL], Int32Values([1, 2]))
  writer.add_attribute("/names", "units", [], FixedStringValues(["name"], 4))
  let file = Hdf5File::parse(writer.encode())
  let patched = file.patch_dataset_values(
    "/names",
    VariableStringValues(["第一个", "第一个", "", "终值", "第一个"]),
  )
  let patched_file = Hdf5File::parse(patched)
  assert_eq(
    patched_file.read_dataset_metadata("/names").layout.kind(),
    Contiguous,
  )
  assert_eq(string_values(patched_file.read_dataset("/names")), [
    "第一个", "第一个", "", "终值", "第一个",
  ])
  assert_eq(integer_values(patched_file.read_dataset("/numbers")), [1L, 2L])
  match find_attribute(patched_file.attributes("/names"), "units").values {
    Some(Strings(values)) => assert_eq(values, ["name"])
    _ => fail("expected the attribute to remain unchanged")
  }

  let repatched = patched_file.patch_dataset_values(
    "/names",
    VariableStringValues(["", "next", "next", "", "最后一"]),
  )
  assert_eq(string_values(Hdf5File::parse(repatched).read_dataset("/names")), [
    "", "next", "next", "", "最后一",
  ])
}

///|
test "unallocated contiguous variable-length string datasets are allocated" {
  let writer = Hdf5Writer::new()
  writer.add_dataset(
    "/names",
    [4UL],
    VariableStringValues(["old", "", "旧值", "old"]),
  )
  let encoded = writer.encode()
  let file = Hdf5File::parse(encoded)
  let header = file.object_header_at(file.resolve("/names").address)
  let message = match header.first(msg_data_layout) {
    Some(message) => message
    None => fail("expected a data layout message")
  }
  let metadata = file.read_dataset_metadata("/names")
  assert_true(metadata.layout.data_address_offset >= 0)
  assert_true(metadata.layout.data_address_width >= 1)
  let undefined = if file.superblock.offset_size == 8 {
    0xFFFFFFFFFFFFFFFFUL
  } else {
    (1UL << (file.superblock.offset_size * 8)) - 1UL
  }
  let output = encoded.to_fixedarray()
  patch_little_endian_uint(
    output,
    metadata.layout.data_address_offset,
    metadata.layout.data_address_width,
    undefined,
    "test unallocated dataset address",
  )
  if message.checksum_offset >= 0 {
    patch_metadata_checksum_in_place(
      output,
      message.chunk_start,
      message.checksum_offset,
    )
  }
  let unallocated = Hdf5File::parse(Bytes::from_array(output))
  assert_eq(
    unallocated.read_dataset_metadata("/names").layout.data_address,
    undefined,
  )
  let patched = unallocated.patch_dataset_values(
    "/names",
    VariableStringValues(["one", "", "一", "one"]),
  )
  let patched_file = Hdf5File::parse(patched)
  assert_eq(string_values(patched_file.read_dataset("/names")), [
    "one", "", "一", "one",
  ])
  assert_true(
    patched_file.read_dataset_metadata("/names").layout.data_address !=
    undefined,
  )
}

///|
test "chunked variable-length string datasets can be patched repeatedly" {
  let writer = Hdf5Writer::new()
  writer.add_chunked_dataset(
    "/names",
    [7UL],
    [2U],
    VariableStringValues(["alpha", "", "数据", "alpha", "", "tail", "tail"]),
  )
  writer.add_dataset("/numbers", [2UL], Int64Values([11L, 12L]))
  let file = Hdf5File::parse(writer.encode())
  let patched = file.patch_dataset_values(
    "/names",
    VariableStringValues([
      "第一", "第一", "", "中段", "中段", "", "结束",
    ]),
  )
  let patched_file = Hdf5File::parse(patched)
  assert_eq(patched_file.read_dataset_metadata("/names").layout.kind(), Chunked)
  assert_eq(string_values(patched_file.read_dataset("/names")), [
    "第一", "第一", "", "中段", "中段", "", "结束",
  ])
  assert_eq(integer_values(patched_file.read_dataset("/numbers")), [11L, 12L])

  let repatched = patched_file.patch_dataset_values(
    "/names",
    VariableStringValues(["again", "", "again", "月亮", "", "again", "again"]),
  )
  assert_eq(string_values(Hdf5File::parse(repatched).read_dataset("/names")), [
    "again", "", "again", "月亮", "", "again", "again",
  ])
}

///|
test "filtered chunked variable-length string datasets can be patched" {
  let writer = Hdf5Writer::new()
  writer.add_chunked_dataset(
    "/shuffle",
    [5UL],
    [2U],
    VariableStringValues(["alpha", "", "数据", "alpha", ""]),
    filters=[Shuffle],
  )
  writer.add_chunked_dataset(
    "/deflate",
    [5UL],
    [2U],
    VariableStringValues(["alpha", "", "数据", "alpha", ""]),
    filters=[Shuffle, Deflate(6)],
  )
  let file = Hdf5File::parse(writer.encode())
  let shuffle_patched = file.patch_dataset_values(
    "/shuffle",
    VariableStringValues(["one", "one", "", "二", "one"]),
  )
  assert_eq(
    string_values(Hdf5File::parse(shuffle_patched).read_dataset("/shuffle")),
    ["one", "one", "", "二", "one"],
  )
  let deflate_patched = file.patch_dataset_values(
    "/deflate",
    VariableStringValues(["", "压缩", "压缩", "后", "压缩"]),
  )
  assert_eq(
    string_values(Hdf5File::parse(deflate_patched).read_dataset("/deflate")),
    ["", "压缩", "压缩", "后", "压缩"],
  )
}

///|
test "contiguous variable-length sequence datasets and attributes are patched" {
  let sequence = Datatype::write_vlen_sequence(Datatype::write_integer(true, 4))
  let writer = Hdf5Writer::new()
  writer.add_dataset(
    "/sequences",
    [3UL],
    StructuredValues(sequence, [
      Sequence([]),
      Sequence([SignedInteger(1L), SignedInteger(2L)]),
      Sequence([SignedInteger(3L)]),
    ]),
  )
  writer.add_attribute(
    "/sequences",
    "template",
    [2UL],
    StructuredValues(sequence, [Sequence([SignedInteger(9L)]), Sequence([])]),
  )
  let file = Hdf5File::parse(writer.encode())
  let patched = file.patch_dataset_values(
    "/sequences",
    StructuredValues(sequence, [
      Sequence([SignedInteger(4L), SignedInteger(5L), SignedInteger(6L)]),
      Sequence([]),
      Sequence([SignedInteger(7L), SignedInteger(8L)]),
    ]),
  )
  let patched_file = Hdf5File::parse(patched)
  assert_eq(structured_values(patched_file.read_dataset("/sequences")), [
    Sequence([SignedInteger(4L), SignedInteger(5L), SignedInteger(6L)]),
    Sequence([]),
    Sequence([SignedInteger(7L), SignedInteger(8L)]),
  ])

  let attribute_patched = patched_file.patch_attribute_values(
    "/sequences",
    "template",
    StructuredValues(sequence, [
      Sequence([SignedInteger(10L), SignedInteger(11L)]),
      Sequence([SignedInteger(12L)]),
    ]),
  )
  let attribute_file = Hdf5File::parse(attribute_patched)
  match
    find_attribute(attribute_file.attributes("/sequences"), "template").values {
    Some(Structured(values)) =>
      assert_eq(values, [
        Sequence([SignedInteger(10L), SignedInteger(11L)]),
        Sequence([SignedInteger(12L)]),
      ])
    _ => fail("expected a patched variable-length sequence attribute")
  }

  let empty = attribute_file.patch_dataset_values(
    "/sequences",
    StructuredValues(sequence, [Sequence([]), Sequence([]), Sequence([])]),
  )
  assert_eq(empty.length(), attribute_patched.length())
  assert_eq(
    structured_values(Hdf5File::parse(empty).read_dataset("/sequences")),
    [Sequence([]), Sequence([]), Sequence([])],
  )
}

///|
test "filtered chunked variable-length sequence datasets are patched" {
  let sequence = Datatype::write_vlen_sequence(Datatype::write_integer(true, 4))
  let writer = Hdf5Writer::new()
  writer.add_chunked_dataset(
    "/sequences",
    [5UL],
    [2U],
    StructuredValues(sequence, [
      Sequence([SignedInteger(1L)]),
      Sequence([]),
      Sequence([SignedInteger(2L), SignedInteger(3L)]),
      Sequence([SignedInteger(4L)]),
      Sequence([]),
    ]),
    filters=[Shuffle, Deflate(6)],
  )
  let file = Hdf5File::parse(writer.encode())
  let patched = file.patch_dataset_values(
    "/sequences",
    StructuredValues(sequence, [
      Sequence([SignedInteger(5L), SignedInteger(6L)]),
      Sequence([SignedInteger(7L)]),
      Sequence([]),
      Sequence([SignedInteger(8L), SignedInteger(9L), SignedInteger(10L)]),
      Sequence([SignedInteger(11L)]),
    ]),
  )
  assert_eq(
    structured_values(Hdf5File::parse(patched).read_dataset("/sequences")),
    [
      Sequence([SignedInteger(5L), SignedInteger(6L)]),
      Sequence([SignedInteger(7L)]),
      Sequence([]),
      Sequence([SignedInteger(8L), SignedInteger(9L), SignedInteger(10L)]),
      Sequence([SignedInteger(11L)]),
    ],
  )
}

///|
test "shuffle filter parameters may be omitted as HDF5 writes them" {
  let datatype = Datatype::write_integer(true, 4)
  let filters = patch_write_filters(
    [{ id: 2, parameters: [], decoder: None, }],
    datatype,
    b"\x00\x00\x00\x00",
    [],
  )
  assert_eq(filters.length(), 1)
  assert_true(filters[0] is Shuffle)
}

///|
test "variable-length dataset patching rejects incompatible datatypes" {
  let writer = Hdf5Writer::new()
  let sequence = Datatype::write_vlen_sequence(Datatype::write_integer(true, 4))
  writer.add_chunked_dataset(
    "/sequences",
    [2UL],
    [1U],
    StructuredValues(sequence, [
      Sequence([SignedInteger(1L)]),
      Sequence([SignedInteger(2L)]),
    ]),
    filters=[Deflate(6)],
  )
  writer.add_dataset("/strings", [2UL], VariableStringValues(["a", "b"]))
  let file = Hdf5File::parse(writer.encode())
  let sequence_type_error = try {
    file.patch_dataset_values(
      "/sequences",
      VariableStringValues(["not", "sequences"]),
    )
    |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    sequence_type_error,
    Some(InvalidArgument("patched values must match the dataset datatype")),
  )
  let mismatch_error = try {
    file.patch_dataset_values("/strings", FixedStringValues(["a", "b"], 2))
    |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    mismatch_error,
    Some(InvalidArgument("patched values must match the dataset datatype")),
  )
}
