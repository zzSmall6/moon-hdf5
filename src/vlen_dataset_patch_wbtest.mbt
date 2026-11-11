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
  let sequence_error = try {
    file.patch_dataset_values(
      "/sequences",
      StructuredValues(sequence, [
        Sequence([SignedInteger(3L)]),
        Sequence([SignedInteger(4L)]),
      ]),
    )
    |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    sequence_error,
    Some(
      UnsupportedFeature("only variable-length string datatypes can be patched"),
    ),
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
  let reference_writer = Hdf5Writer::new()
  reference_writer.add_dataset(
    "/reference",
    [1UL],
    ObjectReferenceValues([None]),
  )
  let reference_file = Hdf5File::parse(reference_writer.encode())
  let reference_error = try {
    reference_file.patch_dataset_values(
      "/reference",
      ObjectReferenceValues([None]),
    )
    |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    reference_error,
    Some(UnsupportedFeature("reference datatypes cannot be patched")),
  )
}
