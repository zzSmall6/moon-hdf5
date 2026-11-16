///|
fn resize_test_integers(dataset : Dataset) -> Array[Int64] raise Hdf5Error {
  match dataset.values {
    Integers(values) => values
    _ => raise InvalidField("expected integer dataset values")
  }
}

///|
fn resize_test_strings(dataset : Dataset) -> Array[String] raise Hdf5Error {
  match dataset.values {
    Strings(values) => values
    _ => raise InvalidField("expected string dataset values")
  }
}

///|
test "existing files resize extensible-array datasets without moving chunks" {
  let writer = Hdf5Writer::new()
  try {
    writer.add_extensible_dataset(
      "/series",
      [2UL],
      [None],
      [2U],
      Int64Values([1L, 2L]),
      fill_value=Int64Values([-1L]),
    )
    writer.add_attribute("/series", "unit", [], FixedStringValues(["items"], 8))
  } catch {
    error => abort("create: " + error.message())
  }
  let file = Hdf5File::parse(writer.encode())
  let resized = file.resize_dataset("/series", [6UL]) catch {
    error => abort("resize: " + error.message())
  }
  let grown = Hdf5File::parse(resized)
  let dataset = grown.read_dataset("/series") catch {
    error => abort("read: " + error.message())
  }
  assert_eq(dataset.dataspace.dimensions, [6UL])
  assert_eq(dataset.dataspace.maximum_dimensions, [0xFFFFFFFFFFFFFFFFUL])
  assert_true(dataset.layout.chunk_index is ExtensibleArray(_))
  assert_eq(resize_test_integers(dataset), [1L, 2L, -1L, -1L, -1L, -1L])
  assert_eq(grown.attributes("/series")[0].name, "unit")

  let shrunk = Hdf5File::parse(grown.resize_dataset("/series", [2UL])) catch {
    error => abort("shrink: " + error.message())
  }
  assert_eq(resize_test_integers(shrunk.read_dataset("/series")), [1L, 2L])
  let regrown = Hdf5File::parse(shrunk.resize_dataset("/series", [6UL])) catch {
    error => abort("regrow: " + error.message())
  }
  assert_eq(resize_test_integers(regrown.read_dataset("/series")), [
    1L, 2L, -1L, -1L, -1L, -1L,
  ])
}

///|
test "existing files resize multiple unlimited dimensions through a v2 b-tree" {
  let writer = Hdf5Writer::new()
  writer.add_extensible_dataset(
    "/matrix",
    [2UL, 2UL],
    [None, None],
    [1U, 1U],
    Int64Values([1L, 2L, 3L, 4L]),
    fill_value=Int64Values([-1L]),
  )
  let file = Hdf5File::parse(writer.encode())
  let grown = Hdf5File::parse(file.resize_dataset("/matrix", [3UL, 4UL]))
  let dataset = grown.read_dataset("/matrix")
  assert_eq(dataset.dataspace.dimensions, [3UL, 4UL])
  assert_eq(dataset.dataspace.maximum_dimensions, [
    0xFFFFFFFFFFFFFFFFUL, 0xFFFFFFFFFFFFFFFFUL,
  ])
  assert_true(dataset.layout.chunk_index is BTreeV2(_))
  assert_eq(resize_test_integers(dataset), [
    1L, 2L, -1L, -1L, 3L, 4L, -1L, -1L, -1L, -1L, -1L, -1L,
  ])
}

///|
test "existing files reject unsupported dataset resize storage and bounds" {
  let writer = Hdf5Writer::new()
  writer.add_dataset("/fixed", [2UL], Int64Values([1L, 2L]))
  writer.add_chunked_dataset(
    "/fixed_chunks",
    [2UL],
    [2U],
    Int64Values([1L, 2L]),
  )
  writer.add_extensible_dataset(
    "/bounded",
    [2UL],
    [None],
    [2U],
    Int64Values([1L, 2L]),
  )
  let file = Hdf5File::parse(writer.encode())
  let contiguous = try {
    file.resize_dataset("/fixed", [3UL]) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    contiguous,
    Some(UnsupportedFeature("dataset resize requires chunked storage")),
  )
  let fixed_chunks = try {
    file.resize_dataset("/fixed_chunks", [3UL]) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    fixed_chunks,
    Some(UnsupportedFeature("dataset chunk index does not support resizing")),
  )
  let wrong_rank = try {
    file.resize_dataset("/bounded", [5UL, 1UL]) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    wrong_rank,
    Some(InvalidArgument("resized dataset rank does not match")),
  )
}

///|
test "existing files append values at the trailing corner" {
  let writer = Hdf5Writer::new()
  writer.add_extensible_dataset(
    "/series",
    [2UL],
    [None],
    [2U],
    Int64Values([1L, 2L]),
    fill_value=Int64Values([-1L]),
  )
  writer.add_extensible_dataset(
    "/matrix",
    [2UL, 2UL],
    [None, None],
    [1U, 1U],
    Int64Values([1L, 2L, 3L, 4L]),
    fill_value=Int64Values([-1L]),
  )
  let file = Hdf5File::parse(writer.encode())
  let appended_bytes = file.append_dataset(
    "/series",
    [2UL],
    Int64Values([3L, 4L]),
  ) catch {
    error => abort("append integers: " + error.message())
  }
  let appended = Hdf5File::parse(appended_bytes) catch {
    error => abort("parse appended integers: " + error.message())
  }
  let appended_values = appended.read_dataset("/series") catch {
    error => abort("read appended integers: " + error.message())
  }
  assert_eq(resize_test_integers(appended_values), [1L, 2L, 3L, 4L])
  let matrix_bytes = file.append_dataset(
    "/matrix",
    [1UL, 1UL],
    Int64Values([5L]),
  ) catch {
    error => abort("append matrix: " + error.message())
  }
  let matrix_file = Hdf5File::parse(matrix_bytes) catch {
    error => abort("parse appended matrix: " + error.message())
  }
  let matrix = matrix_file.read_dataset("/matrix") catch {
    error => abort("read appended matrix: " + error.message())
  }
  assert_eq(matrix.dataspace.dimensions, [3UL, 3UL])
  assert_eq(resize_test_integers(matrix), [
    1L, 2L, -1L, 3L, 4L, -1L, -1L, -1L, 5L,
  ])
}

///|
test "existing files append variable-length values" {
  let writer = Hdf5Writer::new()
  writer.add_extensible_dataset(
    "/names",
    [1UL],
    [None],
    [1U],
    VariableStringValues(["moon"]),
    fill_value=VariableStringValues(["missing"]),
  )
  let file = Hdf5File::parse(writer.encode())
  let appended_bytes = file.append_dataset(
    "/names",
    [2UL],
    VariableStringValues(["数据", ""]),
  ) catch {
    error => abort("append vlen: " + error.message())
  }
  let appended = Hdf5File::parse(appended_bytes) catch {
    error => abort("parse appended vlen: " + error.message())
  }
  let appended_values = appended.read_dataset("/names") catch {
    error => abort("read appended vlen: " + error.message())
  }
  assert_eq(resize_test_strings(appended_values), ["moon", "数据", ""])
}

///|
test "existing files append filtered variable-length values" {
  let writer = Hdf5Writer::new()
  writer.add_extensible_dataset(
    "/names",
    [1UL],
    [None],
    [2U],
    VariableStringValues(["moon"]),
    fill_value=VariableStringValues(["missing"]),
    filters=[Shuffle, Deflate(6)],
  )
  let file = Hdf5File::parse(writer.encode())
  let bytes = file.append_dataset(
    "/names",
    [3UL],
    VariableStringValues(["数据", "", "tail"]),
  ) catch {
    error => abort("append filtered vlen: " + error.message())
  }
  let appended = Hdf5File::parse(bytes)
  assert_eq(resize_test_strings(appended.read_dataset("/names")), [
    "moon", "数据", "", "tail",
  ])
}

///|
test "existing files append across extensible-array data blocks" {
  let cases : Array[(Array[Int32], Array[Hdf5WriteFilter])] = [
    (Array::make(2000, 5), []),
    (Array::make(513, 7), [Shuffle, Deflate(6)]),
  ]
  for (appended_values, filters) in cases {
    let writer = Hdf5Writer::new()
    writer.add_extensible_dataset(
      "/data",
      [4UL],
      [None],
      [4U],
      Int32Values([1, 2, 3, 4]),
      fill_value=Int32Values([-7]),
      filters~,
    )
    let file = Hdf5File::parse(writer.encode())
    let bytes = file.append_dataset(
      "/data",
      [appended_values.length().to_uint64()],
      Int32Values(appended_values),
    ) catch {
      error => abort("append across data blocks: " + error.message())
    }
    let appended = Hdf5File::parse(bytes)
    let dataset = appended.read_dataset("/data")
    assert_true(dataset.layout.chunk_index is ExtensibleArray(_))
    let expected = [1L, 2L, 3L, 4L]
    for value in appended_values {
      expected.push(value.to_int64())
    }
    assert_eq(resize_test_integers(dataset), expected)
  }
}

///|
test "existing files append v1 and v2 b-tree datasets" {
  let v1_bytes = @fs.read_file_to_bytes("testdata/interop_earliest.h5")
  let v1 = Hdf5File::parse(v1_bytes)
  let v1_before = resize_test_integers(v1.read_dataset("/data/unlimited_one"))
  let v1_appended_values = Array::make(14, 900)
  let v1_bytes = v1.append_dataset(
    "/data/unlimited_one",
    [2UL, 7UL],
    Int32Values(v1_appended_values),
  ) catch {
    error => abort("append v1 tree: " + error.message())
  }
  let v1_after = Hdf5File::parse(v1_bytes)
  let v1_dataset = v1_after.read_dataset("/data/unlimited_one")
  assert_true(v1_dataset.layout.chunk_index is BTreeV1)
  assert_eq(v1_dataset.dataspace.dimensions, [7UL, 7UL])
  let v1_expected = v1_before.copy()
  for value in v1_appended_values {
    v1_expected.push(value.to_int64())
  }
  assert_eq(resize_test_integers(v1_dataset), v1_expected)

  let writer = Hdf5Writer::new()
  writer.add_extensible_dataset(
    "/data",
    [2UL, 2UL],
    [None, None],
    [1U, 1U],
    Int32Values([1, 2, 3, 4]),
    fill_value=Int32Values([-1]),
    filters=[Shuffle, Deflate(5)],
  )
  let v2 = Hdf5File::parse(writer.encode())
  let v2_bytes = v2.append_dataset(
    "/data",
    [2UL, 2UL],
    Int32Values([5, 6, 7, 8]),
  ) catch {
    error => abort("append v2 tree: " + error.message())
  }
  let v2_dataset = Hdf5File::parse(v2_bytes).read_dataset("/data")
  assert_true(v2_dataset.layout.chunk_index is BTreeV2(_))
  assert_eq(v2_dataset.dataspace.dimensions, [4UL, 4UL])
  assert_eq(resize_test_integers(v2_dataset), [
    1L, 2L, -1L, -1L, 3L, 4L, -1L, -1L, -1L, -1L, 5L, 6L, -1L, -1L, 7L, 8L,
  ])
}

///|
test "existing files append variable-length values beyond the index entries" {
  let writer = Hdf5Writer::new()
  let initial = ["moon"]
  writer.add_extensible_dataset(
    "/names",
    [1UL],
    [None],
    [1U],
    VariableStringValues(initial),
    fill_value=VariableStringValues(["missing"]),
  )
  let file = Hdf5File::parse(writer.encode())
  let appended_values : Array[String] = []
  for index in 0..<200 {
    appended_values.push("value-" + index.to_string())
  }
  let bytes = file.append_dataset(
    "/names",
    [appended_values.length().to_uint64()],
    VariableStringValues(appended_values),
  ) catch {
    error => abort("append large vlen: " + error.message())
  }
  let appended = Hdf5File::parse(bytes)
  let expected = ["moon"]
  for value in appended_values {
    expected.push(value)
  }
  assert_eq(resize_test_strings(appended.read_dataset("/names")), expected)
}
