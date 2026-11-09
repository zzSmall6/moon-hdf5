///|
fn chunk_integer_values(chunk : DatasetChunk) -> Array[Int64] raise Hdf5Error {
  match chunk.values {
    Integers(values) => values
    _ => raise InvalidField("expected integer chunk values")
  }
}

///|
test "chunk iterator clips tiles and reads bounded selections" {
  let writer = Hdf5Writer::new()
  let values = Array::make(20, 0L)
  for index in 0..<values.length() {
    values[index] = index.to_int64()
  }
  writer.add_dataset("/grid", [5UL, 4UL], Int64Values(values))
  let file = Hdf5File::parse(writer.encode())
  let iterator = file.chunk_iterator("/grid", [2UL, 3UL])
  let offsets : Array[Array[UInt64]] = []
  let dimensions : Array[Array[UInt64]] = []
  let first_values : Array[Int64] = []
  let mut next = iterator.next()
  while next is Some(chunk) {
    offsets.push(chunk.offset)
    dimensions.push(chunk.dimensions)
    if first_values.length() == 0 {
      for value in chunk_integer_values(chunk) {
        first_values.push(value)
      }
    }
    next = iterator.next()
  }
  assert_eq(offsets, [
    [0UL, 0UL],
    [0UL, 3UL],
    [2UL, 0UL],
    [2UL, 3UL],
    [4UL, 0UL],
    [4UL, 3UL],
  ])
  assert_eq(dimensions, [
    [2UL, 3UL],
    [2UL, 1UL],
    [2UL, 3UL],
    [2UL, 1UL],
    [1UL, 3UL],
    [1UL, 1UL],
  ])
  assert_eq(first_values, [0L, 1L, 2L, 4L, 5L, 6L])
}

///|
test "chunk iterator supports chunked storage and early stop" {
  let writer = Hdf5Writer::new()
  let values = Array::make(24, 0L)
  for index in 0..<values.length() {
    values[index] = index.to_int64()
  }
  writer.add_chunked_dataset(
    "/chunked",
    [4UL, 6UL],
    [2U, 2U],
    Int64Values(values),
  )
  let file = Hdf5File::parse(writer.encode())
  let iterator = file.chunk_iterator("/chunked", [3UL, 4UL])
  let first = iterator.next().unwrap()
  assert_eq(first.offset, [0UL, 0UL])
  assert_eq(first.dimensions, [3UL, 4UL])
  assert_eq(chunk_integer_values(first), [
    0L, 1L, 2L, 3L, 6L, 7L, 8L, 9L, 12L, 13L, 14L, 15L,
  ])
}

///|
test "chunk iterator handles scalar and empty datasets" {
  let writer = Hdf5Writer::new()
  writer.add_dataset("/scalar", [], Int64Values([9L]))
  writer.add_dataset("/empty", [0UL, 3UL], Int64Values([]))
  let file = Hdf5File::parse(writer.encode())
  let scalar = file.chunk_iterator("/scalar", [])
  let scalar_chunk = scalar.next().unwrap()
  assert_eq(scalar_chunk.offset, [])
  assert_eq(scalar_chunk.dimensions, [])
  assert_eq(chunk_integer_values(scalar_chunk), [9L])
  assert_true(scalar.next() is None)
  let empty = file.chunk_iterator("/empty", [2UL, 2UL])
  assert_true(empty.next() is None)
}

///|
test "selection iterator returns bounded pages in selection order" {
  let writer = Hdf5Writer::new()
  let values = Array::make(48, 0L)
  for index in 0..<values.length() {
    values[index] = index.to_int64()
  }
  writer.add_chunked_dataset(
    "/grid",
    [6UL, 8UL],
    [2U, 3U],
    Int64Values(values),
    filters=[Shuffle, Deflate(4)],
  )
  let file = Hdf5File::parse(writer.encode())
  let selection = Hyperslab([1UL, 2UL], [4UL, 5UL], [1UL, 1UL])
  let expected = file.read_dataset_selection("/grid", selection)
  let expected_values = match expected.values {
    Integers(items) => items
    _ => fail("expected integer selection")
  }
  let iterator = file.selection_iterator("/grid", selection, 3)
  let offsets : Array[UInt64] = []
  let actual : Array[Int64] = []
  let mut next = iterator.next()
  while next is Some(page) {
    offsets.push(page.offset)
    match page.values {
      Integers(items) => {
        assert_true(items.length() > 0 && items.length() <= 3)
        for value in items {
          actual.push(value)
        }
      }
      _ => fail("expected integer selection page")
    }
    next = iterator.next()
  }
  assert_eq(offsets, [0UL, 3UL, 6UL, 9UL, 12UL, 15UL, 18UL])
  assert_eq(actual, expected_values)
}

///|
test "selection iterator handles points and empty selections" {
  let writer = Hdf5Writer::new()
  writer.add_dataset("/values", [5UL], Int64Values([10L, 20L, 30L, 40L, 50L]))
  let file = Hdf5File::parse(writer.encode())
  let points = Points([[3UL], [0UL], [3UL], [1UL]])
  let iterator = file.selection_iterator("/values", points, 2)
  let first = iterator.next().unwrap()
  assert_eq(first.offset, 0UL)
  match first.values {
    Integers(values) => assert_eq(values, [40L, 10L])
    _ => fail("expected first point page")
  }
  let second = iterator.next().unwrap()
  assert_eq(second.offset, 2UL)
  match second.values {
    Integers(values) => assert_eq(values, [40L, 20L])
    _ => fail("expected second point page")
  }
  assert_true(iterator.next() is None)

  let empty = file.selection_iterator(
    "/values",
    Hyperslab([0UL], [0UL], [1UL]),
    4,
  )
  assert_true(empty.next() is None)
}

///|
test "selection iterator validates page size and dataset targets" {
  let writer = Hdf5Writer::new()
  writer.add_dataset("/values", [2UL], Int64Values([1L, 2L]))
  let file = Hdf5File::parse(writer.encode())
  let invalid_page = try {
    file.selection_iterator("/values", Points([[0UL]]), 0) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_true(invalid_page is Some(InvalidArgument(_)))
  let not_dataset = try {
    file.selection_iterator("/", Points([]), 1) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_true(not_dataset is Some(InvalidField(_)))
}
