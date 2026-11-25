///|
priv struct GeneratedInt32ChunkSource {
  dimensions : Array[UInt64]
  values : Array[Int32]
  mut reads : Int32
  mut largest : Int32
  fail_at : Int32
}

///|
fn chunk_source_shape_count(dimensions : Array[UInt64]) -> Int64 {
  let mut count = 1L
  for dimension in dimensions {
    count = count * dimension.reinterpret_as_int64()
  }
  count
}

///|
impl Hdf5ChunkSource for GeneratedInt32ChunkSource with fn read_chunk(
  self : GeneratedInt32ChunkSource,
  chunk_index : Int32,
  offset : Array[UInt64],
  dimensions : Array[UInt64],
) -> Hdf5WriteValues raise Hdf5Error {
  self.reads = self.reads + 1
  if self.fail_at == chunk_index {
    raise InvalidArgument(
      "generated source rejected chunk " + chunk_index.to_string(),
    )
  }
  guard offset.length() == self.dimensions.length() &&
    dimensions.length() == self.dimensions.length() else {
    raise InvalidField("source chunk rank changed")
  }
  let count = chunk_source_shape_count(dimensions)
  if count.to_int() > self.largest {
    self.largest = count.to_int()
  }
  let items : Array[Int32] = []
  let rank = dimensions.length()
  for linear in 0..<count.to_int() {
    let coordinates = Array::make(rank, 0)
    let mut remaining = linear.to_int64()
    let mut axis = rank
    while axis > 0 {
      axis = axis - 1
      let width = dimensions[axis].reinterpret_as_int64()
      coordinates[axis] = (remaining % width).to_int()
      remaining = remaining / width
    }
    let mut source_index = 0L
    for axis in 0..<rank {
      let coordinate = offset[axis].reinterpret_as_int64() +
        coordinates[axis].to_int64()
      guard coordinate < self.dimensions[axis].reinterpret_as_int64() else {
        raise InvalidField("source chunk lies outside the dataset")
      }
      source_index = source_index * self.dimensions[axis].reinterpret_as_int64() +
        coordinate
    }
    items.push(self.values[source_index.to_int()])
  }
  Int32Values(items)
}

///|
priv struct GeneratedStringChunkSource {
  values : Array[String]
  mut reads : Int32
}

///|
impl Hdf5ChunkSource for GeneratedStringChunkSource with fn read_chunk(
  self : GeneratedStringChunkSource,
  chunk_index : Int32,
  offset : Array[UInt64],
  dimensions : Array[UInt64],
) -> Hdf5WriteValues raise Hdf5Error {
  self.reads = self.reads + 1
  ignore(chunk_index)
  guard offset.length() == 1 && dimensions.length() == 1 else {
    raise InvalidField("string source expects rank one chunks")
  }
  let items : Array[String] = []
  for index in 0..<dimensions[0].reinterpret_as_int64().to_int() {
    items.push(self.values[offset[0].reinterpret_as_int64().to_int() + index])
  }
  VariableStringValues(items)
}

///|
fn chunk_source_file_integers(
  bytes : Bytes,
  path : String,
) -> Array[Int64] raise Hdf5Error {
  match Hdf5File::parse(bytes).read_dataset(path).values {
    Integers(values) => values
    _ => raise InvalidField("expected integer dataset values")
  }
}

///|
fn chunk_source_file_strings(
  bytes : Bytes,
  path : String,
) -> Array[String] raise Hdf5Error {
  match Hdf5File::parse(bytes).read_dataset(path).values {
    Strings(values) => values
    _ => raise InvalidField("expected string dataset values")
  }
}

///|
test "finite chunk source matches in-memory filtered chunks" {
  let dimensions = [5UL, 7UL]
  let values = Array::make(35, 0)
  for index in 0..<values.length() {
    values[index] = index - 10
  }
  values[0] = -7
  values[11] = -7
  let source = { dimensions, values, reads: 0, largest: 0, fail_at: -1, }
  let source_writer = Hdf5Writer::new()
  source_writer.add_chunked_dataset_from_source(
    "/values",
    dimensions,
    [2U, 3U],
    Int32Values([]),
    source,
    fill_value=Int32Values([-7]),
    filters=[Shuffle, Deflate(6), Fletcher32],
  )
  let source_bytes = source_writer.encode()
  let direct_writer = Hdf5Writer::new()
  direct_writer.add_chunked_dataset(
    "/values",
    dimensions,
    [2U, 3U],
    Int32Values(values),
    fill_value=Int32Values([-7]),
    filters=[Shuffle, Deflate(6), Fletcher32],
  )
  assert_eq(
    chunk_source_file_integers(source_bytes, "/values"),
    chunk_source_file_integers(direct_writer.encode(), "/values"),
  )
  assert_true(source.reads > 0 && source.reads <= 15 * 4)
  assert_true(source.largest <= 6)
}

///|
test "extensible chunk sources cover one and multiple unlimited axes" {
  let dimensions = [3UL, 4UL]
  let values = Array::make(12, 0)
  for index in 0..<values.length() {
    values[index] = index + 1
  }
  let one_source = { dimensions, values, reads: 0, largest: 0, fail_at: -1, }
  let one = Hdf5Writer::new()
  one.add_extensible_dataset_from_source(
    "/series",
    dimensions,
    [None, Some(4UL)],
    [2U, 2U],
    Int32Values([]),
    one_source,
    fill_value=Int32Values([-1]),
    filters=[Deflate(4)],
  )
  assert_eq(chunk_source_file_integers(one.encode(), "/series"), [
    1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, 10L, 11L, 12L,
  ])

  let many_source = { dimensions, values, reads: 0, largest: 0, fail_at: -1, }
  let many = Hdf5Writer::new()
  many.add_extensible_dataset_from_source(
    "/matrix",
    dimensions,
    [None, None],
    [1U, 2U],
    Int32Values([]),
    many_source,
    fill_value=Int32Values([-1]),
    filters=[Shuffle, Deflate(4)],
  )
  assert_eq(chunk_source_file_integers(many.encode(), "/matrix"), [
    1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, 10L, 11L, 12L,
  ])
}

///|
test "variable-length chunk sources build the global heap incrementally" {
  let values = ["alpha", "", "数据", "alpha", "beta", "", "数据"]
  let source = { values, reads: 0, }
  let writer = Hdf5Writer::new()
  writer.add_chunked_dataset_from_source(
    "/names",
    [7UL],
    [3U],
    VariableStringValues([]),
    source,
    filters=[Shuffle, Deflate(6)],
  )
  assert_eq(chunk_source_file_strings(writer.encode(), "/names"), [
    "alpha", "", "数据", "alpha", "beta", "", "数据",
  ])
  assert_true(source.reads > 0)
}

///|
test "chunk source failures propagate and streamed datasets are immutable" {
  let failing = {
    dimensions: [4UL],
    values: [1, 2, 3, 4],
    reads: 0,
    largest: 0,
    fail_at: 1,
  }
  let writer = Hdf5Writer::new()
  writer.add_chunked_dataset_from_source(
    "/values",
    [4UL],
    [2U],
    Int32Values([]),
    failing,
  )
  assert_true(
    try {
      writer.encode() |> ignore
      false
    } catch {
      InvalidArgument(message) => message.contains("rejected chunk 1")
      _ => false
    },
  )

  let source = {
    dimensions: [4UL],
    values: [1, 2, 3, 4],
    reads: 0,
    largest: 0,
    fail_at: -1,
  }
  let extensible = Hdf5Writer::new()
  extensible.add_extensible_dataset_from_source(
    "/values",
    [4UL],
    [None],
    [2U],
    Int32Values([]),
    source,
  )
  assert_true(
    try {
      extensible.resize_dataset("/values", [6UL])
      false
    } catch {
      UnsupportedFeature(_) => true
      _ => false
    },
  )
  assert_true(
    try {
      extensible.write_dataset_slab("/values", [0UL], [1UL], Int32Values([9]))
      false
    } catch {
      UnsupportedFeature(_) => true
      _ => false
    },
  )
}

///|
test "chunk source validates prototype and returned datatype" {
  let source = {
    dimensions: [2UL],
    values: [1, 2],
    reads: 0,
    largest: 0,
    fail_at: -1,
  }
  let writer = Hdf5Writer::new()
  assert_true(
    try {
      writer.add_chunked_dataset_from_source(
        "/bad",
        [2UL],
        [2U],
        Int32Values([1, 2]),
        source,
      )
      false
    } catch {
      InvalidArgument(_) => true
      _ => false
    },
  )

  let mismatched = Hdf5Writer::new()
  mismatched.add_chunked_dataset_from_source(
    "/values",
    [2UL],
    [2U],
    Int64Values([]),
    source,
  )
  assert_true(
    try {
      mismatched.encode() |> ignore
      false
    } catch {
      InvalidArgument(message) => message.contains("different datatype")
      _ => false
    },
  )
}

///|
test "chunk source writes through user-block sinks" {
  let source = {
    dimensions: [6UL],
    values: [3, 1, 4, 1, 5, 9],
    reads: 0,
    largest: 0,
    fail_at: -1,
  }
  let writer = Hdf5Writer::new()
  writer.set_user_block_content(b"source-prefix")
  writer.add_chunked_dataset_from_source(
    "/digits",
    [6UL],
    [2U],
    Int32Values([]),
    source,
    filters=[Deflate(4)],
  )
  let sink = Hdf5MemorySink::new()
  writer.write_to(sink)
  let bytes = sink.to_bytes()
  assert_eq(bytes.get_view(start=0, end=13).unwrap(), b"source-prefix")
  assert_eq(chunk_source_file_integers(bytes, "/digits"), [
    3L, 1L, 4L, 1L, 5L, 9L,
  ])
}
