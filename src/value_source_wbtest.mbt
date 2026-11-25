///|
priv struct GeneratedInt32ValueSource {
  dimensions : Array[UInt64]
  values : Array[Int32]
  mut source_reads : Int32
  mut source_largest : Int32
  source_fail_at : Int32
}

///|
fn value_source_shape_count(dimensions : Array[UInt64]) -> Int64 {
  let mut count = 1L
  for dimension in dimensions {
    count = count * dimension.reinterpret_as_int64()
  }
  count
}

///|
impl Hdf5ValueSource for GeneratedInt32ValueSource with fn read_values(
  self : GeneratedInt32ValueSource,
  offset : Array[UInt64],
  dimensions : Array[UInt64],
) -> Hdf5WriteValues raise Hdf5Error {
  self.source_reads = self.source_reads + 1
  if self.source_fail_at == self.source_reads - 1 {
    raise InvalidArgument(
      "generated source rejected block " + self.source_reads.to_string(),
    )
  }
  guard offset.length() == self.dimensions.length() &&
    dimensions.length() == self.dimensions.length() else {
    raise InvalidField("source block rank changed")
  }
  let count = value_source_shape_count(dimensions)
  if count.to_int() > self.source_largest {
    self.source_largest = count.to_int()
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
        raise InvalidField("source block lies outside the dataset")
      }
      source_index = source_index * self.dimensions[axis].reinterpret_as_int64() +
        coordinate
    }
    items.push(self.values[source_index.to_int()])
  }
  Int32Values(items)
}

///|
priv struct GeneratedStringValueSource {
  values : Array[String]
  mut source_reads : Int32
}

///|
impl Hdf5ValueSource for GeneratedStringValueSource with fn read_values(
  self : GeneratedStringValueSource,
  offset : Array[UInt64],
  dimensions : Array[UInt64],
) -> Hdf5WriteValues raise Hdf5Error {
  self.source_reads = self.source_reads + 1
  guard offset.length() == 1 && dimensions.length() == 1 else {
    raise InvalidField("string source expects rank one blocks")
  }
  let items : Array[String] = []
  for index in 0..<dimensions[0].reinterpret_as_int64().to_int() {
    items.push(self.values[offset[0].reinterpret_as_int64().to_int() + index])
  }
  VariableStringValues(items)
}

///|
fn value_source_file_integers(
  bytes : Bytes,
  path : String,
) -> Array[Int64] raise Hdf5Error {
  match Hdf5File::parse(bytes).read_dataset(path).values {
    Integers(values) => values
    _ => raise InvalidField("expected integer dataset values")
  }
}

///|
fn value_source_file_strings(
  bytes : Bytes,
  path : String,
) -> Array[String] raise Hdf5Error {
  match Hdf5File::parse(bytes).read_dataset(path).values {
    Strings(values) => values
    _ => raise InvalidField("expected string dataset values")
  }
}

///|
test "contiguous value sources match in-memory edge-clipped blocks" {
  let dimensions = [5UL, 7UL]
  let values = Array::make(35, 0)
  for index in 0..<values.length() {
    values[index] = index - 10
  }
  let source = {
    dimensions,
    values,
    source_reads: 0,
    source_largest: 0,
    source_fail_at: -1,
  }
  let source_writer = Hdf5Writer::new()
  source_writer.add_dataset_from_source(
    "/values",
    dimensions,
    [2U, 3U],
    Int32Values([]),
    source,
    fill_value=Int32Values([-7]),
  )
  let direct_writer = Hdf5Writer::new()
  direct_writer.add_dataset(
    "/values",
    dimensions,
    Int32Values(values),
    fill_value=Int32Values([-7]),
  )
  assert_eq(
    value_source_file_integers(source_writer.encode(), "/values"),
    value_source_file_integers(direct_writer.encode(), "/values"),
  )
  assert_true(source.source_reads > 0 && source.source_reads <= 35 * 4)
  assert_true(source.source_largest <= 6)
}

///|
test "contiguous value sources cover scalars and empty dataspaces" {
  let scalar_source = {
    dimensions: [],
    values: [42],
    source_reads: 0,
    source_largest: 0,
    source_fail_at: -1,
  }
  let scalar = Hdf5Writer::new()
  scalar.add_dataset_from_source(
    "/scalar",
    [],
    [],
    Int32Values([]),
    scalar_source,
  )
  assert_eq(value_source_file_integers(scalar.encode(), "/scalar"), [42L])

  let empty_source = {
    dimensions: [0UL, 3UL],
    values: Array::make(0, 0),
    source_reads: 0,
    source_largest: 0,
    source_fail_at: -1,
  }
  let empty = Hdf5Writer::new()
  empty.add_dataset_from_source(
    "/empty",
    [0UL, 3UL],
    [2U, 2U],
    Int32Values([]),
    empty_source,
  )
  assert_eq(value_source_file_integers(empty.encode(), "/empty"), [])
  assert_eq(empty_source.source_reads, 0)
}

///|
test "variable-length contiguous sources build the global heap incrementally" {
  let values = ["alpha", "", "数据", "alpha", "beta", "", "数据"]
  let source = { values, source_reads: 0, }
  let writer = Hdf5Writer::new()
  writer.add_dataset_from_source(
    "/names",
    [7UL],
    [3U],
    VariableStringValues([]),
    source,
  )
  assert_eq(value_source_file_strings(writer.encode(), "/names"), [
    "alpha", "", "数据", "alpha", "beta", "", "数据",
  ])
  assert_true(source.source_reads > 0)
}

///|
test "value source failures propagate and streamed datasets are immutable" {
  let failing = {
    dimensions: [4UL],
    values: [1, 2, 3, 4],
    source_reads: 0,
    source_largest: 0,
    source_fail_at: 1,
  }
  let writer = Hdf5Writer::new()
  writer.add_dataset_from_source(
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
      InvalidArgument(message) => message.contains("rejected block 2")
      _ => false
    },
  )

  let source = {
    dimensions: [4UL],
    values: [1, 2, 3, 4],
    source_reads: 0,
    source_largest: 0,
    source_fail_at: -1,
  }
  let immutable = Hdf5Writer::new()
  immutable.add_dataset_from_source(
    "/values",
    [4UL],
    [2U],
    Int32Values([]),
    source,
  )
  assert_true(
    try {
      immutable.resize_dataset("/values", [6UL])
      false
    } catch {
      UnsupportedFeature(_) => true
      _ => false
    },
  )
  assert_true(
    try {
      immutable.write_dataset_slab("/values", [0UL], [1UL], Int32Values([9]))
      false
    } catch {
      UnsupportedFeature(_) => true
      _ => false
    },
  )
}

///|
test "value source validates prototype, block shape and returned datatype" {
  let source = {
    dimensions: [2UL],
    values: [1, 2],
    source_reads: 0,
    source_largest: 0,
    source_fail_at: -1,
  }
  let writer = Hdf5Writer::new()
  assert_true(
    try {
      writer.add_dataset_from_source(
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
  assert_true(
    try {
      writer.add_dataset_from_source(
        "/bad-blocks",
        [2UL],
        [0U],
        Int32Values([]),
        source,
      )
      false
    } catch {
      InvalidArgument(_) => true
      _ => false
    },
  )

  let mismatched = Hdf5Writer::new()
  mismatched.add_dataset_from_source(
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
test "contiguous value sources write through user-block sinks" {
  let source = {
    dimensions: [6UL],
    values: [3, 1, 4, 1, 5, 9],
    source_reads: 0,
    source_largest: 0,
    source_fail_at: -1,
  }
  let writer = Hdf5Writer::new()
  writer.set_user_block_content(b"source-prefix")
  writer.add_dataset_from_source(
    "/digits",
    [6UL],
    [2U],
    Int32Values([]),
    source,
  )
  let sink = Hdf5MemorySink::new()
  writer.write_to(sink)
  let bytes = sink.to_bytes()
  assert_eq(bytes.get_view(start=0, end=13).unwrap(), b"source-prefix")
  assert_eq(value_source_file_integers(bytes, "/digits"), [
    3L, 1L, 4L, 1L, 5L, 9L,
  ])
}
