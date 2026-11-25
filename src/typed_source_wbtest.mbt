///|
priv struct TypedInt32Source {
  dimensions : Array[UInt64]
  values : Array[Int32]
  mut typed_reads : Int32
  mut typed_largest : Int32
}

///|
fn typed_source_shape_count(dimensions : Array[UInt64]) -> Int64 {
  let mut count = 1L
  for dimension in dimensions {
    count = count * dimension.reinterpret_as_int64()
  }
  count
}

///|
fn typed_source_values(
  source : TypedInt32Source,
  offset : Array[UInt64],
  dimensions : Array[UInt64],
) -> Hdf5WriteValues raise Hdf5Error {
  source.typed_reads = source.typed_reads + 1
  guard offset.length() == source.dimensions.length() &&
    dimensions.length() == source.dimensions.length() else {
    raise InvalidField("typed source block rank changed")
  }
  let count = typed_source_shape_count(dimensions)
  if count.to_int() > source.typed_largest {
    source.typed_largest = count.to_int()
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
      guard coordinate < source.dimensions[axis].reinterpret_as_int64() else {
        raise InvalidField("typed source block lies outside the dataset")
      }
      source_index = source_index *
        source.dimensions[axis].reinterpret_as_int64() +
        coordinate
    }
    items.push(source.values[source_index.to_int()])
  }
  Int32Values(items)
}

///|
impl Hdf5ChunkSource for TypedInt32Source with fn read_chunk(
  self : TypedInt32Source,
  chunk_index : Int32,
  offset : Array[UInt64],
  dimensions : Array[UInt64],
) -> Hdf5WriteValues raise Hdf5Error {
  ignore(chunk_index)
  typed_source_values(self, offset, dimensions)
}

///|
impl Hdf5ValueSource for TypedInt32Source with fn read_values(
  self : TypedInt32Source,
  offset : Array[UInt64],
  dimensions : Array[UInt64],
) -> Hdf5WriteValues raise Hdf5Error {
  typed_source_values(self, offset, dimensions)
}

///|
priv struct TypedStringSource {
  values : Array[String]
  mut typed_string_reads : Int32
}

///|
impl Hdf5ValueSource for TypedStringSource with fn read_values(
  self : TypedStringSource,
  offset : Array[UInt64],
  dimensions : Array[UInt64],
) -> Hdf5WriteValues raise Hdf5Error {
  self.typed_string_reads = self.typed_string_reads + 1
  guard offset.length() == 1 && dimensions.length() == 1 else {
    raise InvalidField("typed string source expects rank one blocks")
  }
  let items : Array[String] = []
  for index in 0..<dimensions[0].reinterpret_as_int64().to_int() {
    items.push(self.values[offset[0].reinterpret_as_int64().to_int() + index])
  }
  VariableStringValues(items)
}

///|
fn typed_source_dataset(
  bytes : Bytes,
  path : String,
) -> Dataset raise Hdf5Error {
  Hdf5File::parse(bytes).read_dataset(path)
}

///|
fn typed_source_integers(dataset : Dataset) -> Array[Int64] {
  match dataset.values {
    Integers(items) => items
    _ => abort("expected typed source integer values")
  }
}

///|
fn typed_source_strings(dataset : Dataset) -> Array[String] {
  match dataset.values {
    Strings(items) => items
    _ => abort("expected typed source string values")
  }
}

///|
test "typed contiguous sources convert blocks to the file datatype" {
  let dimensions = [5UL, 7UL]
  let values = Array::make(35, 0)
  for index in 0..<values.length() {
    values[index] = index - 10
  }
  let source = { dimensions, values, typed_reads: 0, typed_largest: 0, }
  let writer = Hdf5Writer::new()
  writer.add_dataset_from_source_typed(
    "/values",
    dimensions,
    [2U, 3U],
    Datatype::write_integer(true, 8),
    Int32Values([]),
    source,
    fill_value=Int32Values([-1]),
  )
  let dataset = typed_source_dataset(writer.encode(), "/values")
  assert_eq(dataset.datatype.class, 0)
  assert_eq(dataset.datatype.size, 8)
  assert_true(dataset.datatype.signed)
  assert_eq(
    typed_source_integers(dataset),
    values.map(value => value.to_int64()),
  )
  assert_true(source.typed_reads > 0 && source.typed_largest <= 6)
}

///|
test "typed chunk sources cover finite and extensible layouts" {
  let dimensions = [3UL, 4UL]
  let values = [11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 34]
  let finite_source = { dimensions, values, typed_reads: 0, typed_largest: 0, }
  let finite = Hdf5Writer::new()
  finite.add_chunked_dataset_from_source_typed(
    "/finite",
    dimensions,
    [2U, 2U],
    Datatype::write_integer(true, 8),
    Int32Values([]),
    finite_source,
    fill_value=Int32Values([-1]),
    filters=[Shuffle, Deflate(4)],
  )
  let finite_dataset = typed_source_dataset(finite.encode(), "/finite")
  assert_eq(finite_dataset.datatype.size, 8)
  assert_eq(typed_source_integers(finite_dataset), [
    11L, 12L, 13L, 14L, 21L, 22L, 23L, 24L, 31L, 32L, 33L, 34L,
  ])

  let extensible_source = {
    dimensions,
    values,
    typed_reads: 0,
    typed_largest: 0,
  }
  let extensible = Hdf5Writer::new()
  extensible.add_extensible_dataset_from_source_typed(
    "/extensible",
    dimensions,
    [None, None],
    [1U, 2U],
    Datatype::write_integer(true, 8),
    Int32Values([]),
    extensible_source,
    filters=[Deflate(4)],
  )
  let extensible_dataset = typed_source_dataset(
    extensible.encode(),
    "/extensible",
  )
  assert_eq(extensible_dataset.datatype.size, 8)
  assert_eq(typed_source_integers(extensible_dataset), [
    11L, 12L, 13L, 14L, 21L, 22L, 23L, 24L, 31L, 32L, 33L, 34L,
  ])
}

///|
test "typed value sources convert variable strings to fixed strings" {
  let source = {
    values: ["alpha", "beta", "", "gamma"],
    typed_string_reads: 0,
  }
  let writer = Hdf5Writer::new()
  writer.add_dataset_from_source_typed(
    "/names",
    [4UL],
    [2U],
    Datatype::write_fixed_string(8),
    VariableStringValues([]),
    source,
  )
  let dataset = typed_source_dataset(writer.encode(), "/names")
  assert_eq(dataset.datatype.class, 3)
  assert_eq(dataset.datatype.size, 8)
  assert_eq(typed_source_strings(dataset), ["alpha", "beta", "", "gamma"])
  assert_true(source.typed_string_reads > 0)
}

///|
test "typed sources enforce strict, lossy and no-conversion policies" {
  let strict_source = {
    dimensions: [1UL],
    values: [300],
    typed_reads: 0,
    typed_largest: 0,
  }
  let strict = Hdf5Writer::new()
  strict.add_dataset_from_source_typed(
    "/values",
    [1UL],
    [1U],
    Datatype::write_integer(true, 1),
    Int32Values([]),
    strict_source,
  )
  assert_true(
    try {
      strict.encode() |> ignore
      false
    } catch {
      InvalidArgument(message) => message.contains("out of range")
      _ => false
    },
  )

  let lossy_source = {
    dimensions: [1UL],
    values: [300],
    typed_reads: 0,
    typed_largest: 0,
  }
  let lossy = Hdf5Writer::new()
  lossy.add_dataset_from_source_typed(
    "/values",
    [1UL],
    [1U],
    Datatype::write_integer(true, 1),
    Int32Values([]),
    lossy_source,
    options=Hdf5ConversionOptions::lossy(),
  )
  let lossy_dataset = Hdf5File::parse(lossy.encode()).read_dataset("/values")
  assert_eq(typed_source_integers(lossy_dataset), [44L])

  let no_conversion = Hdf5Writer::new()
  assert_true(
    try {
      no_conversion.add_dataset_from_source_typed(
        "/values",
        [1UL],
        [1U],
        Datatype::write_integer(true, 1),
        Int32Values([]),
        lossy_source,
        options=Hdf5ConversionOptions::none(),
      )
      false
    } catch {
      InvalidArgument(message) => message.contains("identical")
      _ => false
    },
  )
}
