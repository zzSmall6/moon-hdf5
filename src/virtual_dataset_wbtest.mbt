///|
fn virtual_dataset_integers(dataset : Dataset) -> Array[Int64] raise Hdf5Error {
  match dataset.values {
    Integers(values) => values
    _ => raise InvalidField("expected virtual dataset integer values")
  }
}

///|
fn virtual_dataset_floats(dataset : Dataset) -> Array[Double] raise Hdf5Error {
  match dataset.values {
    Floats(values) => values
    _ => raise InvalidField("expected virtual dataset floating values")
  }
}

///|
fn virtual_dataset_strings(dataset : Dataset) -> Array[String] raise Hdf5Error {
  match dataset.values {
    Strings(values) => values
    _ => raise InvalidField("expected virtual dataset string values")
  }
}

///|
fn virtual_dataset_enums(dataset : Dataset) -> Array[String] raise Hdf5Error {
  match dataset.values {
    Enums(values) => values
    _ => raise InvalidField("expected virtual dataset enum values")
  }
}

///|
fn virtual_dataset_structured(
  dataset : Dataset,
) -> Array[DatasetValue] raise Hdf5Error {
  match dataset.values {
    Structured(values) => values
    _ => raise InvalidField("expected virtual dataset structured values")
  }
}

///|
fn virtual_selection_integers(
  result : DatasetSelectionResult,
) -> Array[Int64] raise Hdf5Error {
  match result.values {
    Integers(values) => values
    _ => raise InvalidField("expected virtual selection integer values")
  }
}

///|
fn virtual_page_integers(page : DatasetSelectionPage) -> Array[Int64] {
  match page.values {
    Integers(values) => values
    _ => abort("expected virtual selection-page integers")
  }
}

///|
fn virtual_chunk_integers(chunk : DatasetChunk) -> Array[Int64] {
  match chunk.values {
    Integers(values) => values
    _ => abort("expected virtual chunk integers")
  }
}

///|
test "virtual datasets read same-file and registered external mappings" {
  let bytes = @fs.read_file_to_bytes("testdata/virtual_dataset.h5")
  let file = Hdf5File::parse(bytes)
  assert_eq(virtual_dataset_integers(file.read_dataset("/same_file")), [
    0L, 1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, 10L, 11L,
  ])
  let missing = try {
    file.read_dataset("/mapped") |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    missing,
    Some(
      InvalidField(
        "virtual dataset source file is not registered: virtual_dataset_source.h5",
      ),
    ),
  )

  let source = @fs.read_file_to_bytes("testdata/virtual_dataset_source.h5")
  let registered = file.with_external_source(
    "virtual_dataset_source.h5", source,
  )
  let mapped = registered.read_dataset("/mapped")
  assert_eq(mapped.dataspace.dimensions, [8UL, 10UL])
  assert_eq(mapped.layout.kind(), Virtual)
  assert_eq(virtual_dataset_integers(mapped), [
    0L, 1L, 2L, 3L, 4L, -1L, -1L, -1L, -1L, -1L, 5L, 6L, 7L, 8L, 9L, -1L, -1L, -1L,
    -1L, -1L, 10L, 11L, 12L, 13L, 14L, -1L, -1L, -1L, -1L, -1L, 15L, 16L, 17L, 18L,
    19L, -1L, -1L, -1L, -1L, -1L, -1L, -1L, -1L, -1L, -1L, 0L, 1L, 2L, 3L, 4L, -1L,
    -1L, -1L, -1L, -1L, 5L, 6L, 7L, 8L, 9L, -1L, -1L, -1L, -1L, -1L, 10L, 11L, 12L,
    13L, 14L, -1L, -1L, -1L, -1L, -1L, 15L, 16L, 17L, 18L, 19L,
  ])
  assert_eq(virtual_dataset_integers(registered.read_dataset("/strided")), [
    100L, 102L, 104L, 112L, 114L, 116L, 124L, 126L, 128L,
  ])
  assert_eq(
    virtual_dataset_integers(registered.read_dataset("/multiple_mappings")),
    [0L, 1L, 2L, 3L, 4L, 5L],
  )
}

///|
test "virtual datasets support selections, tiles and selection pages" {
  let file = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/virtual_dataset.h5"),
  ).with_external_source(
    "virtual_dataset_source.h5",
    @fs.read_file_to_bytes("testdata/virtual_dataset_source.h5"),
  )
  let slab = file.read_dataset_selection(
    "/mapped",
    Hyperslab([3UL, 3UL], [2UL, 2UL], [2UL, 3UL]),
  )
  assert_eq(slab.dimensions, [2UL, 2UL])
  assert_eq(virtual_selection_integers(slab), [18L, -1L, -1L, 6L])
  let points = file.read_dataset_selection(
    "/mapped",
    Points([[0UL, 4UL], [4UL, 5UL], [7UL, 9UL], [2UL, 0UL]]),
  )
  assert_eq(virtual_selection_integers(points), [4L, 0L, 19L, 10L])

  let chunks = file.chunk_iterator("/mapped", [3UL, 4UL])
  let first = chunks.next().unwrap()
  assert_eq(first.offset, [0UL, 0UL])
  assert_eq(first.dimensions, [3UL, 4UL])
  assert_eq(virtual_chunk_integers(first), [
    0L, 1L, 2L, 3L, 5L, 6L, 7L, 8L, 10L, 11L, 12L, 13L,
  ])
  let second = chunks.next().unwrap()
  assert_eq(second.offset, [0UL, 4UL])
  assert_eq(second.dimensions, [3UL, 4UL])
  assert_eq(virtual_chunk_integers(second), [
    4L, -1L, -1L, -1L, 9L, -1L, -1L, -1L, 14L, -1L, -1L, -1L,
  ])

  let pages = file.selection_iterator(
    "/mapped",
    Points([[0UL, 0UL], [0UL, 5UL], [4UL, 0UL], [4UL, 5UL], [7UL, 9UL]]),
    3,
  )
  let first_page = pages.next().unwrap()
  assert_eq(first_page.offset, 0UL)
  assert_eq(virtual_page_integers(first_page), [0L, -1L, -1L])
  let second_page = pages.next().unwrap()
  assert_eq(second_page.offset, 3UL)
  assert_eq(virtual_page_integers(second_page), [0L, 19L])
  assert_true(pages.next() is None)
}

///|
test "virtual datasets resolve sources and four-byte addresses" {
  let resolver = Hdf5Resolver::new(
    @fs.read_file_to_bytes("testdata/virtual_dataset.h5"),
  )
  resolver.add_source(
    "virtual_dataset_source.h5",
    @fs.read_file_to_bytes("testdata/virtual_dataset_source.h5"),
  )
  assert_eq(resolver.read_dataset("/mapped").layout.kind(), Virtual)
  let selected = resolver.read_dataset_selection(
    "/mapped",
    Hyperslab([4UL, 5UL], [4UL, 5UL], [1UL, 1UL]),
  )
  assert_eq(virtual_selection_integers(selected), [
    0L, 1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, 10L, 11L, 12L, 13L, 14L, 15L, 16L, 17L,
    18L, 19L,
  ])

  let main_32 = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/virtual_dataset_32.h5"),
  ).with_external_source(
    "virtual_dataset_source_32.h5",
    @fs.read_file_to_bytes("testdata/virtual_dataset_source_32.h5"),
  )
  assert_eq(virtual_dataset_integers(main_32.read_dataset("/mapped")), [
    200L, 201L, 202L, 203L, -9L, -9L, 204L, 205L, 206L, 207L, -9L, -9L, 208L, 209L,
    210L, 211L, -9L, -9L, -9L, -9L, -9L, -9L, -9L, -9L,
  ])
}

///|
test "virtual datasets can map another virtual dataset" {
  let bytes = @fs.read_file_to_bytes("testdata/virtual_dataset_nested.h5")
  let nested = Hdf5File::parse(bytes)
    .with_external_source(
      "virtual_dataset.h5",
      @fs.read_file_to_bytes("testdata/virtual_dataset.h5"),
    )
    .with_external_source(
      "virtual_dataset_source.h5",
      @fs.read_file_to_bytes("testdata/virtual_dataset_source.h5"),
    )
  assert_eq(virtual_dataset_integers(nested.read_dataset("/nested")), [
    0L, 1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, 10L, 11L, 12L, 13L, 14L, 15L, 16L, 17L,
    18L, 19L,
  ])

  let resolver = Hdf5Resolver::new(bytes)
  resolver.add_source(
    "virtual_dataset.h5",
    @fs.read_file_to_bytes("testdata/virtual_dataset.h5"),
  )
  resolver.add_source(
    "virtual_dataset_source.h5",
    @fs.read_file_to_bytes("testdata/virtual_dataset_source.h5"),
  )
  assert_eq(virtual_dataset_integers(resolver.read_dataset("/nested")), [
    0L, 1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, 10L, 11L, 12L, 13L, 14L, 15L, 16L, 17L,
    18L, 19L,
  ])
}

///|
test "virtual datasets address large direct mappings without expansion" {
  let dimensions = [1000000UL, 1000000UL]
  let dataspace = {
    kind: Simple,
    dimensions,
    maximum_dimensions: dimensions.copy(),
  }
  let hyperslab = RegionHyperslab([0UL, 0UL], [1UL, 1UL], [1000UL, 1000UL], [
    1UL, 1UL,
  ])
  assert_true(virtual_selection_is_direct(hyperslab))
  assert_eq(virtual_selection_count(dataspace, hyperslab), 1000000L)
  let target = coordinate_linear_index([500UL, 500UL], dimensions)
  assert_eq(
    virtual_selection_ordinal_of(dataspace, hyperslab, target),
    Some(500500L),
  )
  assert_eq(virtual_selection_index_at(dataspace, hyperslab, 500500L), target)

  let all = RegionAll
  assert_eq(virtual_selection_count(dataspace, all), 1000000000000L)
  assert_eq(
    virtual_selection_ordinal_of(dataspace, all, 999999999999UL),
    Some(999999999999L),
  )
}

///|
test "virtual datasets expand printf names and convert datatypes" {
  let printf_file = Hdf5File::parse(
      @fs.read_file_to_bytes("testdata/virtual_dataset_printf.h5"),
    )
    .with_external_source(
      "virtual_printf_source_0.h5",
      @fs.read_file_to_bytes("testdata/virtual_printf_source_0.h5"),
    )
    .with_external_source(
      "virtual_printf_source_1.h5",
      @fs.read_file_to_bytes("testdata/virtual_printf_source_1.h5"),
    )
  let grid = printf_file.read_dataset("/printf_grid")
  assert_eq(grid.dataspace.dimensions, [4UL, 4UL])
  assert_eq(grid.dataspace.maximum_dimensions, [virtual_unlimited, 4UL])
  assert_eq(virtual_dataset_integers(grid), [
    0L, 1L, 2L, 3L, 4L, 5L, 6L, 7L, 100L, 101L, 102L, 103L, 104L, 105L, 106L, 107L,
  ])
  assert_eq(
    virtual_dataset_integers(printf_file.read_dataset("/literal_percent")),
    [0L, 1L, 2L, 3L],
  )
  let dataset_printf = printf_file.read_dataset("/printf_dataset")
  assert_eq(dataset_printf.dataspace.maximum_dimensions, [virtual_unlimited])
  assert_eq(virtual_dataset_integers(dataset_printf), [10L, 11L, 20L, 21L])
  let selected = printf_file.read_dataset_selection(
    "/printf_grid",
    Hyperslab([3UL, 1UL], [1UL, 2UL], [1UL, 1UL]),
  )
  assert_eq(virtual_selection_integers(selected), [105L, 106L])

  let conversion = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/virtual_dataset_conversion.h5"),
  )
  assert_eq(virtual_dataset_integers(conversion.read_dataset("/i16_i64")), [
    1L, -2L, 3L, 4L, 5L, -6L,
  ])
  assert_eq(virtual_dataset_floats(conversion.read_dataset("/i16_f64")), [
    1.0, -2.0, 3.0, 4.0, 5.0, -6.0,
  ])
  assert_eq(virtual_dataset_integers(conversion.read_dataset("/f32_i32")), [
    1L, -2L, 3L, -4L, 5L, -6L,
  ])
  assert_eq(virtual_dataset_strings(conversion.read_dataset("/s3_s8")), [
    "ab", "c", "def",
  ])
  assert_eq(virtual_dataset_integers(conversion.read_dataset("/enum_to_int")), [
    1L, 2L,
  ])
  assert_eq(virtual_dataset_enums(conversion.read_dataset("/int_to_enum")), [
    "ONE", "TWO",
  ])
  assert_eq(virtual_dataset_enums(conversion.read_dataset("/enum_to_enum")), [
    "ONE", "TWO",
  ])
  assert_eq(
    virtual_dataset_structured(conversion.read_dataset("/cmp_i16_i32")),
    [
      Compound([
        { name: "a", value: SignedInteger(1L), },
        { name: "b", value: SignedInteger(2L), },
      ]),
      Compound([
        { name: "a", value: SignedInteger(3L), },
        { name: "b", value: SignedInteger(4L), },
      ]),
    ],
  )
  assert_eq(
    virtual_dataset_structured(conversion.read_dataset("/cmp_missing")),
    [
      Compound([
        { name: "a", value: SignedInteger(1L), },
        { name: "b", value: SignedInteger(0L), },
      ]),
      Compound([
        { name: "a", value: SignedInteger(3L), },
        { name: "b", value: SignedInteger(0L), },
      ]),
    ],
  )
  assert_eq(virtual_dataset_structured(conversion.read_dataset("/cmp_extra")), [
    Compound([{ name: "a", value: SignedInteger(1L), }]),
    Compound([{ name: "a", value: SignedInteger(3L), }]),
  ])
  assert_eq(
    virtual_dataset_structured(conversion.read_dataset("/cmp_reorder")),
    [
      Compound([
        { name: "b", value: SignedInteger(2L), },
        { name: "a", value: SignedInteger(1L), },
      ]),
      Compound([
        { name: "b", value: SignedInteger(4L), },
        { name: "a", value: SignedInteger(3L), },
      ]),
    ],
  )
  assert_eq(
    virtual_dataset_structured(conversion.read_dataset("/cmp_nested")),
    [
      Compound([
        {
          name: "p",
          value: Compound([
            { name: "x", value: SignedInteger(1L), },
            { name: "y", value: FloatingPoint(2.5), },
          ]),
        },
      ]),
      Compound([
        {
          name: "p",
          value: Compound([
            { name: "x", value: SignedInteger(3L), },
            { name: "y", value: FloatingPoint(4.5), },
          ]),
        },
      ]),
    ],
  )
  assert_eq(
    virtual_dataset_structured(conversion.read_dataset("/cmp_string")),
    [
      Compound([
        { name: "s", value: Text("ab"), },
        { name: "n", value: SignedInteger(1L), },
      ]),
      Compound([
        { name: "s", value: Text("c"), },
        { name: "n", value: SignedInteger(2L), },
      ]),
    ],
  )
  assert_eq(virtual_dataset_structured(conversion.read_dataset("/cmp_array")), [
    Compound([
      { name: "q", value: Elements([SignedInteger(1L), SignedInteger(2L)]), },
    ]),
    Compound([
      { name: "q", value: Elements([SignedInteger(3L), SignedInteger(4L)]), },
    ]),
  ])
  assert_eq(
    virtual_dataset_structured(conversion.read_dataset("/complex64_128")),
    [
      Compound([
        { name: "r", value: FloatingPoint(1.0), },
        { name: "i", value: FloatingPoint(2.0), },
      ]),
      Compound([
        { name: "r", value: FloatingPoint(3.0), },
        { name: "i", value: FloatingPoint(4.0), },
      ]),
    ],
  )
}

///|
test "virtual complex datatype conversion encodes both components" {
  let component32 = Datatype::write_float(4)
  let component64 = Datatype::write_float(8)
  let source : Datatype = {
    ..component32,
    class: 11,
    size: 8,
    base_types: [component32],
  }
  let target : Datatype = {
    ..component64,
    class: 11,
    size: 16,
    base_types: [component64],
  }
  assert_true(virtual_datatype_convertible(source, target))
  let encoded = encode_virtual_value(source, target, Complex(1.25, -2.5))
  assert_eq(
    decode_dataset_value(None, b"", encoded, 0, target, 8, 8, 0UL),
    Complex(1.25, -2.5),
  )
}

///|
test "virtual datasets can be written to new files" {
  let writer = Hdf5Writer::new()
  writer.add_dataset(
    "/source",
    [3UL, 4UL],
    Int32Values([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
  )
  let datatype = Datatype::write_integer(true, 4)
  writer.add_virtual_dataset("/same", [3UL, 4UL], datatype, [
    {
      source_file: ".",
      source_dataset: "/source",
      source_selection: RegionAll,
      virtual_selection: RegionAll,
    },
  ])
  writer.add_virtual_dataset(
    "/mapped",
    [2UL, 3UL],
    datatype,
    [
      {
        source_file: ".",
        source_dataset: "/source",
        source_selection: RegionHyperslab([1UL, 1UL], [1UL, 1UL], [2UL, 2UL], [
          1UL, 1UL,
        ]),
        virtual_selection: RegionHyperslab([0UL, 0UL], [1UL, 1UL], [2UL, 2UL], [
          1UL, 1UL,
        ]),
      },
    ],
    fill_value=SignedInteger(-1L),
  )
  writer.add_virtual_dataset(
    "/empty",
    [2UL, 2UL],
    datatype,
    [
      {
        source_file: ".",
        source_dataset: "/source",
        source_selection: RegionEmpty,
        virtual_selection: RegionEmpty,
      },
    ],
    fill_value=SignedInteger(-5L),
  )

  let file = Hdf5File::parse(writer.encode())
  assert_eq(virtual_dataset_integers(file.read_dataset("/same")), [
    1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, 10L, 11L, 12L,
  ])
  let mapped = file.read_dataset("/mapped")
  assert_eq(mapped.layout.kind(), Virtual)
  assert_eq(virtual_dataset_integers(mapped), [6L, 7L, -1L, 10L, 11L, -1L])
  assert_eq(virtual_dataset_integers(file.read_dataset("/empty")), [
    -5L, -5L, -5L, -5L,
  ])
  let metadata = file.read_dataset_metadata("/mapped")
  assert_eq(metadata.datatype.class, 0)
  assert_eq(metadata.dataspace.dimensions, [2UL, 3UL])
}

///|
test "virtual dataset writer emits unlimited printf mappings" {
  let writer = Hdf5Writer::new()
  writer.add_dataset("/block_0", [2UL], Int32Values([10, 11]))
  writer.add_dataset("/block_1", [2UL], Int32Values([20, 21]))
  writer.add_dataset("/literal%value", [2UL], Int32Values([7, 8]))
  writer.add_dataset("/int16_source", [2UL], Int16Values([1, -2]))
  let datatype = Datatype::write_integer(true, 4)
  writer.add_virtual_dataset(
    "/printf",
    [4UL],
    datatype,
    [
      {
        source_file: ".",
        source_dataset: "/block_%b",
        source_selection: RegionAll,
        virtual_selection: RegionHyperslab([0UL], [2UL], [virtual_unlimited], [
          2UL,
        ]),
      },
    ],
    maximum_dimensions=[None],
  )
  writer.add_virtual_dataset(
    "/printf_file",
    [2UL],
    datatype,
    [
      {
        source_file: "source_%b.h5",
        source_dataset: "/data",
        source_selection: RegionAll,
        virtual_selection: RegionHyperslab([0UL], [1UL], [virtual_unlimited], [
          1UL,
        ]),
      },
    ],
    maximum_dimensions=[None],
  )
  writer.add_virtual_dataset("/literal", [2UL], datatype, [
    {
      source_file: ".",
      source_dataset: "/literal%%value",
      source_selection: RegionAll,
      virtual_selection: RegionAll,
    },
  ])
  writer.add_virtual_dataset("/converted", [2UL], datatype, [
    {
      source_file: ".",
      source_dataset: "/int16_source",
      source_selection: RegionAll,
      virtual_selection: RegionAll,
    },
  ])

  let file = Hdf5File::parse(writer.encode())
  let dataset = file.read_dataset("/printf")
  assert_eq(dataset.dataspace.maximum_dimensions, [virtual_unlimited])
  assert_eq(virtual_dataset_integers(dataset), [10L, 11L, 20L, 21L])
  assert_eq(
    file.read_dataset_metadata("/printf_file").dataspace.maximum_dimensions,
    [virtual_unlimited],
  )
  assert_eq(virtual_dataset_integers(file.read_dataset("/literal")), [7L, 8L])
  assert_eq(virtual_dataset_integers(file.read_dataset("/converted")), [1L, -2L])
}

///|
test "virtual dataset writer rejects unsupported mapping selections" {
  let writer = Hdf5Writer::new()
  writer.add_dataset("/source", [2UL, 2UL], Int32Values([1, 2, 3, 4]))
  let datatype = Datatype::write_integer(true, 4)
  let points_error = try {
    writer.add_virtual_dataset("/points", [2UL, 2UL], datatype, [
      {
        source_file: ".",
        source_dataset: "/source",
        source_selection: RegionPoints([[0UL, 0UL], [1UL, 1UL]]),
        virtual_selection: RegionPoints([[0UL, 0UL], [1UL, 1UL]]),
      },
    ])
    None
  } catch {
    error => Some(error)
  }
  assert_true(points_error is Some(UnsupportedFeature(_)))

  let blocks_error = try {
    writer.add_virtual_dataset("/blocks", [2UL, 2UL], datatype, [
      {
        source_file: ".",
        source_dataset: "/source",
        source_selection: RegionBlocks([{ start: [0UL, 0UL], end: [1UL, 1UL], }]),
        virtual_selection: RegionBlocks([
          { start: [0UL, 0UL], end: [1UL, 1UL], },
        ]),
      },
    ])
    None
  } catch {
    error => Some(error)
  }
  assert_true(blocks_error is Some(UnsupportedFeature(_)))
}

///|
test "virtual dataset writer validates printf mappings" {
  let writer = Hdf5Writer::new()
  writer.add_dataset("/source", [2UL], Int32Values([1, 2]))
  let datatype = Datatype::write_integer(true, 4)
  let invalid_specifier = try {
    writer.add_virtual_dataset("/invalid_specifier", [2UL], datatype, [
      {
        source_file: ".",
        source_dataset: "/source%q",
        source_selection: RegionAll,
        virtual_selection: RegionAll,
      },
    ])
    None
  } catch {
    error => Some(error)
  }
  assert_true(invalid_specifier is Some(InvalidField(_)))

  let printf_without_unlimited = try {
    writer.add_virtual_dataset("/printf_without_unlimited", [2UL], datatype, [
      {
        source_file: ".",
        source_dataset: "/source_%b",
        source_selection: RegionAll,
        virtual_selection: RegionAll,
      },
    ])
    None
  } catch {
    error => Some(error)
  }
  assert_true(printf_without_unlimited is Some(InvalidArgument(_)))

  let unlimited_without_printf = try {
    writer.add_virtual_dataset(
      "/unlimited_without_printf",
      [2UL],
      datatype,
      [
        {
          source_file: ".",
          source_dataset: "/source",
          source_selection: RegionAll,
          virtual_selection: RegionHyperslab(
            [0UL],
            [1UL],
            [virtual_unlimited],
            [1UL],
          ),
        },
      ],
      maximum_dimensions=[None],
    )
    None
  } catch {
    error => Some(error)
  }
  assert_true(unlimited_without_printf is Some(InvalidArgument(_)))

  let unlimited_source = try {
    writer.add_virtual_dataset("/unlimited_source", [2UL], datatype, [
      {
        source_file: ".",
        source_dataset: "/source",
        source_selection: RegionHyperslab([0UL], [1UL], [virtual_unlimited], [
          1UL,
        ]),
        virtual_selection: RegionAll,
      },
    ])
    None
  } catch {
    error => Some(error)
  }
  assert_true(unlimited_source is Some(UnsupportedFeature(_)))

  let limited_maximum = try {
    writer.add_virtual_dataset(
      "/limited_maximum",
      [2UL],
      datatype,
      [
        {
          source_file: ".",
          source_dataset: "/source_%b",
          source_selection: RegionAll,
          virtual_selection: RegionHyperslab(
            [0UL],
            [1UL],
            [virtual_unlimited],
            [1UL],
          ),
        },
      ],
      maximum_dimensions=[Some(2UL)],
    )
    None
  } catch {
    error => Some(error)
  }
  assert_true(limited_maximum is Some(InvalidArgument(_)))
}

///|
test "existing files can gain virtual datasets" {
  let writer = Hdf5Writer::new()
  writer.add_group("/group")
  writer.add_dataset("/group/source", [2UL, 2UL], Int32Values([1, 2, 3, 4]))
  let file = Hdf5File::parse(writer.encode())
  let created = file.add_virtual_dataset(
    "/group",
    "mapped",
    [3UL, 3UL],
    Datatype::write_integer(true, 4),
    [
      {
        source_file: ".",
        source_dataset: "/group/source",
        source_selection: RegionAll,
        virtual_selection: RegionHyperslab([1UL, 1UL], [1UL, 1UL], [2UL, 2UL], [
          1UL, 1UL,
        ]),
      },
    ],
    fill_value=SignedInteger(-1L),
  ) catch {
    error => fail("create virtual dataset: " + error.message())
  }
  let edited = Hdf5File::parse(created)
  let metadata = edited.read_dataset_metadata("/group/mapped")
  assert_eq(metadata.layout.kind(), Virtual)
  assert_eq(metadata.dataspace.dimensions, [3UL, 3UL])
  assert_eq(virtual_dataset_integers(edited.read_dataset("/group/mapped")), [
    -1L, -1L, -1L, -1L, 1L, 2L, -1L, 3L, 4L,
  ])
  assert_eq(edited.resolve("/group/source").kind, Dataset)
}

///|
test "existing four-byte files can gain virtual datasets" {
  let file = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/symbol_table_32.h5"),
  )
  let source = file.add_dataset(
    "/group",
    "virtual_source",
    [2UL],
    Int32Values([7, 8]),
  ) catch {
    error => fail("create virtual source: " + error.message())
  }
  let parsed_source = Hdf5File::parse(source)
  let created = parsed_source.add_virtual_dataset(
    "/group",
    "virtual_mapped",
    [2UL],
    Datatype::write_integer(true, 4),
    [
      {
        source_file: ".",
        source_dataset: "/group/virtual_source",
        source_selection: RegionAll,
        virtual_selection: RegionAll,
      },
    ],
  ) catch {
    error => fail("create four-byte virtual dataset: " + error.message())
  }
  let edited = Hdf5File::parse(created)
  assert_eq(edited.superblock.offset_size, 4)
  assert_eq(
    edited.read_dataset_metadata("/group/virtual_mapped").layout.kind(),
    Virtual,
  )
  assert_eq(
    virtual_dataset_integers(edited.read_dataset("/group/virtual_mapped")),
    [7L, 8L],
  )

  let reference_type = Datatype::write_object_reference_with_address_size(4) catch {
    error => fail("create four-byte reference datatype: " + error.message())
  }
  let references = edited.add_virtual_dataset(
    "/group",
    "virtual_references",
    [1UL],
    reference_type,
    [
      {
        source_file: ".",
        source_dataset: "/group/virtual_source",
        source_selection: RegionAll,
        virtual_selection: RegionAll,
      },
    ],
  ) catch {
    error =>
      fail("create four-byte virtual reference dataset: " + error.message())
  }
  let reference_metadata = Hdf5File::parse(references).read_dataset_metadata(
    "/group/virtual_references",
  )
  assert_eq(reference_metadata.datatype.class, 7)
  assert_eq(reference_metadata.datatype.size, 4)
}
