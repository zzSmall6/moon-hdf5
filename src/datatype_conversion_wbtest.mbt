///|
fn conversion_test_error(operation : () -> Unit raise Hdf5Error) -> Hdf5Error? {
  try {
    operation()
    None
  } catch {
    error => Some(error)
  }
}

///|
fn conversion_test_integers(values : DatasetValues) -> Array[Int64] {
  match values {
    Integers(items) => items
    _ => abort("expected converted signed integers")
  }
}

///|
fn conversion_test_unsigned(values : DatasetValues) -> Array[UInt64] {
  match values {
    UnsignedIntegers(items) => items
    _ => abort("expected converted unsigned integers")
  }
}

///|
fn conversion_test_floats(values : DatasetValues) -> Array[Double] {
  match values {
    Floats(items) => items
    _ => abort("expected converted floating-point values")
  }
}

///|
fn conversion_test_strings(values : DatasetValues) -> Array[String] {
  match values {
    Strings(items) => items
    _ => abort("expected converted strings")
  }
}

///|
fn conversion_test_enums(values : DatasetValues) -> Array[String] {
  match values {
    Enums(items) => items
    _ => abort("expected converted enumerations")
  }
}

///|
fn conversion_test_structured(values : DatasetValues) -> Array[DatasetValue] {
  match values {
    Structured(items) => items
    _ => abort("expected converted structured values")
  }
}

///|
fn conversion_test_big(value : Int32) -> @bigint.BigInt {
  @bigint.BigInt::from_int(1).shl(value)
}

///|
test "numeric conversion covers ranges precision and wide integers" {
  let int32_type = Datatype::write_integer(true, 4)
  let int8_type = Datatype::write_integer(true, 1)
  let uint8_type = Datatype::write_integer(false, 1)
  let float32_type = Datatype::write_float(4)
  let float64_type = Datatype::write_float(8)

  assert_eq(
    conversion_test_unsigned(
      int32_type.convert_values(
        Integers([1L, 2L, 127L]),
        Datatype::write_integer(false, 2),
      ),
    ),
    [1UL, 2UL, 127UL],
  )
  assert_eq(
    conversion_test_floats(
      int32_type.convert_values(Integers([1L, -2L, 127L]), float64_type),
    ),
    [1.0, -2.0, 127.0],
  )
  assert_eq(
    conversion_test_unsigned(
      int32_type.convert_values(
        Integers([300L]),
        uint8_type,
        options=Hdf5ConversionOptions::lossy(),
      ),
    ),
    [44UL],
  )
  assert_eq(
    conversion_test_integers(
      int32_type.convert_values(
        Integers([-129L]),
        int8_type,
        options=Hdf5ConversionOptions::lossy(),
      ),
    ),
    [127L],
  )
  assert_true(
    conversion_test_error(() => {
      int32_type.convert_values(Integers([200L]), int8_type) |> ignore
    })
    is Some(InvalidArgument("strict integer conversion is out of range")),
  )
  assert_eq(
    conversion_test_integers(
      float64_type.convert_values(
        Floats([1.75, -3.9]),
        Datatype::write_integer(true, 8),
        options=Hdf5ConversionOptions::lossy(),
      ),
    ),
    [1L, -3L],
  )
  assert_true(
    conversion_test_error(() => {
      float64_type.convert_values(
        Floats([1.5]),
        Datatype::write_integer(true, 8),
      )
      |> ignore
    })
    is Some(
      InvalidArgument(
        "strict conversion would discard the floating fractional part"
      )
    ),
  )
  assert_true(
    conversion_test_error(() => {
      int32_type.convert_values(Integers([16777217L]), float32_type) |> ignore
    })
    is Some(
      InvalidArgument("strict integer-to-float conversion would lose precision")
    ),
  )
  assert_eq(
    conversion_test_floats(
      int32_type.convert_values(
        Integers([16777217L]),
        float32_type,
        options=Hdf5ConversionOptions::lossy(),
      ),
    ),
    [16777216.0],
  )

  let wide_source = Datatype::write_integer(true, 16)
  let wide_target = Datatype::write_integer(true, 20)
  let value = conversion_test_big(80).neg()
  match wide_source.convert_value(BigInteger(value), wide_target) {
    BigInteger(converted) => assert_eq(converted, value)
    _ => abort("expected a wide converted integer")
  }
  assert_true(
    conversion_test_error(() => {
      wide_source.convert_values(
        BigIntegers([conversion_test_big(80)]),
        Datatype::write_integer(true, 8),
      )
      |> ignore
    })
    is Some(InvalidArgument("strict integer conversion is out of range")),
  )
}

///|
test "empty conversions still enforce no-conversion datatype identity" {
  let source = Datatype::write_integer(true, 4)
  let target = Datatype::write_integer(true, 8)
  let none = Hdf5ConversionOptions::none()
  assert_true(
    conversion_test_error(() => {
      source.convert_values(Integers([]), target, options=none) |> ignore
    })
    is Some(
      InvalidArgument(
        "no-conversion mode requires identical source and target datatypes"
      )
    ),
  )
  let values = StructuredValues(source, [])
  assert_true(
    conversion_test_error(() => {
      values.convert_to(target, options=none) |> ignore
    })
    is Some(
      InvalidArgument(
        "no-conversion mode requires identical source and target datatypes"
      )
    ),
  )
  assert_true(
    source.convert_values(Integers([]), source, options=none) is Integers([]),
  )
  let writer = Hdf5Writer::new()
  writer.add_dataset("/empty", [0UL], Int32Values([]))
  let file = Hdf5File::parse(writer.encode())
  assert_true(
    conversion_test_error(() => {
      file.chunk_iterator_as("/empty", [1UL], target, options=none) |> ignore
    })
    is Some(
      InvalidArgument(
        "no-conversion mode requires identical source and target datatypes"
      )
    ),
  )
  assert_true(
    conversion_test_error(() => {
      file.selection_iterator_as("/empty", Points([]), 1, target, options=none)
      |> ignore
    })
    is Some(
      InvalidArgument(
        "no-conversion mode requires identical source and target datatypes"
      )
    ),
  )
}

///|
test "conversion capability reflects strict and lossy policies" {
  let int8_type = Datatype::write_integer(true, 1)
  let int16_type = Datatype::write_integer(true, 2)
  let uint16_type = Datatype::write_integer(false, 2)
  assert_true(int8_type.can_convert_to(int16_type))
  assert_true(!int8_type.can_convert_to(uint16_type))
  assert_true(
    int8_type.can_convert_to(
      uint16_type,
      options=Hdf5ConversionOptions::lossy(),
    ),
  )
  assert_true(int8_type.can_convert_to(int8_type))
  assert_true(
    !int8_type.can_convert_to(int16_type, options=Hdf5ConversionOptions::none()),
  )
  assert_true(
    conversion_test_error(() => {
      int8_type.convert_values(
        Integers([1L]),
        int16_type,
        options=Hdf5ConversionOptions::none(),
      )
      |> ignore
    })
    is Some(
      InvalidArgument(
        "no-conversion mode requires identical source and target datatypes"
      )
    ),
  )
}

///|
test "string enum and opaque conversion enforce widths and members" {
  let fixed8 = Datatype::write_fixed_string(8)
  let vlen = Datatype::write_vlen_string()
  assert_eq(
    conversion_test_strings(
      fixed8.convert_values(Strings(["moon", "hdf5"]), vlen),
    ),
    ["moon", "hdf5"],
  )
  assert_eq(
    conversion_test_strings(
      vlen.convert_values(
        Strings(["a longer string"]),
        Datatype::write_fixed_string(4),
        options=Hdf5ConversionOptions::lossy(),
      ),
    ),
    ["a lo"],
  )
  assert_true(
    conversion_test_error(() => {
      vlen.convert_values(Strings(["long"]), Datatype::write_fixed_string(3))
      |> ignore
    })
    is Some(
      InvalidArgument("strict fixed-string conversion exceeds the target width")
    ),
  )

  let enum_type = Datatype::write_enum(Datatype::write_integer(true, 1), [
    { name: "ZERO", value: EnumValue::Signed(0L), },
    { name: "ONE", value: EnumValue::Signed(1L), },
    { name: "TWO", value: EnumValue::Signed(2L), },
  ])
  assert_eq(
    conversion_test_enums(
      Datatype::write_integer(true, 2).convert_values(
        Integers([1L, 2L]),
        enum_type,
      ),
    ),
    ["ONE", "TWO"],
  )
  assert_true(
    conversion_test_error(() => {
      Datatype::write_integer(true, 2).convert_values(Integers([9L]), enum_type)
      |> ignore
    })
    is Some(
      InvalidArgument(
        "conversion target enum has no member for the converted integer"
      )
    ),
  )
  assert_eq(
    conversion_test_integers(
      enum_type.convert_values(Enums(["TWO"]), Datatype::write_integer(true, 2)),
    ),
    [2L],
  )
  assert_true(enum_type.can_convert_to(Datatype::write_integer(true, 2)))

  let opaque4 = Datatype::write_opaque(4, "tag")
  let opaque6 = Datatype::write_opaque(6, "tag")
  let opaque2 = Datatype::write_opaque(2, "tag")
  assert_eq(
    conversion_test_structured(
      opaque4.convert_values(
        Structured([Opaque(b"ABCD")]),
        opaque6,
        options=Hdf5ConversionOptions::lossy(),
      ),
    ),
    [Opaque(b"ABCD\x00\x00")],
  )
  assert_eq(
    conversion_test_structured(
      opaque4.convert_values(
        Structured([Opaque(b"ABCD")]),
        opaque2,
        options=Hdf5ConversionOptions::lossy(),
      ),
    ),
    [Opaque(b"AB")],
  )
  assert_true(
    conversion_test_error(() => {
      opaque4.convert_values(Structured([Opaque(b"ABCD")]), opaque6) |> ignore
    })
    is Some(
      InvalidArgument(
        "strict opaque conversion requires equal source and target sizes"
      )
    ),
  )
}

///|
test "compound array sequence and complex conversion preserve structure" {
  let id32 = Datatype::write_integer(true, 4)
  let id64 = Datatype::write_integer(true, 8)
  let float64 = Datatype::write_float(8)
  let source_compound = Datatype::write_compound([
    ("id", id32),
    ("weight", float64),
    ("extra", Datatype::write_fixed_string(4)),
  ])
  let target_compound = Datatype::write_compound([
    ("weight", float64),
    ("id", id64),
    ("note", Datatype::write_fixed_string(6)),
  ])
  let source_value = Compound([
    { name: "id", value: SignedInteger(7L), },
    { name: "weight", value: FloatingPoint(1.5), },
    { name: "extra", value: Text("x"), },
  ])
  let options = {
    mode: Hdf5ConversionOptions::strict().mode,
    allow_missing_compound_members: true,
  }
  assert_true(source_compound.can_convert_to(target_compound, options~))
  assert_eq(
    source_compound.convert_value(source_value, target_compound, options~),
    Compound([
      { name: "weight", value: FloatingPoint(1.5), },
      { name: "id", value: SignedInteger(7L), },
      { name: "note", value: Text(""), },
    ]),
  )
  assert_true(!source_compound.can_convert_to(target_compound))
  assert_true(
    conversion_test_error(() => {
      source_compound.convert_value(source_value, target_compound) |> ignore
    })
    is Some(
      InvalidArgument("compound conversion is missing target member: note")
    ),
  )

  let source_array = Datatype::write_array(Datatype::write_integer(true, 2), [
    2UL, 2UL,
  ])
  let target_array = Datatype::write_array(float64, [2UL, 2UL])
  assert_eq(
    source_array.convert_value(
      Elements([
        SignedInteger(1L),
        SignedInteger(2L),
        SignedInteger(3L),
        SignedInteger(4L),
      ]),
      target_array,
    ),
    Elements([
      FloatingPoint(1.0),
      FloatingPoint(2.0),
      FloatingPoint(3.0),
      FloatingPoint(4.0),
    ]),
  )

  let source_sequence = Datatype::write_vlen_sequence(
    Datatype::write_integer(true, 2),
  )
  let target_sequence = Datatype::write_vlen_sequence(float64)
  assert_eq(
    source_sequence.convert_value(
      Sequence([SignedInteger(1L), SignedInteger(-2L)]),
      target_sequence,
    ),
    Sequence([FloatingPoint(1.0), FloatingPoint(-2.0)]),
  )

  let compound_complex = Datatype::write_compound([
    ("r", float64),
    ("i", float64),
  ])
  let complex_type = Datatype::write_complex(float64)
  assert_eq(
    compound_complex.convert_value(
      Compound([
        { name: "r", value: FloatingPoint(2.5), },
        { name: "i", value: FloatingPoint(-4.0), },
      ]),
      complex_type,
    ),
    Complex(2.5, -4.0),
  )
  assert_eq(
    complex_type.convert_value(Complex(3.25, -1.5), compound_complex),
    Compound([
      { name: "r", value: FloatingPoint(3.25), },
      { name: "i", value: FloatingPoint(-1.5), },
    ]),
  )
  let array_complex = Datatype::write_array(float64, [2UL])
  assert_eq(
    array_complex.convert_value(
      Elements([FloatingPoint(6.0), FloatingPoint(7.0)]),
      complex_type,
    ),
    Complex(6.0, 7.0),
  )
  assert_eq(
    complex_type.convert_value(Complex(8.0, -9.0), array_complex),
    Elements([FloatingPoint(8.0), FloatingPoint(-9.0)]),
  )
  let renamed_complex_compound = Datatype::write_compound([
    ("real", float64),
    ("imag", float64),
  ])
  assert_eq(
    array_complex.convert_value(
      Elements([FloatingPoint(10.0), FloatingPoint(11.0)]),
      renamed_complex_compound,
    ),
    Compound([
      { name: "real", value: FloatingPoint(10.0), },
      { name: "imag", value: FloatingPoint(11.0), },
    ]),
  )
  assert_eq(
    Datatype::write_complex(Datatype::write_float(4)).convert_value(
      Complex(1.5, -2.25),
      complex_type,
    ),
    Complex(1.5, -2.25),
  )
}

///|
test "missing compound members use zero-valued enum defaults" {
  let enum_type = Datatype::write_enum(Datatype::write_integer(true, 1), [
    { name: "NEGATIVE", value: EnumValue::Signed(-1L), },
    { name: "ZERO", value: EnumValue::Signed(0L), },
  ])
  let source = Datatype::write_compound([
    ("id", Datatype::write_integer(true, 4)),
  ])
  let target = Datatype::write_compound([
    ("id", Datatype::write_integer(true, 4)),
    ("state", enum_type),
  ])
  let options = {
    mode: Hdf5ConversionOptions::strict().mode,
    allow_missing_compound_members: true,
  }
  assert_true(source.can_convert_to(target, options~))
  assert_eq(
    source.convert_value(
      Compound([{ name: "id", value: SignedInteger(7L), }]),
      target,
      options~,
    ),
    Compound([
      { name: "id", value: SignedInteger(7L), },
      { name: "state", value: Enumeration("ZERO"), },
    ]),
  )
  let no_zero_enum = Datatype::write_enum(Datatype::write_integer(true, 1), [
    { name: "ONE", value: EnumValue::Signed(1L), },
  ])
  let unsupported_target = Datatype::write_compound([
    ("id", Datatype::write_integer(true, 4)),
    ("state", no_zero_enum),
  ])
  assert_true(!source.can_convert_to(unsupported_target, options~))
}

///|
test "time bitfield and reference conversion preserve target kinds" {
  let int32_type = Datatype::write_integer(true, 4)
  assert_eq(
    int32_type.convert_value(SignedInteger(12L), Datatype::write_time(8)),
    Time(12UL),
  )
  assert_eq(
    int32_type.convert_value(SignedInteger(5L), Datatype::write_bitfield(2, 12)),
    Bitfield(5UL),
  )
  let object_type = Datatype::write_object_reference()
  assert_eq(
    object_type.convert_value(ObjectReference(Some("/target")), object_type),
    ObjectReference(Some("/target")),
  )
  let raw = b"\x01\x02\x03\x04\x05\x06\x07\x08"
  assert_eq(
    object_type.convert_value(Reference(raw), object_type),
    Reference(raw),
  )
  let region_type = Datatype::write_region_reference()
  assert_eq(
    object_type.convert_value(
      ObjectReference(Some("/target")),
      Datatype::write_version4_object_reference(),
    ),
    ObjectReference(Some("/target")),
  )
  assert_eq(
    region_type.convert_value(
      RegionReference(Some("/target"), RegionAll),
      region_type,
    ),
    RegionReference(Some("/target"), RegionAll),
  )
}

///|
test "typed writer and reader APIs convert across storage layouts" {
  let writer = Hdf5Writer::new()
  writer.add_group("/g")
  writer.add_dataset_typed(
    "/g/contiguous",
    [3UL],
    Datatype::write_integer(false, 2),
    Int64Values([1L, 2L, 3L]),
  )
  writer.add_compact_dataset_typed(
    "/g/compact",
    [2UL],
    Datatype::write_float(4),
    Int64Values([4L, 5L]),
  )
  writer.add_chunked_dataset_typed(
    "/g/chunked",
    [4UL],
    [2U],
    Datatype::write_integer(false, 8),
    UInt16Values([10U, 11U, 12U, 13U]),
    fill_value=UInt16Values([9U]),
    filters=[Shuffle],
  )
  writer.add_extensible_dataset_typed(
    "/g/extensible",
    [2UL],
    [None],
    [2U],
    Datatype::write_float(8),
    Int32Values([1, 2]),
  )
  writer.add_attribute_typed(
    "/g/contiguous",
    "weights",
    [3UL],
    Datatype::write_float(8),
    Int64Values([2L, 4L, 8L]),
  )
  writer.add_dataset_typed(
    "/g/complex",
    [2UL],
    Datatype::write_complex(Datatype::write_float(8)),
    Complex128Values([(1.5, -2.5), (3.0, 4.0)]),
  )
  writer.add_dataset_typed(
    "/g/complex_compound",
    [1UL],
    Datatype::write_compound([
      ("r", Datatype::write_float(8)),
      ("i", Datatype::write_float(8)),
    ]),
    Complex128Values([(5.0, -6.0)]),
  )
  let external_slot : Hdf5ExternalSlot = {
    file_name: "typed_external.bin",
    file_offset: 0UL,
    data_size: 8UL,
  }
  writer.add_external_dataset_typed(
    "/g/external",
    [2UL],
    Datatype::write_integer(false, 4),
    UInt16Values([7U, 8U]),
    [external_slot],
  )
  writer.add_dataset("/g/source", [2UL], Int16Values([20, 21]))
  writer.add_virtual_dataset_typed(
    "/g/virtual",
    [2UL],
    Datatype::write_integer(true, 4),
    [
      {
        source_file: ".",
        source_dataset: "/g/source",
        source_selection: RegionAll,
        virtual_selection: RegionAll,
      },
    ],
    fill_value=Int64Values([-1L]),
  )

  let bytes = writer.encode()
  let file = Hdf5File::parse(bytes)
  assert_eq(
    conversion_test_unsigned(file.read_dataset("/g/contiguous").values),
    [1UL, 2UL, 3UL],
  )
  assert_eq(conversion_test_floats(file.read_dataset("/g/compact").values), [
    4.0, 5.0,
  ])
  assert_eq(conversion_test_unsigned(file.read_dataset("/g/chunked").values), [
    10UL, 11UL, 12UL, 13UL,
  ])
  assert_eq(conversion_test_floats(file.read_dataset("/g/extensible").values), [
    1.0, 2.0,
  ])
  assert_eq(
    conversion_test_floats(
      file.read_attribute("/g/contiguous", "weights").values.unwrap(),
    ),
    [2.0, 4.0, 8.0],
  )
  assert_eq(
    conversion_test_structured(file.read_dataset("/g/complex").values),
    [Complex(1.5, -2.5), Complex(3.0, 4.0)],
  )
  assert_eq(
    conversion_test_structured(file.read_dataset("/g/complex_compound").values),
    [
      Compound([
        { name: "r", value: FloatingPoint(5.0), },
        { name: "i", value: FloatingPoint(-6.0), },
      ]),
    ],
  )
  assert_eq(
    conversion_test_integers(
      file.read_dataset_as("/g/contiguous", Datatype::write_integer(true, 8)).values,
    ),
    [1L, 2L, 3L],
  )
  assert_eq(
    conversion_test_floats(
      file.read_attribute_as(
        "/g/contiguous",
        "weights",
        Datatype::write_float(4),
      ).values.unwrap(),
    ),
    [2.0, 4.0, 8.0],
  )
  assert_true(file.exists("/g/contiguous"))
  assert_eq(file.object_kind_at("/g/contiguous"), Dataset)

  let chunks = file.chunk_iterator_as(
    "/g/chunked",
    [2UL],
    Datatype::write_integer(true, 8),
  )
  let first_chunk = chunks.next().unwrap()
  assert_eq(first_chunk.offset, [0UL])
  assert_eq(conversion_test_integers(first_chunk.values), [10L, 11L])
  let second_chunk = chunks.next().unwrap()
  assert_eq(conversion_test_integers(second_chunk.values), [12L, 13L])
  assert_true(chunks.next() is None)

  let pages = file.selection_iterator_as(
    "/g/contiguous",
    Points([[2UL], [0UL], [1UL]]),
    2,
    Datatype::write_integer(true, 8),
  )
  let first_page = pages.next().unwrap()
  assert_eq(first_page.offset, 0UL)
  assert_eq(first_page.datatype.class, 0)
  assert_eq(conversion_test_integers(first_page.values), [3L, 1L])
  let second_page = pages.next().unwrap()
  assert_eq(conversion_test_integers(second_page.values), [2L])
  assert_true(pages.next() is None)

  let external = @buffer.Buffer::Buffer()
  external.write_uint_le(7U)
  external.write_uint_le(8U)
  let resolver = Hdf5Resolver::new(bytes)
  resolver.add_source("typed_external.bin", external.to_bytes())
  assert_eq(
    conversion_test_unsigned(
      resolver.read_dataset_as("/g/external", Datatype::write_integer(false, 8)).values,
    ),
    [7UL, 8UL],
  )
  let resolver_chunks = resolver.chunk_iterator_as(
    "/g/external",
    [2UL],
    Datatype::write_integer(true, 8),
  )
  assert_eq(conversion_test_integers(resolver_chunks.next().unwrap().values), [
    7L, 8L,
  ])
  let resolver_pages = resolver.selection_iterator_as(
    "/g/external",
    Hyperslab([0UL], [2UL], [1UL]),
    1,
    Datatype::write_integer(true, 8),
  )
  assert_eq(conversion_test_integers(resolver_pages.next().unwrap().values), [
    7L,
  ])
  assert_eq(conversion_test_integers(resolver_pages.next().unwrap().values), [
    8L,
  ])
  assert_eq(conversion_test_integers(file.read_dataset("/g/virtual").values), [
    20L, 21L,
  ])
}

///|
test "typed existing-file APIs add and patch converted values" {
  let writer = Hdf5Writer::new()
  writer.add_dataset("/data", [4UL], Int64Values([1L, 2L, 3L, 4L]))
  writer.add_attribute("/data", "weights", [4UL], Int64Values([1L, 2L, 3L, 4L]))
  writer.add_extensible_dataset(
    "/append",
    [2UL],
    [None],
    [2U],
    Int64Values([1L, 2L]),
  )
  let original = Hdf5File::parse(writer.encode())

  let patched = original.patch_dataset_values_typed(
    "/data",
    Datatype::write_integer(true, 8),
    Int32Values([5, 6, 7, 8]),
  )
  let patched_file = Hdf5File::parse(patched)
  assert_eq(
    conversion_test_integers(patched_file.read_dataset("/data").values),
    [5L, 6L, 7L, 8L],
  )
  let patched_attribute = patched_file.patch_attribute_values_typed(
    "/data",
    "weights",
    Int32Values([8, 7, 6, 5]),
  )
  let attribute_file = Hdf5File::parse(patched_attribute)
  assert_eq(
    conversion_test_integers(
      attribute_file.read_attribute("/data", "weights").values.unwrap(),
    ),
    [8L, 7L, 6L, 5L],
  )
  let appended = attribute_file.append_dataset_typed(
    "/append",
    [2UL],
    Int32Values([3, 4]),
  )
  let appended_file = Hdf5File::parse(appended)
  assert_eq(
    conversion_test_integers(appended_file.read_dataset("/append").values),
    [1L, 2L, 3L, 4L],
  )

  let created = appended_file.add_dataset_typed(
    "/",
    "created",
    [3UL],
    Datatype::write_integer(false, 2),
    Int64Values([9L, 10L, 11L]),
  )
  let created_file = Hdf5File::parse(created)
  assert_eq(
    conversion_test_unsigned(created_file.read_dataset("/created").values),
    [9UL, 10UL, 11UL],
  )
  let with_attribute = created_file.add_attribute_typed(
    "/created",
    "labels",
    [2UL],
    Datatype::write_fixed_string(4),
    VariableStringValues(["ab", "c"]),
  )
  let attribute_file = Hdf5File::parse(with_attribute)
  assert_eq(
    conversion_test_strings(
      attribute_file.read_attribute("/created", "labels").values.unwrap(),
    ),
    ["ab", "c"],
  )
}

///|
fn conversion_test_raw_datatype(
  class_ : Int32,
  size_ : Int32,
  version_~ : Int32 = 1,
  base_types_~ : Array[Datatype] = [],
  compound_members_~ : Array[CompoundMember] = [],
  enum_members_~ : Array[EnumMember] = [],
  array_dimensions_~ : Array[UInt64] = [],
  bit_precision_~ : Int32 = 0,
  reference_kind_~ : Int32 = 0,
) -> Datatype {
  {
    class: class_,
    version: version_,
    size: size_,
    signed: false,
    little_endian: true,
    vlen_is_string: false,
    enum_members: enum_members_,
    compound_members: compound_members_,
    base_types: base_types_,
    array_dimensions: array_dimensions_,
    bit_offset: 0,
    bit_precision: bit_precision_,
    float_sign_position: 0,
    float_exponent_position: 0,
    float_exponent_size: 0,
    float_mantissa_position: 0,
    float_mantissa_size: 0,
    float_exponent_bias: 0,
    float_normalization: 0,
    string_padding: 0,
    string_charset: 0,
    opaque_tag: "",
    reference_kind: reference_kind_,
  }
}

///|
test "malformed and over-deep conversion targets fail before decoding values" {
  let source = Datatype::write_integer(true, 1)
  let int8_type = Datatype::write_integer(true, 1)
  let cases : Array[(Datatype, Hdf5Error)] = [
    (
      conversion_test_raw_datatype(1, 2, bit_precision_=0),
      InvalidArgument("conversion target float layout is invalid"),
    ),
    (
      conversion_test_raw_datatype(6, 4),
      InvalidArgument("conversion target compound has no members"),
    ),
    (
      conversion_test_raw_datatype(8, 1, base_types_=[int8_type], enum_members_=[
        { name: "OVERFLOW", value: EnumValue::Signed(128L), },
      ]),
      InvalidArgument(
        "conversion target enum member is outside its integer base",
      ),
    ),
    (
      conversion_test_raw_datatype(10, 3, base_types_=[int8_type], array_dimensions_=[
        2UL,
      ]),
      InvalidArgument("conversion target array layout is invalid"),
    ),
    (
      conversion_test_raw_datatype(7, 4, version_=4, reference_kind_=0),
      InvalidArgument("conversion target reference layout is invalid"),
    ),
  ]
  for (target, expected) in cases {
    let error = conversion_test_error(() => {
      source.convert_values(Integers([]), target) |> ignore
    })
    assert_eq(error, Some(expected))
  }

  let mut deep_target = Datatype::write_float(8)
  for _ in 0..<17 {
    deep_target = Datatype::write_array(deep_target, [1UL])
  }
  assert_eq(
    conversion_test_error(() => {
      source.convert_values(Integers([]), deep_target) |> ignore
    }),
    Some(InvalidArgument("conversion target datatype is invalid or too deep")),
  )
}
