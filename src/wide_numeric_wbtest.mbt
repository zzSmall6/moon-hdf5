///|
fn wide_big(value : Int32) -> @bigint.BigInt {
  @bigint.BigInt::from_int(value)
}

///|
fn wide_pow2(bits : Int32) -> @bigint.BigInt {
  @bigint.BigInt::from_int(1).shl(bits)
}

///|
fn wide_float_values(values : DatasetValues) -> Array[Double] {
  match values {
    Floats(items) => items
    _ => abort("expected floating-point values")
  }
}

///|
test "wide fixed-point values are decoded without truncation" {
  let file = Hdf5File::parse(@fs.read_file_to_bytes("testdata/wide_numeric.h5"))
  let signed_expected = [
    wide_big(123),
    wide_big(-456),
    wide_pow2(100),
    wide_pow2(100).neg(),
  ]
  for path in ["/int128_le", "/int128_be"] {
    match file.read_dataset(path).values {
      BigIntegers(items) => assert_eq(items, signed_expected)
      _ => abort("expected wide signed integers for " + path)
    }
  }
  let unsigned_expected = [
    wide_big(0),
    wide_pow2(64),
    wide_pow2(127).add(wide_big(5)),
    wide_pow2(128).sub(wide_big(1)),
  ]
  match file.read_dataset("/uint128_le").values {
    BigIntegers(items) => assert_eq(items, unsigned_expected)
    _ => abort("expected wide unsigned integers")
  }
  match file.read_dataset("/int96_offset_le").values {
    BigIntegers(items) =>
      assert_eq(items, [wide_big(123), wide_big(-456), wide_pow2(70)])
    _ => abort("expected wide offset integers")
  }
}

///|
test "extended floating-point layouts decode to double" {
  let file = Hdf5File::parse(@fs.read_file_to_bytes("testdata/wide_numeric.h5"))
  for
    path in [
      "/float128", "/float80_le", "/float80_be", "/float_implied_wide", "/float_msbset",
    ] {
    let items = wide_float_values(file.read_dataset(path).values)
    assert_eq(items.length(), 10)
    assert_eq(items[0], 1.5)
    assert_eq(items[1], -2.25)
    assert_eq(items[2], 0.125)
    assert_eq(items[3], 0.0)
    assert_eq(items[4], 0.0)
    assert_true(1.0 / items[4] < 0.0)
    assert_eq(items[5], 0x7FF0000000000000UL.reinterpret_as_double())
    assert_eq(items[6], 0xFFF0000000000000UL.reinterpret_as_double())
    assert_true(items[7].is_nan())
    assert_eq(items[8], 0.0)
    assert_eq(items[9], 0x7FF0000000000000UL.reinterpret_as_double())
  }
}

///|
test "writer emits and patches wide integers" {
  let writer = Hdf5Writer::new()
  writer.add_dataset(
    "/signed16",
    [3UL],
    BigIntegerValues([wide_big(1), wide_big(-2), wide_pow2(80)], 16, true),
  )
  writer.add_dataset(
    "/unsigned16",
    [2UL],
    BigIntegerValues(
      [wide_pow2(64), wide_pow2(120).add(wide_big(7))],
      16,
      false,
    ),
  )
  let big_endian = Datatype::write_integer_with_order(true, 16, false)
  writer.add_dataset(
    "/signed16_be",
    [2UL],
    StructuredValues(big_endian, [
      BigInteger(wide_big(-9)),
      BigInteger(wide_pow2(90)),
    ]),
  )
  let encoded = writer.encode()
  let file = Hdf5File::parse(encoded)
  match file.read_dataset("/signed16").values {
    BigIntegers(items) =>
      assert_eq(items, [wide_big(1), wide_big(-2), wide_pow2(80)])
    _ => abort("expected written wide signed integers")
  }
  match file.read_dataset("/unsigned16").values {
    BigIntegers(items) =>
      assert_eq(items, [wide_pow2(64), wide_pow2(120).add(wide_big(7))])
    _ => abort("expected written wide unsigned integers")
  }
  match file.read_dataset("/signed16_be").values {
    BigIntegers(items) => assert_eq(items, [wide_big(-9), wide_pow2(90)])
    _ => abort("expected written big-endian wide integers")
  }
  let patched = file.patch_dataset_values(
    "/signed16",
    BigIntegerValues([wide_big(5), wide_big(-6), wide_pow2(100)], 16, true),
  )
  match Hdf5File::parse(patched).read_dataset("/signed16").values {
    BigIntegers(items) =>
      assert_eq(items, [wide_big(5), wide_big(-6), wide_pow2(100)])
    _ => abort("expected patched wide integers")
  }
}

///|
test "writer emits wide time bitfield and enum values" {
  let writer = Hdf5Writer::new()
  let time_type = Datatype::write_time(16)
  writer.add_dataset(
    "/time16",
    [2UL],
    StructuredValues(time_type, [BigTime(wide_pow2(70)), BigTime(wide_big(42))]),
  )
  let bitfield_type = Datatype::write_bitfield(16, 100, offset=4)
  writer.add_dataset(
    "/bitfield16",
    [2UL],
    StructuredValues(bitfield_type, [
      BigBitfield(wide_pow2(90)),
      BigBitfield(wide_big(3)),
    ]),
  )
  let enum_base = Datatype::write_integer(true, 16)
  let enum_type = Datatype::write_enum(enum_base, [
    { name: "small", value: EnumValue::Big(wide_big(-1)), },
    { name: "large", value: EnumValue::Big(wide_pow2(100)), },
  ])
  writer.add_dataset(
    "/enum16",
    [2UL],
    StructuredValues(enum_type, [Enumeration("small"), Enumeration("large")]),
  )
  let file = Hdf5File::parse(writer.encode())
  match file.read_dataset("/time16").values {
    Structured(values) =>
      assert_eq(values, [BigTime(wide_pow2(70)), BigTime(wide_big(42))])
    _ => abort("expected wide time values")
  }
  match file.read_dataset("/bitfield16").values {
    Structured(values) =>
      assert_eq(values, [BigBitfield(wide_pow2(90)), BigBitfield(wide_big(3))])
    _ => abort("expected wide bitfield values")
  }
  match file.read_dataset("/enum16").values {
    Enums(values) => assert_eq(values, ["small", "large"])
    _ => abort("expected wide enum values")
  }
}
