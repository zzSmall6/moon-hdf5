///|
fn matrix_big(value : Int32) -> @bigint.BigInt {
  @bigint.BigInt::from_int(value)
}

///|
fn matrix_pow2(exponent : Int32) -> @bigint.BigInt {
  @bigint.BigInt::from_int(1).shl(exponent)
}

///|
fn matrix_integer_minimum(datatype : Datatype) -> @bigint.BigInt {
  if datatype.signed {
    matrix_pow2(datatype.bit_precision - 1).neg()
  } else {
    matrix_big(0)
  }
}

///|
fn matrix_integer_maximum(datatype : Datatype) -> @bigint.BigInt {
  if datatype.signed {
    matrix_pow2(datatype.bit_precision - 1).sub(matrix_big(1))
  } else {
    matrix_pow2(datatype.bit_precision).sub(matrix_big(1))
  }
}

///|
fn matrix_integer_in_range(value : @bigint.BigInt, datatype : Datatype) -> Bool {
  value.compare(matrix_integer_minimum(datatype)) >= 0 &&
  value.compare(matrix_integer_maximum(datatype)) <= 0
}

///|
fn matrix_integer_samples(datatype : Datatype) -> Array[@bigint.BigInt] {
  let values = [matrix_big(0), matrix_big(1)]
  if datatype.signed {
    values.push(matrix_big(-1))
  }
  for
    candidate in [
      matrix_pow2(11),
      matrix_pow2(11).add(matrix_big(1)),
      matrix_pow2(23),
      matrix_pow2(23).add(matrix_big(1)),
      matrix_pow2(53),
      matrix_pow2(53).add(matrix_big(1)),
    ] {
    if matrix_integer_in_range(candidate, datatype) {
      values.push(candidate)
    }
    if datatype.signed && matrix_integer_in_range(candidate.neg(), datatype) {
      values.push(candidate.neg())
    }
  }
  values.push(matrix_integer_minimum(datatype))
  values.push(matrix_integer_maximum(datatype))
  values
}

///|
fn matrix_integer_value(
  datatype : Datatype,
  value : @bigint.BigInt,
) -> DatasetValue {
  if datatype.size > 8 {
    BigInteger(value)
  } else if datatype.signed {
    SignedInteger(value.to_int64())
  } else {
    UnsignedInteger(value.to_uint64())
  }
}

///|
fn matrix_integer_result(value : DatasetValue) -> @bigint.BigInt {
  match value {
    SignedInteger(number) => @bigint.BigInt::from_int64(number)
    UnsignedInteger(number) => @bigint.BigInt::from_uint64(number)
    BigInteger(number) => number
    _ => abort("matrix expected an integer conversion result")
  }
}

///|
fn matrix_float_result(value : DatasetValue) -> Double {
  match value {
    FloatingPoint(number) => number
    _ => abort("matrix expected a floating-point conversion result")
  }
}

///|
fn matrix_integer_fits(
  value : @bigint.BigInt,
  precision : Int32,
  signed : Bool,
) -> Bool {
  if signed {
    let half = matrix_pow2(precision - 1)
    value.compare(half.neg()) >= 0 && value.compare(half) < 0
  } else {
    value.compare(matrix_big(0)) >= 0 &&
    value.compare(matrix_pow2(precision)) < 0
  }
}

///|
fn matrix_wrap_integer(
  value : @bigint.BigInt,
  precision : Int32,
  signed : Bool,
) -> @bigint.BigInt {
  let modulus = matrix_pow2(precision)
  let mut remainder = value.mod(modulus)
  if remainder.compare(matrix_big(0)) < 0 {
    remainder = remainder.add(modulus)
  }
  if signed && remainder.compare(matrix_pow2(precision - 1)) >= 0 {
    remainder.sub(modulus)
  } else {
    remainder
  }
}

///|
fn matrix_try_convert(
  source : Datatype,
  value : DatasetValue,
  target : Datatype,
  options : Hdf5ConversionOptions,
) -> DatasetValue? {
  Some(source.convert_value(value, target, options~)) catch {
    _ => None
  }
}

///|
fn[T : Eq] matrix_assert_equal(
  actual : T,
  expected : T,
  context : String,
) -> Unit {
  if actual != expected {
    abort(context + ": converted values differ")
  }
}

///|
fn matrix_assert_true(condition : Bool, context : String) -> Unit {
  if !condition {
    abort(context + ": assertion failed")
  }
}

///|
fn matrix_round_float(value : Double, target : Datatype) -> Double {
  if target.size == 2 {
    float16_to_float(writer_float16_bits_from_double(value)).to_double()
  } else if target.size == 4 {
    Float::from_double(value).to_double()
  } else {
    value
  }
}

///|
test "half conversion avoids double rounding at binary16 boundaries" {
  let float16_type = Datatype::write_float(2)
  let float64_type = Datatype::write_float(8)
  let lossy = Hdf5ConversionOptions::lossy()
  let midpoint = 1.0 + 1.0 / 2048.0
  let above_midpoint = midpoint + 1.0 / 1099511627776.0
  let below_midpoint = midpoint - 1.0 / 1099511627776.0
  let upper = 1.0 + 1.0 / 1024.0
  assert_eq(
    float64_type.convert_value(
      FloatingPoint(above_midpoint),
      float16_type,
      options=lossy,
    ),
    FloatingPoint(upper),
  )
  assert_eq(
    float64_type.convert_value(
      FloatingPoint(-above_midpoint),
      float16_type,
      options=lossy,
    ),
    FloatingPoint(-upper),
  )
  assert_eq(
    float64_type.convert_value(
      FloatingPoint(midpoint),
      float16_type,
      options=lossy,
    ),
    FloatingPoint(1.0),
  )
  assert_eq(
    float64_type.convert_value(
      FloatingPoint(below_midpoint),
      float16_type,
      options=lossy,
    ),
    FloatingPoint(1.0),
  )
  assert_true(
    float16_to_float(writer_float16_bits_from_double(65520.0)).is_inf(),
  )
  assert_eq(
    float16_to_float(writer_float16_bits_from_double(65519.0)).to_double(),
    65504.0,
  )
  assert_eq(writer_float16_bits_from_double(1.0 / 67108864.0), 0U)
  assert_eq(writer_float16_bits_from_double(-1.0 / 67108864.0), 0x8000U)
  assert_eq(writer_float16_bits_from_double(1.0 / 33554432.0), 0U)
  assert_eq(
    writer_float16_bits_from_double(
      1.0 / 33554432.0 + 1.0 / 1152921504606846976.0,
    ),
    1U,
  )
  assert_eq(writer_float16_bits_from_double(1.0 / 16777216.0), 1U)
  assert_eq(writer_float16_bits_from_double(3.0 / 33554432.0), 2U)
}

///|
fn matrix_integer_types() -> Array[(String, Datatype)] raise Hdf5Error {
  [
    ("i8", Datatype::write_integer(true, 1)),
    ("u8", Datatype::write_integer(false, 1)),
    ("i16", Datatype::write_integer(true, 2)),
    ("u16", Datatype::write_integer(false, 2)),
    ("i16be", Datatype::write_integer_with_order(true, 2, false)),
    ("i32", Datatype::write_integer(true, 4)),
    ("u32", Datatype::write_integer(false, 4)),
    ("u32be", Datatype::write_integer_with_order(false, 4, false)),
    ("i64", Datatype::write_integer(true, 8)),
    ("u64", Datatype::write_integer(false, 8)),
    ("i128", Datatype::write_integer(true, 16)),
    ("u128", Datatype::write_integer(false, 16)),
  ]
}

///|
fn matrix_float_types() -> Array[(String, Datatype)] raise Hdf5Error {
  [
    ("f16", Datatype::write_float(2)),
    ("f32", Datatype::write_float(4)),
    ("f64", Datatype::write_float(8)),
  ]
}

///|
test "integer conversion matrix covers every width sign and byte order" {
  let strict = Hdf5ConversionOptions::strict()
  let lossy = Hdf5ConversionOptions::lossy()
  let types = matrix_integer_types()
  for (source_name, source) in types {
    for (target_name, target) in types {
      for value in matrix_integer_samples(source) {
        let source_value = matrix_integer_value(source, value)
        let strict_result = matrix_try_convert(
          source, source_value, target, strict,
        )
        let fits = matrix_integer_fits(
          value,
          target.bit_precision,
          target.signed,
        )
        if fits {
          match strict_result {
            Some(converted) =>
              matrix_assert_equal(
                matrix_integer_result(converted),
                value,
                source_name + "->" + target_name + " strict",
              )
            None =>
              abort(
                "strict integer conversion unexpectedly failed: " +
                source_name +
                "->" +
                target_name +
                " value " +
                value.to_string(),
              )
          }
        } else {
          matrix_assert_true(
            strict_result is None,
            source_name + "->" + target_name + " strict overflow",
          )
        }
        let lossy_result = matrix_try_convert(
          source, source_value, target, lossy,
        )
        match lossy_result {
          Some(converted) =>
            matrix_assert_equal(
              matrix_integer_result(converted),
              matrix_wrap_integer(value, target.bit_precision, target.signed),
              source_name + "->" + target_name + " lossy",
            )
          None =>
            abort(
              "lossy integer conversion unexpectedly failed: " +
              source_name +
              "->" +
              target_name +
              " value " +
              value.to_string(),
            )
        }
      }
    }
    assert_true(
      source.can_convert_to(source, options=Hdf5ConversionOptions::none()),
    )
    assert_true(source.can_convert_to(source, options=lossy))
  }
}

///|
test "integer to floating conversion matrix honors precision boundaries" {
  let strict = Hdf5ConversionOptions::strict()
  let lossy = Hdf5ConversionOptions::lossy()
  let integer_types = matrix_integer_types()
  let float_types = matrix_float_types()
  let raw_samples = [
    matrix_big(0),
    matrix_big(1),
    matrix_big(-1),
    matrix_pow2(11),
    matrix_pow2(11).add(matrix_big(1)),
    matrix_pow2(23),
    matrix_pow2(23).add(matrix_big(1)),
    matrix_pow2(53),
    matrix_pow2(53).add(matrix_big(1)),
  ]
  for (source_name, source) in integer_types {
    for (target_name, target) in float_types {
      for raw in raw_samples {
        if !matrix_integer_in_range(raw, source) {
          continue
        }
        let value = matrix_integer_value(source, raw)
        let strict_result = matrix_try_convert(source, value, target, strict)
        let exact = conversion_integer_exact_in_float(raw, target)
        if exact {
          match strict_result {
            Some(converted) =>
              matrix_assert_equal(
                matrix_float_result(converted),
                conversion_bigint_to_double(raw),
                source_name + "->" + target_name + " strict",
              )
            None =>
              abort(
                "strict integer-to-float conversion unexpectedly failed: " +
                source_name +
                "->" +
                target_name,
              )
          }
        } else {
          matrix_assert_true(
            strict_result is None,
            source_name + "->" + target_name + " precision",
          )
        }
        let lossy_result = matrix_try_convert(source, value, target, lossy)
        match lossy_result {
          Some(converted) =>
            matrix_assert_equal(
              matrix_float_result(converted),
              matrix_round_float(conversion_bigint_to_double(raw), target),
              source_name + "->" + target_name + " lossy",
            )
          None =>
            abort(
              "lossy integer-to-float conversion unexpectedly failed: " +
              source_name +
              "->" +
              target_name,
            )
        }
      }
    }
  }
}

///|
test "floating conversion matrix honors strict rounding and lossy truncation" {
  let strict = Hdf5ConversionOptions::strict()
  let lossy = Hdf5ConversionOptions::lossy()
  let types = matrix_float_types()
  let raw_values = [0.0, 1.5, -2.25, 1.0 / 3.0, 65504.0]
  for (source_name, source) in types {
    for (target_name, target) in types {
      for raw in raw_values {
        let value = FloatingPoint(matrix_round_float(raw, source))
        let strict_result = matrix_try_convert(source, value, target, strict)
        let source_value = matrix_round_float(raw, source)
        let expected = matrix_round_float(source_value, target)
        if expected == source_value {
          match strict_result {
            Some(converted) =>
              matrix_assert_equal(
                matrix_float_result(converted),
                expected,
                source_name + "->" + target_name + " strict",
              )
            None =>
              abort(
                "strict floating conversion unexpectedly failed: " +
                source_name +
                "->" +
                target_name,
              )
          }
        } else {
          matrix_assert_true(
            strict_result is None,
            source_name + "->" + target_name + " precision",
          )
        }
        let lossy_result = matrix_try_convert(source, value, target, lossy)
        match lossy_result {
          Some(converted) =>
            matrix_assert_equal(
              matrix_float_result(converted),
              expected,
              source_name + "->" + target_name + " lossy",
            )
          None =>
            abort(
              "lossy floating conversion unexpectedly failed: " +
              source_name +
              "->" +
              target_name,
            )
        }
      }
    }
  }
}

///|
test "extended floating capability remains conservative under strict mode" {
  let strict = Hdf5ConversionOptions::strict()
  let lossy = Hdf5ConversionOptions::lossy()
  let float64_type = Datatype::write_float(8)
  let float80_type = Datatype::write_float(10)
  assert_true(float64_type.can_convert_to(float80_type, options=strict))
  assert_true(!float80_type.can_convert_to(float64_type, options=strict))
  assert_true(float80_type.can_convert_to(float64_type, options=lossy))
}

///|
test "conversion capability has no strict or lossy false positives" {
  let numeric_types = matrix_integer_types() + matrix_float_types()
  let values : Array[DatasetValue] = [
    SignedInteger(0L),
    SignedInteger(1L),
    SignedInteger(-1L),
    SignedInteger(2048L),
    SignedInteger(2049L),
    SignedInteger(16777217L),
    SignedInteger(9007199254740993L),
    UnsignedInteger(0UL),
    UnsignedInteger(1UL),
    UnsignedInteger(255UL),
    UnsignedInteger(65535UL),
    FloatingPoint(0.0),
    FloatingPoint(1.5),
    FloatingPoint(-2.25),
    FloatingPoint(1.0 / 3.0),
  ]
  for mode in [Hdf5ConversionOptions::strict(), Hdf5ConversionOptions::lossy()] {
    for (source_name, source) in numeric_types {
      for (target_name, target) in numeric_types {
        if !source.can_convert_to(target, options=mode) {
          continue
        }
        for value in values {
          if !matrix_value_matches_source(source, value) {
            continue
          }
          matrix_assert_true(
            matrix_try_convert(source, value, target, mode) is Some(_),
            source_name + "->" + target_name + " capability",
          )
        }
      }
    }
  }
}

///|
fn matrix_value_matches_source(source : Datatype, value : DatasetValue) -> Bool {
  match (source.class, value) {
    (0, SignedInteger(number)) =>
      source.signed &&
      matrix_integer_fits(
        @bigint.BigInt::from_int64(number),
        source.bit_precision,
        true,
      )
    (0, UnsignedInteger(number)) =>
      !source.signed &&
      matrix_integer_fits(
        @bigint.BigInt::from_uint64(number),
        source.bit_precision,
        false,
      )
    (0, BigInteger(number)) =>
      source.size > 8 &&
      matrix_integer_fits(number, source.bit_precision, source.signed)
    (1, FloatingPoint(number)) => matrix_round_float(number, source) == number
    _ => false
  }
}

///|
test "strict capability is conservative but per-value conversion is precise" {
  let strict = Hdf5ConversionOptions::strict()
  let int64_type = Datatype::write_integer(true, 8)
  let int8_type = Datatype::write_integer(true, 1)
  let int32_type = Datatype::write_integer(true, 4)
  let uint32_type = Datatype::write_integer(false, 4)
  let uint8_type = Datatype::write_integer(false, 1)
  let float32_type = Datatype::write_float(4)
  let float64_type = Datatype::write_float(8)
  assert_true(!int64_type.can_convert_to(int8_type, options=strict))
  assert_eq(
    int64_type.convert_value(SignedInteger(1L), int8_type, options=strict),
    SignedInteger(1L),
  )
  assert_true(!int32_type.can_convert_to(uint32_type, options=strict))
  assert_eq(
    int32_type.convert_value(SignedInteger(1L), uint32_type, options=strict),
    UnsignedInteger(1UL),
  )
  assert_true(!uint8_type.can_convert_to(int8_type, options=strict))
  assert_eq(
    uint8_type.convert_value(UnsignedInteger(127UL), int8_type, options=strict),
    SignedInteger(127L),
  )
  assert_true(!float64_type.can_convert_to(float32_type, options=strict))
  assert_eq(
    float64_type.convert_value(FloatingPoint(1.5), float32_type, options=strict),
    FloatingPoint(1.5),
  )
}

///|
test "complex capability honors component precision across bridge forms" {
  let strict = Hdf5ConversionOptions::strict()
  let lossy = Hdf5ConversionOptions::lossy()
  let float16_type = Datatype::write_float(2)
  let float32_type = Datatype::write_float(4)
  let float64_type = Datatype::write_float(8)
  let complex16_type = Datatype::write_complex(float16_type)
  let complex32_type = Datatype::write_complex(float32_type)
  let complex64_type = Datatype::write_complex(float64_type)
  let compound32_type = Datatype::write_compound([
    ("r", float32_type),
    ("i", float32_type),
  ])
  let compound64_type = Datatype::write_compound([
    ("r", float64_type),
    ("i", float64_type),
  ])
  let mixed_compound_type = Datatype::write_compound([
    ("r", float64_type),
    ("i", float32_type),
  ])
  let array32_type = Datatype::write_array(float32_type, [2UL])
  let array64_type = Datatype::write_array(float64_type, [2UL])
  assert_true(complex16_type.can_convert_to(complex32_type, options=strict))
  assert_true(complex32_type.can_convert_to(complex64_type, options=strict))
  assert_true(!complex64_type.can_convert_to(complex32_type, options=strict))
  assert_true(complex64_type.can_convert_to(complex32_type, options=lossy))
  assert_true(!complex64_type.can_convert_to(compound32_type, options=strict))
  assert_true(!complex64_type.can_convert_to(array32_type, options=strict))
  assert_true(complex64_type.can_convert_to(compound64_type, options=strict))
  assert_true(complex64_type.can_convert_to(array64_type, options=strict))
  assert_true(complex32_type.can_convert_to(compound64_type, options=strict))
  assert_true(complex32_type.can_convert_to(array64_type, options=strict))
  assert_true(compound64_type.can_convert_to(complex64_type, options=strict))
  assert_true(array64_type.can_convert_to(complex64_type, options=strict))
  assert_true(!compound64_type.can_convert_to(complex32_type, options=strict))
  assert_true(!array64_type.can_convert_to(complex32_type, options=strict))
  assert_true(
    !mixed_compound_type.can_convert_to(complex32_type, options=strict),
  )
  assert_true(
    mixed_compound_type.can_convert_to(complex64_type, options=strict),
  )
  assert_eq(
    complex64_type.convert_value(
      Complex(1.5, -2.25),
      complex32_type,
      options=strict,
    ),
    Complex(1.5, -2.25),
  )
}

///|
test "time and bitfield conversions cover narrow and wide integer paths" {
  let strict = Hdf5ConversionOptions::strict()
  let time16_type = Datatype::write_time(2)
  let time64_type = Datatype::write_time(8)
  let time128_type = Datatype::write_time(16)
  let bitfield32_type = Datatype::write_bitfield(4, 32)
  let bitfield128_type = Datatype::write_bitfield(16, 80)
  let uint64_type = Datatype::write_integer(false, 8)
  let int64_type = Datatype::write_integer(true, 8)
  let wide = matrix_pow2(70)
  assert_eq(
    time16_type.convert_value(Time(258UL), time128_type, options=strict),
    BigTime(matrix_big(258)),
  )
  assert_eq(
    bitfield32_type.convert_value(
      Bitfield(0x1234UL),
      time128_type,
      options=strict,
    ),
    BigTime(matrix_big(0x1234)),
  )
  assert_eq(
    time64_type.convert_value(Time(1UL << 40), bitfield128_type, options=strict),
    BigBitfield(matrix_pow2(40)),
  )
  assert_eq(
    bitfield128_type.convert_value(
      BigBitfield(wide),
      time128_type,
      options=strict,
    ),
    BigTime(wide),
  )
  assert_eq(
    time16_type.convert_value(Time(5UL), uint64_type, options=strict),
    UnsignedInteger(5UL),
  )
  assert_eq(
    bitfield32_type.convert_value(Bitfield(5UL), int64_type, options=strict),
    SignedInteger(5L),
  )
  assert_true(time16_type.can_convert_to(bitfield32_type, options=strict))
  assert_true(bitfield32_type.can_convert_to(time64_type, options=strict))
  assert_true(bitfield128_type.can_convert_to(time128_type, options=strict))
}

///|
test "nested conversion matrix reorders members and fills typed defaults" {
  let state = Datatype::write_enum(Datatype::write_integer(false, 1), [
    { name: "UNKNOWN", value: EnumValue::Unsigned(0UL), },
    { name: "OPEN", value: EnumValue::Unsigned(1UL), },
    { name: "CLOSED", value: EnumValue::Unsigned(2UL), },
  ])
  let source_nested = Datatype::write_compound([
    ("x", Datatype::write_float(4)),
    ("flags", Datatype::write_bitfield(2, 12)),
  ])
  let target_nested = Datatype::write_compound([
    ("flags", Datatype::write_bitfield(2, 12)),
    ("x", Datatype::write_float(8)),
    ("stamp", Datatype::write_time(8)),
  ])
  let source = Datatype::write_compound([
    ("id", Datatype::write_integer(true, 4)),
    ("state", state),
    ("nested", source_nested),
    (
      "vector",
      Datatype::write_array(Datatype::write_integer(true, 2), [2UL, 2UL]),
    ),
    ("values", Datatype::write_vlen_sequence(Datatype::write_integer(true, 2))),
    ("label", Datatype::write_fixed_string(4)),
  ])
  let target = Datatype::write_compound([
    ("values", Datatype::write_vlen_sequence(Datatype::write_float(8))),
    ("id", Datatype::write_integer(true, 8)),
    ("nested", target_nested),
    ("vector", Datatype::write_array(Datatype::write_float(8), [2UL, 2UL])),
    ("state", state),
    ("note", Datatype::write_fixed_string(6)),
  ])
  let source_value = Compound([
    { name: "id", value: SignedInteger(7L), },
    { name: "state", value: Enumeration("OPEN"), },
    {
      name: "nested",
      value: Compound([
        { name: "x", value: FloatingPoint(1.5), },
        { name: "flags", value: Bitfield(0x0ABUL), },
      ]),
    },
    {
      name: "vector",
      value: Elements([
        SignedInteger(1L),
        SignedInteger(2L),
        SignedInteger(3L),
        SignedInteger(4L),
      ]),
    },
    {
      name: "values",
      value: Sequence([SignedInteger(-1L), SignedInteger(2L)]),
    },
    { name: "label", value: Text("src"), },
  ])
  let options = {
    mode: Hdf5ConversionMode::Strict,
    allow_missing_compound_members: true,
  }
  assert_true(source.can_convert_to(target, options~))
  assert_eq(
    source.convert_value(source_value, target, options~),
    Compound([
      {
        name: "values",
        value: Sequence([FloatingPoint(-1.0), FloatingPoint(2.0)]),
      },
      { name: "id", value: SignedInteger(7L), },
      {
        name: "nested",
        value: Compound([
          { name: "flags", value: Bitfield(0x0ABUL), },
          { name: "x", value: FloatingPoint(1.5), },
          { name: "stamp", value: Time(0UL), },
        ]),
      },
      {
        name: "vector",
        value: Elements([
          FloatingPoint(1.0),
          FloatingPoint(2.0),
          FloatingPoint(3.0),
          FloatingPoint(4.0),
        ]),
      },
      { name: "state", value: Enumeration("OPEN"), },
      { name: "note", value: Text(""), },
    ]),
  )
  assert_true(!source.can_convert_to(target))
  assert_true(
    conversion_test_error(() => {
      source.convert_value(source_value, target) |> ignore
    })
    is Some(
      InvalidArgument("compound conversion is missing target member: stamp")
    ),
  )
}
