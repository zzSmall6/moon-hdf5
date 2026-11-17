///|
/// Controls how a source HDF5 datatype is converted to a requested memory or
/// file datatype.
#derive(Eq, @debug.Debug)
pub(all) enum Hdf5ConversionMode {
  Strict
  Lossy
  NoConversion
}

///|
pub extend Hdf5ConversionMode with Eq::{not_equal, equal}

///|
pub extend Hdf5ConversionMode with @debug.Debug::{to_repr}

///|
/// Options for explicit datatype conversion.
#derive(Eq, @debug.Debug)
pub(all) struct Hdf5ConversionOptions {
  mode : Hdf5ConversionMode
  /// Fills target compound members that have no source member with zero or
  /// empty values of the target member type.
  allow_missing_compound_members : Bool
}

///|
pub extend Hdf5ConversionOptions with Eq::{not_equal, equal}

///|
pub extend Hdf5ConversionOptions with @debug.Debug::{to_repr}

///|
pub fn Hdf5ConversionOptions::strict() -> Hdf5ConversionOptions {
  { mode: Strict, allow_missing_compound_members: false, }
}

///|
pub fn Hdf5ConversionOptions::lossy() -> Hdf5ConversionOptions {
  { mode: Lossy, allow_missing_compound_members: false, }
}

///|
pub fn Hdf5ConversionOptions::none() -> Hdf5ConversionOptions {
  { mode: NoConversion, allow_missing_compound_members: false, }
}

///|
priv enum ConversionNumber {
  ConversionInteger(@bigint.BigInt)
  ConversionFloat(Double)
}

///|
fn conversion_bigint_zero() -> @bigint.BigInt {
  @bigint.BigInt::from_int(0)
}

///|
fn conversion_bigint_one() -> @bigint.BigInt {
  @bigint.BigInt::from_int(1)
}

///|
fn conversion_enum_bigint(value : EnumValue) -> @bigint.BigInt {
  match value {
    Signed(number) => @bigint.BigInt::from_int64(number)
    Unsigned(number) => @bigint.BigInt::from_uint64(number)
    Big(number) => number
  }
}

///|
fn conversion_enum_member_bigint(
  datatype : Datatype,
  name : String,
) -> @bigint.BigInt raise Hdf5Error {
  for enum_item in datatype.enum_members {
    if enum_item.name == name {
      return conversion_enum_bigint(enum_item.value)
    }
  }
  raise InvalidArgument(
    "datatype conversion source enum member was not found: " + name,
  )
}

///|
fn conversion_datatype_equivalent(left : Datatype, right : Datatype) -> Bool {
  if left.class != right.class ||
    left.size != right.size ||
    left.signed != right.signed ||
    left.little_endian != right.little_endian ||
    left.vlen_is_string != right.vlen_is_string ||
    left.bit_offset != right.bit_offset ||
    left.bit_precision != right.bit_precision ||
    left.float_sign_position != right.float_sign_position ||
    left.float_exponent_position != right.float_exponent_position ||
    left.float_exponent_size != right.float_exponent_size ||
    left.float_mantissa_position != right.float_mantissa_position ||
    left.float_mantissa_size != right.float_mantissa_size ||
    left.float_exponent_bias != right.float_exponent_bias ||
    left.float_normalization != right.float_normalization ||
    left.string_padding != right.string_padding ||
    left.string_charset != right.string_charset ||
    left.opaque_tag != right.opaque_tag ||
    left.reference_kind != right.reference_kind ||
    left.array_dimensions != right.array_dimensions ||
    left.base_types.length() != right.base_types.length() ||
    left.compound_members.length() != right.compound_members.length() ||
    left.enum_members.length() != right.enum_members.length() {
    return false
  }
  for index in 0..<left.base_types.length() {
    if !conversion_datatype_equivalent(
        left.base_types[index],
        right.base_types[index],
      ) {
      return false
    }
  }
  for index in 0..<left.compound_members.length() {
    let left_member = left.compound_members[index]
    let right_member = right.compound_members[index]
    if left_member.name != right_member.name ||
      left_member.offset != right_member.offset ||
      !conversion_datatype_equivalent(
        left_member.datatype,
        right_member.datatype,
      ) {
      return false
    }
  }
  for index in 0..<left.enum_members.length() {
    let left_member = left.enum_members[index]
    let right_member = right.enum_members[index]
    if left_member.name != right_member.name ||
      left_member.value != right_member.value {
      return false
    }
  }
  true
}

///|
fn conversion_require_mode_compatible(
  source : Datatype,
  target : Datatype,
  options : Hdf5ConversionOptions,
) -> Unit raise Hdf5Error {
  if options.mode == NoConversion &&
    !conversion_datatype_equivalent(source, target) {
    raise InvalidArgument(
      "no-conversion mode requires identical source and target datatypes",
    )
  }
}

///|
fn conversion_validate_target_datatype(
  datatype : Datatype,
  depth : Int32,
) -> Unit raise Hdf5Error {
  guard depth < 16 && datatype.size > 0 else {
    raise InvalidArgument("conversion target datatype is invalid or too deep")
  }
  match datatype.class {
    0 | 4 => {
      guard datatype.bit_precision > 0 &&
        datatype.bit_offset >= 0 &&
        datatype.bit_offset.to_int64() + datatype.bit_precision.to_int64() <=
        datatype.size.to_int64() * 8L else {
        raise InvalidArgument(
          "conversion target integer or bitfield layout is invalid",
        )
      }
    }
    1 => {
      guard datatype.bit_precision > 0 &&
        datatype.bit_offset >= 0 &&
        datatype.bit_offset.to_int64() + datatype.bit_precision.to_int64() <=
        datatype.size.to_int64() * 8L else {
        raise InvalidArgument("conversion target float layout is invalid")
      }
    }
    2 => {
      guard datatype.bit_precision > 0 &&
        datatype.bit_offset == 0 &&
        datatype.bit_precision.to_int64() <= datatype.size.to_int64() * 8L else {
        raise InvalidArgument("conversion target time layout is invalid")
      }
    }
    3 => {
      guard datatype.string_padding >= 0 &&
        datatype.string_padding <= 2 &&
        datatype.string_charset >= 0 &&
        datatype.string_charset <= 1 else {
        raise InvalidArgument("conversion target string layout is invalid")
      }
    }
    6 => {
      guard datatype.compound_members.length() > 0 else {
        raise InvalidArgument("conversion target compound has no members")
      }
      for compound_member in datatype.compound_members {
        guard compound_member.datatype.size > 0 &&
          compound_member.datatype.size <= datatype.size &&
          compound_member.offset >= 0 &&
          compound_member.offset <=
          datatype.size - compound_member.datatype.size else {
          raise InvalidArgument(
            "conversion target compound member is outside the datatype",
          )
        }
        conversion_validate_target_datatype(compound_member.datatype, depth + 1)
      }
    }
    7 => {
      guard datatype.size > 0 &&
        (
          (
            datatype.version < 4 &&
            (datatype.reference_kind == 0 || datatype.reference_kind == 1)
          ) ||
          (
            datatype.version >= 4 &&
            datatype.reference_kind >= 2 &&
            datatype.reference_kind <= 4
          )
        ) else {
        raise InvalidArgument("conversion target reference layout is invalid")
      }
    }
    8 => {
      guard datatype.base_types.length() == 1 &&
        datatype.base_types[0].class == 0 &&
        datatype.base_types[0].size == datatype.size &&
        datatype.enum_members.length() > 0 else {
        raise InvalidArgument("conversion target enum layout is invalid")
      }
      let base = datatype.base_types[0]
      conversion_validate_target_datatype(base, depth + 1)
      for enum_item in datatype.enum_members {
        guard conversion_integer_fits(
          conversion_enum_bigint(enum_item.value),
          base.bit_precision,
          base.signed,
        ) else {
          raise InvalidArgument(
            "conversion target enum member is outside its integer base",
          )
        }
      }
    }
    9 => {
      guard datatype.base_types.length() == 1 &&
        (
          !datatype.vlen_is_string ||
          (
            datatype.string_padding >= 0 &&
            datatype.string_padding <= 2 &&
            datatype.string_charset >= 0 &&
            datatype.string_charset <= 1
          )
        ) else {
        raise InvalidArgument(
          "conversion target variable-length datatype layout is invalid",
        )
      }
      conversion_validate_target_datatype(datatype.base_types[0], depth + 1)
    }
    10 => {
      guard datatype.base_types.length() == 1 &&
        datatype.array_dimensions.length() > 0 &&
        datatype_array_size(
          datatype.array_dimensions,
          datatype.base_types[0].size,
        ) ==
        datatype.size else {
        raise InvalidArgument("conversion target array layout is invalid")
      }
      conversion_validate_target_datatype(datatype.base_types[0], depth + 1)
    }
    11 => {
      guard datatype.base_types.length() == 1 &&
        datatype.base_types[0].class == 1 &&
        datatype.base_types[0].size * 2 == datatype.size else {
        raise InvalidArgument("conversion target complex layout is invalid")
      }
      conversion_validate_target_datatype(datatype.base_types[0], depth + 1)
    }
    _ => ()
  }
}

///|
fn conversion_numeric_source(datatype : Datatype) -> Bool {
  datatype.class == 0 ||
  datatype.class == 1 ||
  datatype.class == 2 ||
  datatype.class == 4 ||
  datatype.class == 8
}

///|
fn conversion_integer_layout(datatype : Datatype) -> Datatype? {
  if datatype.class == 0 || datatype.class == 2 || datatype.class == 4 {
    Some(datatype)
  } else if datatype.class == 8 && datatype.base_types.length() == 1 {
    Some(datatype.base_types[0])
  } else {
    None
  }
}

///|
fn conversion_float_target_supported(datatype : Datatype) -> Bool {
  is_ieee_float_layout(datatype) || is_x87_float_layout(datatype)
}

///|
fn conversion_reference_type_compatible(
  source : Datatype,
  target : Datatype,
) -> Bool {
  source.class == 7 &&
  target.class == 7 &&
  source.reference_kind == target.reference_kind &&
  source.size == target.size
}

///|
fn conversion_complex_source_components(
  source : Datatype,
) -> (Datatype, Datatype)? {
  match source.class {
    11 =>
      if source.base_types.length() == 1 && source.base_types[0].class == 1 {
        Some((source.base_types[0], source.base_types[0]))
      } else {
        None
      }
    10 =>
      if source.base_types.length() == 1 &&
        conversion_numeric_source(source.base_types[0]) &&
        source.size == source.base_types[0].size * 2 {
        Some((source.base_types[0], source.base_types[0]))
      } else {
        None
      }
    6 => {
      let mut real : Datatype? = None
      let mut imaginary : Datatype? = None
      for field in source.compound_members {
        if field.name == "r" || field.name == "real" || field.name == "re" {
          if conversion_numeric_source(field.datatype) {
            real = Some(field.datatype)
          }
        } else if field.name == "i" ||
          field.name == "imag" ||
          field.name == "im" {
          if conversion_numeric_source(field.datatype) {
            imaginary = Some(field.datatype)
          }
        }
      }
      match (real, imaginary) {
        (Some(real_type), Some(imaginary_type)) =>
          Some((real_type, imaginary_type))
        _ => None
      }
    }
    _ => None
  }
}

///|
fn conversion_complex_source_is_compatible(source : Datatype) -> Bool {
  conversion_complex_source_components(source) is Some(_)
}

///|
fn conversion_complex_components_compatible(
  source : Datatype,
  target_real : Datatype,
  target_imaginary : Datatype,
  strict : Bool,
) -> Bool {
  let source_components = conversion_complex_source_components(source)
  match source_components {
    Some((real, imaginary)) =>
      conversion_float_target_compatible(real, target_real, strict) &&
      conversion_float_target_compatible(imaginary, target_imaginary, strict)
    None => false
  }
}

///|
fn conversion_complex_compound_target_is_compatible(
  source : Datatype,
  target : Datatype,
  strict : Bool,
) -> Bool {
  if target.compound_members.length() != 2 {
    return false
  }
  let mut real : Datatype? = None
  let mut imaginary : Datatype? = None
  for field in target.compound_members {
    if field.datatype.class != 1 ||
      !conversion_float_target_supported(field.datatype) {
      return false
    }
    if field.name == "r" || field.name == "real" || field.name == "re" {
      real = Some(field.datatype)
    } else if field.name == "i" || field.name == "imag" || field.name == "im" {
      imaginary = Some(field.datatype)
    } else {
      return false
    }
  }
  match (real, imaginary) {
    (Some(target_real), Some(target_imaginary)) =>
      conversion_complex_components_compatible(
        source, target_real, target_imaginary, strict,
      )
    _ => false
  }
}

///|
fn conversion_complex_array_target_is_compatible(
  source : Datatype,
  target : Datatype,
  strict : Bool,
) -> Bool {
  target.base_types.length() == 1 &&
  target.base_types[0].class == 1 &&
  conversion_float_target_supported(target.base_types[0]) &&
  target.size == target.base_types[0].size * 2 &&
  conversion_complex_components_compatible(
    source,
    target.base_types[0],
    target.base_types[0],
    strict,
  )
}

///|
fn conversion_complex_target_is_compatible(
  source : Datatype,
  target : Datatype,
  strict : Bool,
) -> Bool {
  target.base_types.length() == 1 &&
  target.base_types[0].class == 1 &&
  conversion_float_target_supported(target.base_types[0]) &&
  conversion_complex_components_compatible(
    source,
    target.base_types[0],
    target.base_types[0],
    strict,
  )
}

///|
fn conversion_default_value_supported(
  datatype : Datatype,
  depth : Int32,
) -> Bool {
  if depth >= 16 || datatype.size <= 0 {
    return false
  }
  match datatype.class {
    6 =>
      datatype.compound_members.all(compound_member => {
        conversion_default_value_supported(compound_member.datatype, depth + 1)
      })
    8 =>
      datatype.enum_members.any(enum_item => {
        conversion_enum_bigint(enum_item.value) == conversion_bigint_zero()
      })
    10 =>
      datatype.base_types.length() == 1 &&
      conversion_default_value_supported(datatype.base_types[0], depth + 1)
    _ => true
  }
}

///|
fn conversion_compound_target_compatible(
  source : Datatype,
  target : Datatype,
  options : Hdf5ConversionOptions,
  depth : Int32,
) -> Bool {
  if source.class != 6 || target.compound_members.length() == 0 {
    return false
  }
  for compound_member in target.compound_members {
    let source_member = source.compound_members.search_by(candidate => {
      candidate.name == compound_member.name
    })
    match source_member {
      Some(index) =>
        if !conversion_type_compatible(
            source.compound_members[index].datatype,
            compound_member.datatype,
            options,
            depth + 1,
          ) {
          return false
        }
      None =>
        if !options.allow_missing_compound_members ||
          !conversion_default_value_supported(
            compound_member.datatype,
            depth + 1,
          ) {
          return false
        }
    }
  }
  true
}

///|
fn conversion_integer_target_compatible(
  source : Datatype,
  target : Datatype,
  strict : Bool,
) -> Bool {
  match conversion_integer_layout(source) {
    Some(layout) =>
      if !strict {
        true
      } else if layout.signed && !target.signed {
        false
      } else {
        let required = if !layout.signed && target.signed {
          layout.bit_precision + 1
        } else {
          layout.bit_precision
        }
        required <= target.bit_precision
      }
    None => !strict && source.class == 1
  }
}

///|
fn conversion_float_target_compatible(
  source : Datatype,
  target : Datatype,
  strict : Bool,
) -> Bool {
  if !conversion_float_target_supported(target) {
    return false
  }
  match conversion_integer_layout(source) {
    Some(layout) =>
      !strict || layout.bit_precision <= target.float_mantissa_size
    None =>
      source.class == 1 &&
      (!strict || source.bit_precision <= target.bit_precision)
  }
}

///|
fn conversion_unsigned_target_compatible(
  source : Datatype,
  target : Datatype,
  strict : Bool,
) -> Bool {
  match conversion_integer_layout(source) {
    Some(layout) =>
      !layout.signed &&
      (!strict || layout.bit_precision <= target.bit_precision)
    None => !strict && source.class == 1
  }
}

///|
fn conversion_source_number(
  source : Datatype,
  value : DatasetValue,
) -> ConversionNumber raise Hdf5Error {
  guard conversion_numeric_source(source) else {
    raise InvalidArgument("datatype conversion source is not numeric")
  }
  match value {
    SignedInteger(number) =>
      ConversionInteger(@bigint.BigInt::from_int64(number))
    UnsignedInteger(number) =>
      ConversionInteger(@bigint.BigInt::from_uint64(number))
    BigInteger(number) => ConversionInteger(number)
    FloatingPoint(number) => ConversionFloat(number)
    Enumeration(name) =>
      ConversionInteger(conversion_enum_member_bigint(source, name))
    Time(number) => ConversionInteger(@bigint.BigInt::from_uint64(number))
    BigTime(number) => ConversionInteger(number)
    Bitfield(number) => ConversionInteger(@bigint.BigInt::from_uint64(number))
    BigBitfield(number) => ConversionInteger(number)
    _ =>
      raise InvalidArgument(
        "datatype conversion value does not match its numeric source",
      )
  }
}

///|
fn conversion_bigint_to_double(number : @bigint.BigInt) -> Double {
  if number.compare(conversion_bigint_zero()) < 0 {
    big_significand_to_double(true, number.neg(), 0)
  } else {
    big_significand_to_double(false, number, 0)
  }
}

///|
fn conversion_double_integer_to_bigint(
  number : Double,
) -> @bigint.BigInt raise Hdf5Error {
  guard !number.is_nan() && !number.is_inf() else {
    raise InvalidArgument("non-finite floating value cannot become an integer")
  }
  guard number.trunc() == number else {
    raise InvalidArgument(
      "floating value has a fractional part and cannot become an integer",
    )
  }
  let bits = number.reinterpret_as_uint64()
  let negative = bits >> 63 != 0UL
  let exponent = ((bits >> 52) & 0x7FFUL).to_int()
  let fraction = bits & ((1UL << 52) - 1UL)
  if exponent == 0 {
    if fraction == 0UL {
      return conversion_bigint_zero()
    }
    raise InvalidArgument("subnormal floating value is not integral")
  }
  let significand = @bigint.BigInt::from_uint64(fraction | (1UL << 52))
  let shift = exponent - 1023 - 52
  let magnitude = if shift >= 0 {
    significand.shl(shift)
  } else {
    significand.shr(-shift)
  }
  if negative {
    magnitude.neg()
  } else {
    magnitude
  }
}

///|
fn conversion_float_to_bigint(
  number : Double,
  strict : Bool,
) -> @bigint.BigInt raise Hdf5Error {
  guard !number.is_nan() && !number.is_inf() else {
    raise InvalidArgument("non-finite floating value cannot become an integer")
  }
  let truncated = number.trunc()
  if strict && truncated != number {
    raise InvalidArgument(
      "strict conversion would discard the floating fractional part",
    )
  }
  conversion_double_integer_to_bigint(truncated)
}

///|
fn conversion_integer_fits(
  number : @bigint.BigInt,
  precision : Int32,
  signed : Bool,
) -> Bool {
  if precision <= 0 {
    return false
  }
  let one = conversion_bigint_one()
  if signed {
    let half = one.shl(precision - 1)
    number.compare(half.neg()) >= 0 && number.compare(half) < 0
  } else {
    number.compare(conversion_bigint_zero()) >= 0 &&
    number.compare(one.shl(precision)) < 0
  }
}

///|
fn conversion_canonical_integer(
  number : @bigint.BigInt,
  precision : Int32,
  signed : Bool,
  mode : Hdf5ConversionMode,
) -> @bigint.BigInt raise Hdf5Error {
  guard precision > 0 else {
    raise InvalidArgument("conversion target integer precision is invalid")
  }
  if mode == Strict {
    guard conversion_integer_fits(number, precision, signed) else {
      raise InvalidArgument("strict integer conversion is out of range")
    }
    return number
  }
  let modulus = conversion_bigint_one().shl(precision)
  let mut remainder = number.mod(modulus)
  if remainder.compare(conversion_bigint_zero()) < 0 {
    remainder = remainder.add(modulus)
  }
  if signed &&
    remainder.compare(conversion_bigint_one().shl(precision - 1)) >= 0 {
    remainder.sub(modulus)
  } else {
    remainder
  }
}

///|
fn conversion_integer_value(
  number : @bigint.BigInt,
  datatype : Datatype,
  signed : Bool,
  mode : Hdf5ConversionMode,
) -> DatasetValue raise Hdf5Error {
  let canonical = conversion_canonical_integer(
    number,
    datatype.bit_precision,
    signed,
    mode,
  )
  if datatype.size > 8 {
    BigInteger(canonical)
  } else if signed {
    SignedInteger(canonical.to_int64())
  } else {
    UnsignedInteger(canonical.to_uint64())
  }
}

///|
fn conversion_round_float(
  number : Double,
  target : Datatype,
) -> Double raise Hdf5Error {
  if is_ieee_float_layout(target) {
    if target.size == 2 {
      let bits = writer_float16_bits_from_double(number)
      float16_to_float(bits).to_double()
    } else if target.size == 4 {
      Float::from_double(number).to_double()
    } else {
      number
    }
  } else if is_x87_float_layout(target) {
    number
  } else {
    raise UnsupportedFeature(
      "conversion target floating-point layout is not standard",
    )
  }
}

///|
fn conversion_integer_exact_in_float(
  number : @bigint.BigInt,
  target : Datatype,
) -> Bool {
  let converted = conversion_bigint_to_double(number)
  if converted.is_nan() || converted.is_inf() {
    return false
  }
  let rounded = conversion_round_float(converted, target) catch {
    _ => return false
  }
  let restored = Some(conversion_double_integer_to_bigint(rounded)) catch {
    _ => None
  }
  match restored {
    Some(value) => value == number
    None => false
  }
}

///|
fn conversion_numeric_to_float(
  source : Datatype,
  value : DatasetValue,
  target : Datatype,
  mode : Hdf5ConversionMode,
) -> Double raise Hdf5Error {
  match conversion_source_number(source, value) {
    ConversionInteger(number) => {
      if mode == Strict && !conversion_integer_exact_in_float(number, target) {
        raise InvalidArgument(
          "strict integer-to-float conversion would lose precision",
        )
      }
      conversion_round_float(conversion_bigint_to_double(number), target)
    }
    ConversionFloat(number) =>
      if number.is_nan() || number.is_inf() {
        number
      } else {
        let rounded = conversion_round_float(number, target)
        if mode == Strict && rounded != number {
          raise InvalidArgument(
            "strict floating-point conversion would lose precision",
          )
        }
        rounded
      }
  }
}

///|
fn conversion_numeric_to_integer(
  source : Datatype,
  value : DatasetValue,
  target : Datatype,
  mode : Hdf5ConversionMode,
) -> DatasetValue raise Hdf5Error {
  let number = match conversion_source_number(source, value) {
    ConversionInteger(number) => number
    ConversionFloat(number) =>
      conversion_float_to_bigint(number, mode == Strict)
  }
  conversion_integer_value(number, target, target.signed, mode)
}

///|
fn conversion_numeric_to_unsigned(
  source : Datatype,
  value : DatasetValue,
  target : Datatype,
  mode : Hdf5ConversionMode,
) -> @bigint.BigInt raise Hdf5Error {
  let number = match conversion_source_number(source, value) {
    ConversionInteger(number) => number
    ConversionFloat(number) =>
      conversion_float_to_bigint(number, mode == Strict)
  }
  conversion_canonical_integer(number, target.bit_precision, false, mode)
}

///|
fn conversion_target_enum(
  source : Datatype,
  value : DatasetValue,
  target : Datatype,
  mode : Hdf5ConversionMode,
) -> DatasetValue raise Hdf5Error {
  let number = match conversion_source_number(source, value) {
    ConversionInteger(number) => number
    ConversionFloat(number) =>
      conversion_float_to_bigint(number, mode == Strict)
  }
  let base = target.base_types[0]
  let canonical = conversion_canonical_integer(
    number,
    base.bit_precision,
    base.signed,
    mode,
  )
  for enum_item in target.enum_members {
    if conversion_enum_bigint(enum_item.value) == canonical {
      return Enumeration(enum_item.name)
    }
  }
  raise InvalidArgument(
    "conversion target enum has no member for the converted integer",
  )
}

///|
fn conversion_ascii_ok(text : String) -> Bool {
  for byte in @utf8.encode(text) {
    if byte.to_int() >= 128 {
      return false
    }
  }
  true
}

///|
fn conversion_truncate_utf8(text : String, width : Int32) -> String {
  if width <= 0 {
    return ""
  }
  let mut output = ""
  let mut used = 0
  for character in text {
    let part = character.to_string()
    let length = @utf8.encode(part).length()
    if used + length > width {
      break
    }
    output = output + part
    used = used + length
  }
  output
}

///|
fn conversion_string_value(
  value : DatasetValue,
  target : Datatype,
  mode : Hdf5ConversionMode,
) -> DatasetValue raise Hdf5Error {
  guard value is Text(text) else {
    raise InvalidArgument("string conversion requires a string value")
  }
  let target_is_vlen = target.class == 9 && target.vlen_is_string
  let result = if !target_is_vlen && @utf8.encode(text).length() > target.size {
    if mode == Strict {
      raise InvalidArgument(
        "strict fixed-string conversion exceeds the target width",
      )
    }
    conversion_truncate_utf8(text, target.size)
  } else {
    text
  }
  if target.string_charset == 0 && !conversion_ascii_ok(result) {
    raise InvalidArgument(
      "conversion target string charset cannot encode non-ASCII text",
    )
  }
  Text(result)
}

///|
fn conversion_target_is_vlen_string(datatype : Datatype) -> Bool {
  datatype.class == 9 && datatype.vlen_is_string
}

///|
fn conversion_complex_components(
  source : Datatype,
  value : DatasetValue,
) -> (ConversionNumber, ConversionNumber) raise Hdf5Error {
  match value {
    Complex(real, imaginary) =>
      (ConversionFloat(real), ConversionFloat(imaginary))
    Compound(fields) => {
      let mut real : ConversionNumber? = None
      let mut imaginary : ConversionNumber? = None
      for field in fields {
        let member_index = source.compound_members.search_by(source_member => {
          source_member.name == field.name
        })
        match member_index {
          Some(index) => {
            let name = source.compound_members[index].name
            let number = conversion_source_number(
              source.compound_members[index].datatype,
              field.value,
            )
            if name == "r" || name == "real" || name == "re" {
              real = Some(number)
            } else if name == "i" || name == "imag" || name == "im" {
              imaginary = Some(number)
            }
          }
          None => ()
        }
      }
      match (real, imaginary) {
        (Some(real_value), Some(imaginary_value)) =>
          (real_value, imaginary_value)
        _ =>
          raise InvalidArgument(
            "complex conversion source is missing real or imaginary fields",
          )
      }
    }
    Elements(items) => {
      guard source.class == 10 &&
        source.base_types.length() == 1 &&
        conversion_numeric_source(source.base_types[0]) &&
        source.size == source.base_types[0].size * 2 &&
        items.length() == 2 else {
        raise InvalidArgument(
          "complex conversion requires a complex, compound or two-element array",
        )
      }
      (
        conversion_source_number(source.base_types[0], items[0]),
        conversion_source_number(source.base_types[0], items[1]),
      )
    }
    _ =>
      raise InvalidArgument(
        "complex conversion requires a complex, compound or two-element array",
      )
  }
}

///|
fn conversion_number_to_target_float(
  number : ConversionNumber,
  target : Datatype,
  mode : Hdf5ConversionMode,
) -> Double raise Hdf5Error {
  match number {
    ConversionInteger(value) => {
      if mode == Strict && !conversion_integer_exact_in_float(value, target) {
        raise InvalidArgument(
          "strict integer-to-float conversion would lose precision",
        )
      }
      conversion_round_float(conversion_bigint_to_double(value), target)
    }
    ConversionFloat(value) =>
      if value.is_nan() || value.is_inf() {
        value
      } else {
        let rounded = conversion_round_float(value, target)
        if mode == Strict && rounded != value {
          raise InvalidArgument(
            "strict floating-point conversion would lose precision",
          )
        }
        rounded
      }
  }
}

///|
fn conversion_default_value(
  datatype : Datatype,
  depth : Int32,
) -> DatasetValue raise Hdf5Error {
  guard depth < 16 else {
    raise InvalidArgument("conversion default value is too deeply nested")
  }
  match datatype.class {
    0 =>
      if datatype.size > 8 {
        BigInteger(conversion_bigint_zero())
      } else if datatype.signed {
        SignedInteger(0L)
      } else {
        UnsignedInteger(0UL)
      }
    1 => FloatingPoint(0.0)
    2 =>
      if datatype.size > 8 {
        BigTime(conversion_bigint_zero())
      } else {
        Time(0UL)
      }
    3 => Text("")
    4 =>
      if datatype.size > 8 {
        BigBitfield(conversion_bigint_zero())
      } else {
        Bitfield(0UL)
      }
    5 => Opaque(Bytes::make(datatype.size, b'\x00'))
    6 =>
      Compound(
        datatype.compound_members.map(compound_member => {
          name: compound_member.name,
          value: conversion_default_value(compound_member.datatype, depth + 1),
        }),
      )
    7 =>
      if datatype.reference_kind == 1 || datatype.reference_kind == 3 {
        RegionReference(None, RegionEmpty)
      } else if datatype.reference_kind == 4 {
        AttributeReference(None, "")
      } else {
        ObjectReference(None)
      }
    8 => {
      let zero = datatype.enum_members.search_by(enum_item => {
        conversion_enum_bigint(enum_item.value) == conversion_bigint_zero()
      })
      guard zero is Some(index) else {
        raise InvalidArgument(
          "conversion default enum member has no zero-valued entry",
        )
      }
      Enumeration(datatype.enum_members[index].name)
    }
    9 => if datatype.vlen_is_string { Text("") } else { Sequence([]) }
    10 => {
      let base = datatype.base_types[0]
      Elements(
        Array::make(
          datatype.size / base.size,
          conversion_default_value(base, depth + 1),
        ),
      )
    }
    11 => Complex(0.0, 0.0)
    _ =>
      raise InvalidArgument(
        "conversion cannot create a default for this datatype",
      )
  }
}

///|
fn conversion_complex_target_value(
  source : Datatype,
  value : DatasetValue,
  target : Datatype,
  mode : Hdf5ConversionMode,
) -> DatasetValue raise Hdf5Error {
  guard target.base_types.length() == 1 else {
    raise InvalidArgument("complex target is missing its base type")
  }
  let (real, imaginary) = conversion_complex_components(source, value)
  let base = target.base_types[0]
  Complex(
    conversion_number_to_target_float(real, base, mode),
    conversion_number_to_target_float(imaginary, base, mode),
  )
}

///|
fn conversion_compound_target_value(
  source : Datatype,
  value : DatasetValue,
  target : Datatype,
  options : Hdf5ConversionOptions,
  depth : Int32,
) -> DatasetValue raise Hdf5Error {
  if conversion_complex_source_is_compatible(source) &&
    conversion_complex_compound_target_is_compatible(
      source,
      target,
      options.mode == Strict,
    ) {
    let (real, imaginary) = conversion_complex_components(source, value)
    let output : Array[CompoundFieldValue] = []
    for compound_member in target.compound_members {
      let number = if compound_member.name == "r" ||
        compound_member.name == "real" ||
        compound_member.name == "re" {
        real
      } else {
        imaginary
      }
      output.push({
        name: compound_member.name,
        value: FloatingPoint(
          conversion_number_to_target_float(
            number,
            compound_member.datatype,
            options.mode,
          ),
        ),
      })
    }
    return Compound(output)
  }
  guard value is Compound(fields) else {
    raise InvalidArgument("compound conversion requires a compound value")
  }
  guard source.class == 6 else {
    raise InvalidArgument(
      "compound conversion requires a compound source datatype",
    )
  }
  let output : Array[CompoundFieldValue] = []
  for target_member in target.compound_members {
    let source_index = fields.search_by(field => {
      field.name == target_member.name
    })
    match source_index {
      Some(index) => {
        let source_member = source.compound_members.search_by(compound_member => {
          compound_member.name == target_member.name
        })
        let source_type = match source_member {
          Some(member_index) => source.compound_members[member_index].datatype
          None =>
            raise InvalidArgument(
              "compound conversion source has no member: " + target_member.name,
            )
        }
        output.push({
          name: target_member.name,
          value: conversion_convert_value(
            source_type,
            fields[index].value,
            target_member.datatype,
            options,
            depth + 1,
          ),
        })
      }
      None =>
        if options.allow_missing_compound_members {
          output.push({
            name: target_member.name,
            value: conversion_default_value(target_member.datatype, depth + 1),
          })
        } else {
          raise InvalidArgument(
            "compound conversion is missing target member: " +
            target_member.name,
          )
        }
    }
  }
  Compound(output)
}

///|
fn conversion_array_target_value(
  source : Datatype,
  value : DatasetValue,
  target : Datatype,
  options : Hdf5ConversionOptions,
  depth : Int32,
) -> DatasetValue raise Hdf5Error {
  guard target.base_types.length() == 1 else {
    raise InvalidArgument("array conversion target is missing its base type")
  }
  let base = target.base_types[0]
  if conversion_complex_array_target_is_compatible(
      source,
      target,
      options.mode == Strict,
    ) {
    let (real, imaginary) = conversion_complex_components(source, value)
    return Elements([
      FloatingPoint(conversion_number_to_target_float(real, base, options.mode)),
      FloatingPoint(
        conversion_number_to_target_float(imaginary, base, options.mode),
      ),
    ])
  }
  guard source.class == 10 &&
    source.base_types.length() == 1 &&
    source.array_dimensions == target.array_dimensions else {
    raise InvalidArgument(
      "array conversion requires matching shapes and base types",
    )
  }
  guard value is Elements(items) else {
    raise InvalidArgument("array conversion requires an element sequence")
  }
  guard items.length() == target.size / base.size else {
    raise InvalidArgument("array conversion value count is invalid")
  }
  let output : Array[DatasetValue] = []
  for item in items {
    output.push(
      conversion_convert_value(
        source.base_types[0],
        item,
        base,
        options,
        depth + 1,
      ),
    )
  }
  Elements(output)
}

///|
fn conversion_sequence_target_value(
  source : Datatype,
  value : DatasetValue,
  target : Datatype,
  options : Hdf5ConversionOptions,
  depth : Int32,
) -> DatasetValue raise Hdf5Error {
  guard source.class == 9 &&
    !source.vlen_is_string &&
    source.base_types.length() == 1 &&
    target.base_types.length() == 1 else {
    raise InvalidArgument(
      "variable-length sequence conversion requires matching base types",
    )
  }
  guard value is Sequence(items) else {
    raise InvalidArgument(
      "variable-length sequence conversion requires a sequence value",
    )
  }
  let output : Array[DatasetValue] = []
  for item in items {
    output.push(
      conversion_convert_value(
        source.base_types[0],
        item,
        target.base_types[0],
        options,
        depth + 1,
      ),
    )
  }
  Sequence(output)
}

///|
fn conversion_reference_target_value(
  source : Datatype,
  value : DatasetValue,
  target : Datatype,
) -> DatasetValue raise Hdf5Error {
  match value {
    Reference(bytes) => {
      guard source.reference_kind == target.reference_kind &&
        bytes.length() == target.size else {
        raise InvalidArgument(
          "reference conversion requires matching kind and width",
        )
      }
      Reference(bytes)
    }
    ObjectReference(_) =>
      if target.reference_kind == 0 ||
        (target.reference_kind == 2 && target.version >= 4) {
        value
      } else {
        raise InvalidArgument("object reference kind does not match the target")
      }
    RegionReference(_, _) =>
      if target.reference_kind == 1 ||
        (target.reference_kind == 3 && target.version >= 4) {
        conversion_copy_value(value)
      } else {
        raise InvalidArgument("region reference kind does not match the target")
      }
    AttributeReference(_, _) =>
      if target.reference_kind == 4 {
        value
      } else {
        raise InvalidArgument(
          "attribute reference kind does not match the target",
        )
      }
    ExternalObjectReference(_, _) =>
      if target.reference_kind == 2 && target.version >= 4 {
        value
      } else {
        raise InvalidArgument(
          "external object reference requires a version-4 object target",
        )
      }
    ExternalRegionReference(_, _, _) =>
      if target.reference_kind == 3 && target.version >= 4 {
        value
      } else {
        raise InvalidArgument(
          "external region reference requires a version-4 region target",
        )
      }
    ExternalAttributeReference(_, _, _) =>
      if target.reference_kind == 4 && target.version >= 4 {
        value
      } else {
        raise InvalidArgument(
          "external attribute reference requires a version-4 attribute target",
        )
      }
    _ =>
      raise InvalidArgument("reference conversion requires a reference value")
  }
}

///|
fn conversion_copy_value(value : DatasetValue) -> DatasetValue {
  match value {
    Compound(fields) =>
      Compound(
        fields.map(field => {
          name: field.name,
          value: conversion_copy_value(field.value),
        }),
      )
    Elements(items) => Elements(items.map(conversion_copy_value))
    Sequence(items) => Sequence(items.map(conversion_copy_value))
    RegionReference(path, selection) =>
      RegionReference(path, copy_region_selection(selection))
    ExternalRegionReference(file_name, path, selection) =>
      ExternalRegionReference(file_name, path, copy_region_selection(selection))
    _ => value
  }
}

///|
fn conversion_convert_value(
  source : Datatype,
  value : DatasetValue,
  target : Datatype,
  options : Hdf5ConversionOptions,
  depth : Int32,
) -> DatasetValue raise Hdf5Error {
  guard depth < 16 else {
    raise InvalidArgument("datatype conversion is too deeply nested")
  }
  if conversion_datatype_equivalent(source, target) {
    return conversion_copy_value(value)
  }
  if options.mode == NoConversion {
    raise InvalidArgument(
      "no-conversion mode requires identical source and target datatypes",
    )
  }
  match target.class {
    0 =>
      if !conversion_numeric_source(source) {
        raise InvalidArgument(
          "integer conversion requires a numeric source datatype",
        )
      } else {
        conversion_numeric_to_integer(source, value, target, options.mode)
      }
    1 => {
      guard conversion_numeric_source(source) else {
        raise InvalidArgument(
          "floating-point conversion requires a numeric source datatype",
        )
      }
      FloatingPoint(
        conversion_numeric_to_float(source, value, target, options.mode),
      )
    }
    2 => {
      let number = conversion_numeric_to_unsigned(
        source,
        value,
        target,
        options.mode,
      )
      if target.size > 8 {
        BigTime(number)
      } else {
        Time(number.to_uint64())
      }
    }
    3 => conversion_string_value(value, target, options.mode)
    4 => {
      let number = conversion_numeric_to_unsigned(
        source,
        value,
        target,
        options.mode,
      )
      if target.size > 8 {
        BigBitfield(number)
      } else {
        Bitfield(number.to_uint64())
      }
    }
    5 => {
      guard source.class == 5 && value is Opaque(bytes) else {
        raise InvalidArgument("opaque conversion requires opaque values")
      }
      if bytes.length() == target.size {
        Opaque(bytes)
      } else if options.mode == Lossy {
        let count = if bytes.length() < target.size {
          bytes.length()
        } else {
          target.size
        }
        let mutable = FixedArray::make(target.size, b'\x00')
        mutable.blit_from_bytes(0, bytes, 0, count)
        Opaque(mutable.unsafe_reinterpret_as_bytes())
      } else {
        raise InvalidArgument(
          "strict opaque conversion requires equal source and target sizes",
        )
      }
    }
    6 => conversion_compound_target_value(source, value, target, options, depth)
    7 => conversion_reference_target_value(source, value, target)
    8 => conversion_target_enum(source, value, target, options.mode)
    9 if target.vlen_is_string =>
      conversion_string_value(value, target, options.mode)
    9 => conversion_sequence_target_value(source, value, target, options, depth)
    10 => conversion_array_target_value(source, value, target, options, depth)
    11 => conversion_complex_target_value(source, value, target, options.mode)
    _ =>
      raise UnsupportedFeature(
        "datatype conversion target class is unsupported",
      )
  }
}

///|
fn conversion_values_to_dataset_values(
  values : DatasetValues,
) -> Array[DatasetValue] {
  match values {
    Integers(items) => items.map(value => SignedInteger(value))
    UnsignedIntegers(items) => items.map(value => UnsignedInteger(value))
    BigIntegers(items) => items.map(value => BigInteger(value))
    Floats(items) => items.map(value => FloatingPoint(value))
    Strings(items) => items.map(value => Text(value))
    Enums(items) => items.map(value => Enumeration(value))
    Structured(items) => items
  }
}

///|
fn conversion_pack_values(
  target : Datatype,
  items : Array[DatasetValue],
) -> DatasetValues raise Hdf5Error {
  if target.class == 0 {
    if target.size > 8 {
      let values : Array[@bigint.BigInt] = []
      for item in items {
        guard item is BigInteger(number) else {
          raise InvalidArgument("converted wide integer has the wrong kind")
        }
        values.push(number)
      }
      return BigIntegers(values)
    } else if target.signed {
      let values : Array[Int64] = []
      for item in items {
        guard item is SignedInteger(number) else {
          raise InvalidArgument("converted signed integer has the wrong kind")
        }
        values.push(number)
      }
      return Integers(values)
    } else {
      let values : Array[UInt64] = []
      for item in items {
        guard item is UnsignedInteger(number) else {
          raise InvalidArgument("converted unsigned integer has the wrong kind")
        }
        values.push(number)
      }
      return UnsignedIntegers(values)
    }
  } else if target.class == 1 {
    let values : Array[Double] = []
    for item in items {
      guard item is FloatingPoint(number) else {
        raise InvalidArgument("converted float has the wrong kind")
      }
      values.push(number)
    }
    return Floats(values)
  } else if target.class == 3 || conversion_target_is_vlen_string(target) {
    let values : Array[String] = []
    for item in items {
      guard item is Text(text) else {
        raise InvalidArgument("converted string has the wrong kind")
      }
      values.push(text)
    }
    return Strings(values)
  } else if target.class == 8 {
    let values : Array[String] = []
    for item in items {
      guard item is Enumeration(name) else {
        raise InvalidArgument("converted enum has the wrong kind")
      }
      values.push(name)
    }
    return Enums(values)
  }
  Structured(items)
}

///|
fn conversion_type_compatible(
  source : Datatype,
  target : Datatype,
  options : Hdf5ConversionOptions,
  depth : Int32,
) -> Bool {
  if depth >= 16 {
    return false
  }
  if conversion_datatype_equivalent(source, target) {
    return true
  }
  if options.mode == NoConversion {
    return false
  }
  let strict = options.mode == Strict
  match target.class {
    0 => conversion_integer_target_compatible(source, target, strict)
    1 => conversion_float_target_compatible(source, target, strict)
    2 | 4 => conversion_unsigned_target_compatible(source, target, strict)
    3 =>
      if source.class == 3 {
        (target.string_charset == 1 || source.string_charset == 0) &&
        (!strict || source.size <= target.size)
      } else {
        source.class == 9 &&
        source.vlen_is_string &&
        !strict &&
        (target.string_charset == 1 || source.string_charset == 0)
      }
    5 => source.class == 5 && (!strict || source.size == target.size)
    6 =>
      conversion_compound_target_compatible(source, target, options, depth) ||
      (
        conversion_complex_source_is_compatible(source) &&
        conversion_complex_compound_target_is_compatible(source, target, strict)
      )
    7 => conversion_reference_type_compatible(source, target)
    8 =>
      if target.base_types.length() != 1 {
        false
      } else if source.class == 8 {
        for enum_item in source.enum_members {
          let matched = target.enum_members.any(target_member => {
            conversion_enum_bigint(target_member.value) ==
            conversion_enum_bigint(enum_item.value)
          })
          if !matched {
            return false
          }
        }
        true
      } else {
        !strict &&
        (
          source.class == 0 ||
          source.class == 1 ||
          source.class == 2 ||
          source.class == 4
        )
      }
    9 if target.vlen_is_string =>
      (source.class == 3 || (source.class == 9 && source.vlen_is_string)) &&
      (target.string_charset == 1 || source.string_charset == 0)
    9 =>
      source.class == 9 &&
      !source.vlen_is_string &&
      !target.vlen_is_string &&
      source.base_types.length() == 1 &&
      target.base_types.length() == 1 &&
      conversion_type_compatible(
        source.base_types[0],
        target.base_types[0],
        options,
        depth + 1,
      )
    10 =>
      if source.class == 10 &&
        source.array_dimensions == target.array_dimensions &&
        source.base_types.length() == 1 &&
        target.base_types.length() == 1 {
        conversion_type_compatible(
          source.base_types[0],
          target.base_types[0],
          options,
          depth + 1,
        )
      } else {
        conversion_complex_array_target_is_compatible(source, target, strict)
      }
    11 => conversion_complex_target_is_compatible(source, target, strict)
    _ => false
  }
}

///|
/// Converts one decoded value from this datatype to a target datatype.
pub fn Datatype::convert_value(
  self : Datatype,
  value : DatasetValue,
  target : Datatype,
  options? : Hdf5ConversionOptions,
) -> DatasetValue raise Hdf5Error {
  let policy = options.unwrap_or(Hdf5ConversionOptions::strict())
  conversion_validate_target_datatype(target, 0)
  conversion_convert_value(self, value, target, policy, 0)
}

///|
/// Converts a decoded value array from this datatype to a target datatype.
/// The returned representation follows the target datatype, so a conversion to
/// a signed integer, float, string, enum or structured type returns the same
/// `DatasetValues` variant as direct decoding of that type.
pub fn Datatype::convert_values(
  self : Datatype,
  values : DatasetValues,
  target : Datatype,
  options? : Hdf5ConversionOptions,
) -> DatasetValues raise Hdf5Error {
  let policy = options.unwrap_or(Hdf5ConversionOptions::strict())
  conversion_validate_target_datatype(target, 0)
  conversion_require_mode_compatible(self, target, policy)
  let source_items = conversion_values_to_dataset_values(values)
  let converted : Array[DatasetValue] = []
  for item in source_items {
    converted.push(conversion_convert_value(self, item, target, policy, 0))
  }
  conversion_pack_values(target, converted)
}

///|
/// Reports whether this datatype can be converted to a target datatype under
/// the supplied policy. Strict mode is conservative and requires type-level
/// losslessness; individual values may still fail range or precision checks.
pub fn Datatype::can_convert_to(
  self : Datatype,
  target : Datatype,
  options? : Hdf5ConversionOptions,
) -> Bool {
  let policy = options.unwrap_or(Hdf5ConversionOptions::strict())
  try {
    conversion_validate_target_datatype(target, 0)
    conversion_type_compatible(self, target, policy, 0)
  } catch {
    _ => false
  }
}

///|
/// Converts a dataset and records the target memory datatype in its metadata.
pub fn Dataset::convert_to(
  self : Dataset,
  target : Datatype,
  options? : Hdf5ConversionOptions,
) -> Dataset raise Hdf5Error {
  {
    dataspace: self.dataspace,
    datatype: target,
    layout: self.layout,
    values: self.datatype.convert_values(self.values, target, options?),
  }
}

///|
/// Converts a dataset selection result and records the target memory datatype.
pub fn DatasetSelectionResult::convert_to(
  self : DatasetSelectionResult,
  target : Datatype,
  options? : Hdf5ConversionOptions,
) -> DatasetSelectionResult raise Hdf5Error {
  {
    dimensions: self.dimensions,
    datatype: target,
    values: self.datatype.convert_values(self.values, target, options?),
  }
}

///|
/// Converts one bounded-memory dataset tile to a memory datatype.
pub fn DatasetChunk::convert_to(
  self : DatasetChunk,
  source_datatype : Datatype,
  target : Datatype,
  options? : Hdf5ConversionOptions,
) -> DatasetChunk raise Hdf5Error {
  {
    offset: self.offset,
    dimensions: self.dimensions,
    values: source_datatype.convert_values(self.values, target, options?),
  }
}

///|
/// Converts one bounded-memory selection page to a memory datatype.
pub fn DatasetSelectionPage::convert_to(
  self : DatasetSelectionPage,
  target : Datatype,
  options? : Hdf5ConversionOptions,
) -> DatasetSelectionPage raise Hdf5Error {
  {
    offset: self.offset,
    datatype: target,
    values: self.datatype.convert_values(self.values, target, options?),
  }
}

///|
/// Converts an attribute and records the target memory datatype.
pub fn Attribute::convert_to(
  self : Attribute,
  target : Datatype,
  options? : Hdf5ConversionOptions,
) -> Attribute raise Hdf5Error {
  let policy = options.unwrap_or(Hdf5ConversionOptions::strict())
  conversion_validate_target_datatype(target, 0)
  conversion_require_mode_compatible(self.datatype, target, policy)
  let values = match self.values {
    Some(items) => Some(self.datatype.convert_values(items, target, options?))
    None => None
  }
  { name: self.name, dataspace: self.dataspace, datatype: target, values, }
}

///|
fn Hdf5File::attribute_by_name(
  self : Hdf5File,
  path : String,
  name : String,
) -> Attribute raise Hdf5Error {
  for attribute in self.attributes(path) {
    if attribute.name == name {
      return attribute
    }
  }
  raise InvalidArgument("attribute was not found: " + path + ":" + name)
}

///|
/// Reads a dataset and converts its values to a requested memory datatype.
pub fn Hdf5File::read_dataset_as(
  self : Hdf5File,
  path : String,
  memory_datatype : Datatype,
  options? : Hdf5ConversionOptions,
) -> Dataset raise Hdf5Error {
  self.read_dataset(path).convert_to(memory_datatype, options?)
}

///|
/// Reads a dataset selection and converts its values to a memory datatype.
pub fn Hdf5File::read_dataset_selection_as(
  self : Hdf5File,
  path : String,
  selection : DatasetSelection,
  memory_datatype : Datatype,
  options? : Hdf5ConversionOptions,
) -> DatasetSelectionResult raise Hdf5Error {
  self
  .read_dataset_selection(path, selection)
  .convert_to(memory_datatype, options?)
}

///|
/// Reads one named attribute from an object.
pub fn Hdf5File::read_attribute(
  self : Hdf5File,
  path : String,
  name : String,
) -> Attribute raise Hdf5Error {
  self.attribute_by_name(path, name)
}

///|
/// Reads one named attribute and converts it to a memory datatype.
pub fn Hdf5File::read_attribute_as(
  self : Hdf5File,
  path : String,
  name : String,
  memory_datatype : Datatype,
  options? : Hdf5ConversionOptions,
) -> Attribute raise Hdf5Error {
  self.attribute_by_name(path, name).convert_to(memory_datatype, options?)
}

///|
/// Returns whether a path resolves to an object in the local file.
pub fn Hdf5File::exists(self : Hdf5File, path : String) -> Bool {
  try {
    ignore(self.resolve(path))
    true
  } catch {
    _ => false
  }
}

///|
/// Returns the kind of object stored at a path.
pub fn Hdf5File::object_kind_at(
  self : Hdf5File,
  path : String,
) -> Hdf5ObjectKind raise Hdf5Error {
  self.resolve(path).kind
}

///|
fn Hdf5Resolver::attribute_by_name(
  self : Hdf5Resolver,
  path : String,
  name : String,
) -> Attribute raise Hdf5Error {
  for attribute in self.attributes(path) {
    if attribute.name == name {
      return attribute
    }
  }
  raise InvalidArgument("attribute was not found: " + path + ":" + name)
}

///|
/// Resolves and reads a dataset, converting values to a memory datatype.
pub fn Hdf5Resolver::read_dataset_as(
  self : Hdf5Resolver,
  path : String,
  memory_datatype : Datatype,
  options? : Hdf5ConversionOptions,
) -> Dataset raise Hdf5Error {
  self.read_dataset(path).convert_to(memory_datatype, options?)
}

///|
/// Resolves and reads a dataset selection, converting to a memory datatype.
pub fn Hdf5Resolver::read_dataset_selection_as(
  self : Hdf5Resolver,
  path : String,
  selection : DatasetSelection,
  memory_datatype : Datatype,
  options? : Hdf5ConversionOptions,
) -> DatasetSelectionResult raise Hdf5Error {
  self
  .read_dataset_selection(path, selection)
  .convert_to(memory_datatype, options?)
}

///|
/// Resolves and reads one named attribute.
pub fn Hdf5Resolver::read_attribute(
  self : Hdf5Resolver,
  path : String,
  name : String,
) -> Attribute raise Hdf5Error {
  self.attribute_by_name(path, name)
}

///|
/// Resolves and reads one named attribute with datatype conversion.
pub fn Hdf5Resolver::read_attribute_as(
  self : Hdf5Resolver,
  path : String,
  name : String,
  memory_datatype : Datatype,
  options? : Hdf5ConversionOptions,
) -> Attribute raise Hdf5Error {
  self.attribute_by_name(path, name).convert_to(memory_datatype, options?)
}

///|
/// Returns whether a path resolves through the resolver.
pub fn Hdf5Resolver::exists(self : Hdf5Resolver, path : String) -> Bool {
  try {
    ignore(self.resolve(path))
    true
  } catch {
    _ => false
  }
}

///|
/// Returns the resolved object kind at a path.
pub fn Hdf5Resolver::object_kind_at(
  self : Hdf5Resolver,
  path : String,
) -> Hdf5ObjectKind raise Hdf5Error {
  self.resolve(path).kind
}

///|
/// Returns the logical memory datatype represented by writer values.
pub fn Hdf5WriteValues::memory_datatype(
  self : Hdf5WriteValues,
) -> Datatype raise Hdf5Error {
  match self {
    Complex64Values(_) => Datatype::write_complex(Datatype::write_float(4))
    Complex128Values(_) => Datatype::write_complex(Datatype::write_float(8))
    _ => writer_values_datatype(self)
  }
}

///|
/// Converts writer values to the generic structured value representation.
pub fn Hdf5WriteValues::to_dataset_values(
  self : Hdf5WriteValues,
) -> Array[DatasetValue] {
  match self {
    Int8Values(items) => items.map(value => SignedInteger(value.to_int64()))
    Int16Values(items) => items.map(value => SignedInteger(value.to_int64()))
    Int32Values(items) => items.map(value => SignedInteger(value.to_int64()))
    Int64Values(items) => items.map(value => SignedInteger(value))
    UInt8Values(items) => items.map(value => UnsignedInteger(value.to_uint64()))
    UInt16Values(items) =>
      items.map(value => UnsignedInteger(value.to_uint64()))
    UInt32Values(items) =>
      items.map(value => UnsignedInteger(value.to_uint64()))
    UInt64Values(items) => items.map(value => UnsignedInteger(value))
    BigIntegerValues(items, _, _) => items.map(value => BigInteger(value))
    Float16Values(items) => items.map(value => FloatingPoint(value.to_double()))
    Float32Values(items) => items.map(value => FloatingPoint(value.to_double()))
    Float64Values(items) => items.map(value => FloatingPoint(value))
    FixedStringValues(items, _) => items.map(value => Text(value))
    VariableStringValues(items) => items.map(value => Text(value))
    ObjectReferenceValues(items) => items.map(value => ObjectReference(value))
    Complex64Values(items) =>
      items.map(value => Complex(value.0.to_double(), value.1.to_double()))
    Complex128Values(items) => items.map(value => Complex(value.0, value.1))
    StructuredValues(_, items) => items.map(conversion_copy_value)
  }
}

///|
/// Converts writer values from their logical memory datatype to an explicit
/// file datatype. The result uses `StructuredValues` so the target datatype is
/// preserved exactly during encoding.
pub fn Hdf5WriteValues::convert_to(
  self : Hdf5WriteValues,
  file_datatype : Datatype,
  options? : Hdf5ConversionOptions,
) -> Hdf5WriteValues raise Hdf5Error {
  let source = self.memory_datatype()
  let policy = options.unwrap_or(Hdf5ConversionOptions::strict())
  conversion_validate_target_datatype(file_datatype, 0)
  conversion_require_mode_compatible(source, file_datatype, policy)
  let converted : Array[DatasetValue] = []
  for value in self.to_dataset_values() {
    converted.push(source.convert_value(value, file_datatype, options?))
  }
  StructuredValues(file_datatype, converted)
}

///|
fn conversion_writer_values(
  values : Hdf5WriteValues,
  file_datatype : Datatype,
  options : Hdf5ConversionOptions?,
) -> Hdf5WriteValues raise Hdf5Error {
  values.convert_to(file_datatype, options?)
}

///|
fn conversion_writer_fill(
  fill_value : Hdf5WriteValues?,
  file_datatype : Datatype,
  options : Hdf5ConversionOptions?,
) -> Hdf5WriteValues? raise Hdf5Error {
  match fill_value {
    Some(value) => Some(value.convert_to(file_datatype, options?))
    None => None
  }
}

///|
/// Adds a contiguous dataset with explicit memory-to-file datatype conversion.
pub fn Hdf5Writer::add_dataset_typed(
  self : Hdf5Writer,
  path : String,
  dimensions : Array[UInt64],
  file_datatype : Datatype,
  values : Hdf5WriteValues,
  fill_value? : Hdf5WriteValues,
  options? : Hdf5ConversionOptions,
) -> Unit raise Hdf5Error {
  let converted = conversion_writer_values(values, file_datatype, options)
  let converted_fill = conversion_writer_fill(
    fill_value, file_datatype, options,
  )
  self.add_dataset(path, dimensions, converted, fill_value?=converted_fill)
}

///|
/// Adds a compact dataset with explicit memory-to-file datatype conversion.
pub fn Hdf5Writer::add_compact_dataset_typed(
  self : Hdf5Writer,
  path : String,
  dimensions : Array[UInt64],
  file_datatype : Datatype,
  values : Hdf5WriteValues,
  fill_value? : Hdf5WriteValues,
  options? : Hdf5ConversionOptions,
) -> Unit raise Hdf5Error {
  let converted = conversion_writer_values(values, file_datatype, options)
  let converted_fill = conversion_writer_fill(
    fill_value, file_datatype, options,
  )
  self.add_compact_dataset(
    path,
    dimensions,
    converted,
    fill_value?=converted_fill,
  )
}

///|
/// Adds an external-storage dataset with explicit datatype conversion.
pub fn Hdf5Writer::add_external_dataset_typed(
  self : Hdf5Writer,
  path : String,
  dimensions : Array[UInt64],
  file_datatype : Datatype,
  values : Hdf5WriteValues,
  slots : Array[Hdf5ExternalSlot],
  fill_value? : Hdf5WriteValues,
  options? : Hdf5ConversionOptions,
) -> Unit raise Hdf5Error {
  let converted = conversion_writer_values(values, file_datatype, options)
  let converted_fill = conversion_writer_fill(
    fill_value, file_datatype, options,
  )
  self.add_external_dataset(
    path,
    dimensions,
    converted,
    slots,
    fill_value?=converted_fill,
  )
}

///|
/// Adds a finite chunked dataset with explicit datatype conversion.
pub fn Hdf5Writer::add_chunked_dataset_typed(
  self : Hdf5Writer,
  path : String,
  dimensions : Array[UInt64],
  chunk_dimensions : Array[UInt32],
  file_datatype : Datatype,
  values : Hdf5WriteValues,
  fill_value? : Hdf5WriteValues,
  filters~ : Array[Hdf5WriteFilter] = [],
  options? : Hdf5ConversionOptions,
) -> Unit raise Hdf5Error {
  let converted = conversion_writer_values(values, file_datatype, options)
  let converted_fill = conversion_writer_fill(
    fill_value, file_datatype, options,
  )
  self.add_chunked_dataset(
    path,
    dimensions,
    chunk_dimensions,
    converted,
    fill_value?=converted_fill,
    filters~,
  )
}

///|
/// Adds an extensible dataset with explicit datatype conversion.
pub fn Hdf5Writer::add_extensible_dataset_typed(
  self : Hdf5Writer,
  path : String,
  dimensions : Array[UInt64],
  maximum_dimensions : Array[UInt64?],
  chunk_dimensions : Array[UInt32],
  file_datatype : Datatype,
  values : Hdf5WriteValues,
  fill_value? : Hdf5WriteValues,
  filters~ : Array[Hdf5WriteFilter] = [],
  options? : Hdf5ConversionOptions,
) -> Unit raise Hdf5Error {
  let converted = conversion_writer_values(values, file_datatype, options)
  let converted_fill = conversion_writer_fill(
    fill_value, file_datatype, options,
  )
  self.add_extensible_dataset(
    path,
    dimensions,
    maximum_dimensions,
    chunk_dimensions,
    converted,
    fill_value?=converted_fill,
    filters~,
  )
}

///|
/// Writes a dataset slab with explicit datatype conversion.
pub fn Hdf5Writer::write_dataset_slab_typed(
  self : Hdf5Writer,
  path : String,
  offsets : Array[UInt64],
  dimensions : Array[UInt64],
  file_datatype : Datatype,
  values : Hdf5WriteValues,
  options? : Hdf5ConversionOptions,
) -> Unit raise Hdf5Error {
  self.write_dataset_slab(
    path,
    offsets,
    dimensions,
    conversion_writer_values(values, file_datatype, options),
  )
}

///|
/// Adds an attribute with explicit datatype conversion.
pub fn Hdf5Writer::add_attribute_typed(
  self : Hdf5Writer,
  object_path : String,
  name : String,
  dimensions : Array[UInt64],
  file_datatype : Datatype,
  values : Hdf5WriteValues,
  options? : Hdf5ConversionOptions,
) -> Unit raise Hdf5Error {
  self.add_attribute(
    object_path,
    name,
    dimensions,
    conversion_writer_values(values, file_datatype, options),
  )
}

///|
/// Adds a dataset to an existing file with explicit datatype conversion.
pub fn Hdf5File::add_dataset_typed(
  self : Hdf5File,
  parent_path : String,
  name : String,
  dimensions : Array[UInt64],
  file_datatype : Datatype,
  values : Hdf5WriteValues,
  fill_value? : Hdf5WriteValues,
  options? : Hdf5ConversionOptions,
) -> Bytes raise Hdf5Error {
  let converted = conversion_writer_values(values, file_datatype, options)
  let converted_fill = conversion_writer_fill(
    fill_value, file_datatype, options,
  )
  self.add_dataset(
    parent_path,
    name,
    dimensions,
    converted,
    fill_value?=converted_fill,
  )
}

///|
/// Adds an attribute to an existing file with explicit datatype conversion.
pub fn Hdf5File::add_attribute_typed(
  self : Hdf5File,
  path : String,
  name : String,
  dimensions : Array[UInt64],
  file_datatype : Datatype,
  values : Hdf5WriteValues,
  options? : Hdf5ConversionOptions,
) -> Bytes raise Hdf5Error {
  self.add_attribute(
    path,
    name,
    dimensions,
    conversion_writer_values(values, file_datatype, options),
  )
}

///|
/// Patches existing dataset values with explicit datatype conversion.
pub fn Hdf5File::patch_dataset_values_typed(
  self : Hdf5File,
  path : String,
  file_datatype : Datatype,
  values : Hdf5WriteValues,
  options? : Hdf5ConversionOptions,
) -> Bytes raise Hdf5Error {
  self.patch_dataset_values(
    path,
    conversion_writer_values(values, file_datatype, options),
  )
}

///|
fn conversion_writer_scalar_fill(
  fill_value : Hdf5WriteValues?,
  file_datatype : Datatype,
  options : Hdf5ConversionOptions?,
) -> DatasetValue? raise Hdf5Error {
  match fill_value {
    Some(value) => {
      let source = value.memory_datatype()
      let items = value.to_dataset_values()
      guard items.length() == 1 else {
        raise InvalidArgument("fill value must contain exactly one value")
      }
      Some(source.convert_value(items[0], file_datatype, options?))
    }
    None => None
  }
}

///|
/// Adds a virtual dataset with a converted fill value.
pub fn Hdf5Writer::add_virtual_dataset_typed(
  self : Hdf5Writer,
  path : String,
  dimensions : Array[UInt64],
  file_datatype : Datatype,
  mappings : Array[Hdf5VirtualMapping],
  fill_value? : Hdf5WriteValues,
  maximum_dimensions? : Array[UInt64?],
  options? : Hdf5ConversionOptions,
) -> Unit raise Hdf5Error {
  let converted_fill = conversion_writer_scalar_fill(
    fill_value, file_datatype, options,
  )
  self.add_virtual_dataset(
    path,
    dimensions,
    file_datatype,
    mappings,
    fill_value?=converted_fill,
    maximum_dimensions?,
  )
}

///|
/// Appends a block to a typed extensible writer dataset.
pub fn Hdf5Writer::append_dataset_typed(
  self : Hdf5Writer,
  path : String,
  dimensions : Array[UInt64],
  values : Hdf5WriteValues,
  options? : Hdf5ConversionOptions,
) -> Unit raise Hdf5Error {
  let index = self.writer_dataset_index(path)
  let file_datatype = writer_values_datatype(self.datasets[index].values)
  self.append_dataset(
    path,
    dimensions,
    conversion_writer_values(values, file_datatype, options),
  )
}

///|
/// Adds a compact dataset to an existing file with datatype conversion.
pub fn Hdf5File::add_compact_dataset_typed(
  self : Hdf5File,
  parent_path : String,
  name : String,
  dimensions : Array[UInt64],
  file_datatype : Datatype,
  values : Hdf5WriteValues,
  fill_value? : Hdf5WriteValues,
  options? : Hdf5ConversionOptions,
) -> Bytes raise Hdf5Error {
  let converted = conversion_writer_values(values, file_datatype, options)
  let converted_fill = conversion_writer_fill(
    fill_value, file_datatype, options,
  )
  self.add_compact_dataset(
    parent_path,
    name,
    dimensions,
    converted,
    fill_value?=converted_fill,
  )
}

///|
/// Adds a chunked dataset to an existing file with datatype conversion.
pub fn Hdf5File::add_chunked_dataset_typed(
  self : Hdf5File,
  parent_path : String,
  name : String,
  dimensions : Array[UInt64],
  chunk_dimensions : Array[UInt32],
  file_datatype : Datatype,
  values : Hdf5WriteValues,
  fill_value? : Hdf5WriteValues,
  maximum_dimensions? : Array[UInt64?],
  filters~ : Array[Hdf5WriteFilter] = [],
  custom_filters~ : Array[&Hdf5CustomFilter] = [],
  options? : Hdf5ConversionOptions,
) -> Bytes raise Hdf5Error {
  let converted = conversion_writer_values(values, file_datatype, options)
  let converted_fill = conversion_writer_fill(
    fill_value, file_datatype, options,
  )
  self.add_chunked_dataset(
    parent_path,
    name,
    dimensions,
    chunk_dimensions,
    converted,
    fill_value?=converted_fill,
    maximum_dimensions?,
    filters~,
    custom_filters~,
  )
}

///|
/// Adds an external-storage dataset to an existing file with conversion.
pub fn Hdf5File::add_external_dataset_typed(
  self : Hdf5File,
  parent_path : String,
  name : String,
  dimensions : Array[UInt64],
  file_datatype : Datatype,
  values : Hdf5WriteValues,
  slots : Array[Hdf5ExternalSlot],
  fill_value? : Hdf5WriteValues,
  options? : Hdf5ConversionOptions,
) -> Bytes raise Hdf5Error {
  let converted = conversion_writer_values(values, file_datatype, options)
  let converted_fill = conversion_writer_fill(
    fill_value, file_datatype, options,
  )
  self.add_external_dataset(
    parent_path,
    name,
    dimensions,
    converted,
    slots,
    fill_value?=converted_fill,
  )
}

///|
/// Adds a virtual dataset to an existing file with a converted fill value.
pub fn Hdf5File::add_virtual_dataset_typed(
  self : Hdf5File,
  parent_path : String,
  name : String,
  dimensions : Array[UInt64],
  file_datatype : Datatype,
  mappings : Array[Hdf5VirtualMapping],
  fill_value? : Hdf5WriteValues,
  maximum_dimensions? : Array[UInt64?],
  options? : Hdf5ConversionOptions,
) -> Bytes raise Hdf5Error {
  let converted_fill = conversion_writer_scalar_fill(
    fill_value, file_datatype, options,
  )
  self.add_virtual_dataset(
    parent_path,
    name,
    dimensions,
    file_datatype,
    mappings,
    fill_value?=converted_fill,
    maximum_dimensions?,
  )
}

///|
/// Patches an existing attribute after converting from writer memory values.
pub fn Hdf5File::patch_attribute_values_typed(
  self : Hdf5File,
  path : String,
  name : String,
  values : Hdf5WriteValues,
  options? : Hdf5ConversionOptions,
) -> Bytes raise Hdf5Error {
  let attribute = self.attribute_by_name(path, name)
  self.patch_attribute_values(
    path,
    name,
    conversion_writer_values(values, attribute.datatype, options),
  )
}

///|
/// Patches an allocated contiguous dataset using its stored datatype.
pub fn Hdf5File::patch_contiguous_dataset_typed(
  self : Hdf5File,
  path : String,
  values : Hdf5WriteValues,
  options? : Hdf5ConversionOptions,
) -> Bytes raise Hdf5Error {
  let file_datatype = self.read_dataset_metadata(path).datatype
  self.patch_contiguous_dataset(
    path,
    conversion_writer_values(values, file_datatype, options),
  )
}

///|
/// Appends a block to an existing chunked dataset using its stored datatype.
pub fn Hdf5File::append_dataset_typed(
  self : Hdf5File,
  path : String,
  dimensions : Array[UInt64],
  values : Hdf5WriteValues,
  options? : Hdf5ConversionOptions,
) -> Bytes raise Hdf5Error {
  let file_datatype = self.read_dataset_metadata(path).datatype
  self.append_dataset(
    path,
    dimensions,
    conversion_writer_values(values, file_datatype, options),
  )
}

///|
/// Iterates over dataset tiles while converting each tile to a memory type.
pub struct DatasetChunkConversionIterator {
  priv inner : DatasetChunkIterator
  priv source_datatype : Datatype
  priv target_datatype : Datatype
  priv options : Hdf5ConversionOptions
}

///|
/// Returns the next converted tile, or `None` after the dataset is consumed.
pub fn DatasetChunkConversionIterator::next(
  self : DatasetChunkConversionIterator,
) -> DatasetChunk? raise Hdf5Error {
  match self.inner.next() {
    None => None
    Some(chunk) => {
      let converted = chunk.values
      Some({
        offset: chunk.offset,
        dimensions: chunk.dimensions,
        values: self.source_datatype.convert_values(
          converted,
          self.target_datatype,
          options=self.options,
        ),
      })
    }
  }
}

///|
/// Creates a bounded-memory tile iterator that converts each tile to a memory
/// datatype.
pub fn Hdf5File::chunk_iterator_as(
  self : Hdf5File,
  path : String,
  tile_dimensions : Array[UInt64],
  memory_datatype : Datatype,
  options? : Hdf5ConversionOptions,
) -> DatasetChunkConversionIterator raise Hdf5Error {
  let inner = self.chunk_iterator(path, tile_dimensions)
  let target = memory_datatype
  let policy = options.unwrap_or(Hdf5ConversionOptions::strict())
  conversion_validate_target_datatype(target, 0)
  conversion_require_mode_compatible(inner.metadata.datatype, target, policy)
  {
    inner,
    source_datatype: inner.metadata.datatype,
    target_datatype: target,
    options: policy,
  }
}

///|
/// Creates a resolver-backed tile iterator that converts each tile to a memory
/// datatype.
pub fn Hdf5Resolver::chunk_iterator_as(
  self : Hdf5Resolver,
  path : String,
  tile_dimensions : Array[UInt64],
  memory_datatype : Datatype,
  options? : Hdf5ConversionOptions,
) -> DatasetChunkConversionIterator raise Hdf5Error {
  let inner = self.chunk_iterator(path, tile_dimensions)
  let target = memory_datatype
  let policy = options.unwrap_or(Hdf5ConversionOptions::strict())
  conversion_validate_target_datatype(target, 0)
  conversion_require_mode_compatible(inner.metadata.datatype, target, policy)
  {
    inner,
    source_datatype: inner.metadata.datatype,
    target_datatype: target,
    options: policy,
  }
}

///|
/// Iterates over selection pages while converting each page to a memory type.
pub struct DatasetSelectionConversionIterator {
  priv inner : DatasetSelectionIterator
  priv target_datatype : Datatype
  priv options : Hdf5ConversionOptions
}

///|
/// Returns the next converted selection page, or `None` when exhausted.
pub fn DatasetSelectionConversionIterator::next(
  self : DatasetSelectionConversionIterator,
) -> DatasetSelectionPage? raise Hdf5Error {
  match self.inner.next() {
    None => None
    Some(page) => {
      let converted = page.values
      Some({
        offset: page.offset,
        datatype: self.target_datatype,
        values: page.datatype.convert_values(
          converted,
          self.target_datatype,
          options=self.options,
        ),
      })
    }
  }
}

///|
/// Creates a bounded-memory selection iterator that converts every page to a
/// memory datatype.
pub fn Hdf5File::selection_iterator_as(
  self : Hdf5File,
  path : String,
  selection : DatasetSelection,
  page_size : Int32,
  memory_datatype : Datatype,
  options? : Hdf5ConversionOptions,
) -> DatasetSelectionConversionIterator raise Hdf5Error {
  let inner = self.selection_iterator(path, selection, page_size)
  let target = memory_datatype
  let policy = options.unwrap_or(Hdf5ConversionOptions::strict())
  conversion_validate_target_datatype(target, 0)
  conversion_require_mode_compatible(inner.metadata.datatype, target, policy)
  { inner, target_datatype: target, options: policy, }
}

///|
/// Creates a resolver-backed selection iterator that converts every page to a
/// memory datatype.
pub fn Hdf5Resolver::selection_iterator_as(
  self : Hdf5Resolver,
  path : String,
  selection : DatasetSelection,
  page_size : Int32,
  memory_datatype : Datatype,
  options? : Hdf5ConversionOptions,
) -> DatasetSelectionConversionIterator raise Hdf5Error {
  let inner = self.selection_iterator(path, selection, page_size)
  let target = memory_datatype
  let policy = options.unwrap_or(Hdf5ConversionOptions::strict())
  conversion_validate_target_datatype(target, 0)
  conversion_require_mode_compatible(inner.metadata.datatype, target, policy)
  { inner, target_datatype: target, options: policy, }
}
