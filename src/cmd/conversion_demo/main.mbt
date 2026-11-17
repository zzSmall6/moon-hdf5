///|
fn conversion_large_integer(exponent : Int32) -> @bigint.BigInt {
  @bigint.BigInt::from_int(1).shl(exponent)
}

///|
fn conversion_structured_values(
  values : @lib.DatasetValues,
) -> Array[@lib.DatasetValue] {
  match values {
    Structured(items) => items
    _ => abort("expected structured conversion values")
  }
}

///|
fn conversion_integer_values(values : @lib.DatasetValues) -> Array[Int64] {
  match values {
    Integers(items) => items
    _ => abort("expected converted integers")
  }
}

///|
fn conversion_memory_values(
  datatype : @lib.Datatype,
  values : @lib.DatasetValues,
) -> @lib.Hdf5WriteValues {
  match values {
    Integers(items) => @lib.Int64Values(items)
    UnsignedIntegers(items) => @lib.UInt64Values(items)
    BigIntegers(items) =>
      @lib.BigIntegerValues(items, datatype.size, datatype.signed)
    Floats(items) => @lib.Float64Values(items)
    Structured(items) => @lib.StructuredValues(datatype, items)
    _ => abort("unsupported converted memory values")
  }
}

///|
fn conversion_copy_dataset_typed(
  writer : @lib.Hdf5Writer,
  target_path : String,
  fixture : @lib.Hdf5File,
  source_path : String,
  target_datatype : @lib.Datatype,
  options? : @lib.Hdf5ConversionOptions,
) -> Unit raise @lib.Hdf5Error {
  let converted = fixture.read_dataset_as(
    source_path,
    target_datatype,
    options?,
  )
  writer.add_dataset_typed(
    target_path,
    converted.dataspace.dimensions,
    target_datatype,
    conversion_memory_values(target_datatype, converted.values),
  )
}

///|
fn conversion_string_values(values : @lib.DatasetValues) -> Array[String] {
  match values {
    Strings(items) => items
    _ => abort("expected converted strings")
  }
}

///|
fn conversion_state_type() -> @lib.Datatype raise @lib.Hdf5Error {
  @lib.Datatype::write_enum(@lib.Datatype::write_integer(false, 1), [
    { name: "UNKNOWN", value: @lib.EnumValue::Unsigned(0UL), },
    { name: "OPEN", value: @lib.EnumValue::Unsigned(1UL), },
    { name: "CLOSED", value: @lib.EnumValue::Unsigned(2UL), },
  ])
}

///|
fn conversion_generated_record_type() -> @lib.Datatype raise @lib.Hdf5Error {
  let float64 = @lib.Datatype::write_float(8)
  @lib.Datatype::write_compound([
    ("weight", float64),
    ("id", @lib.Datatype::write_integer(true, 8)),
    ("state", conversion_state_type()),
    ("note", @lib.Datatype::write_fixed_string(6)),
  ])
}

///|
fn conversion_imported_record_type() -> @lib.Datatype raise @lib.Hdf5Error {
  @lib.Datatype::write_compound([
    ("label", @lib.Datatype::write_fixed_string(6)),
    ("id", @lib.Datatype::write_integer(true, 8)),
    ("extra", @lib.Datatype::write_fixed_string(4)),
    ("state", conversion_state_type()),
    (
      "samples",
      @lib.Datatype::write_array(@lib.Datatype::write_float(8), [2UL, 2UL]),
    ),
  ])
}

///|
fn conversion_sequence_type() -> @lib.Datatype raise @lib.Hdf5Error {
  @lib.Datatype::write_vlen_sequence(
    @lib.Datatype::write_compound([
      ("id", @lib.Datatype::write_integer(true, 8)),
      ("value", @lib.Datatype::write_float(8)),
    ]),
  )
}

///|
fn conversion_double(bits : UInt64) -> Double {
  bits.reinterpret_as_double()
}

///|
fn conversion_collect_chunk_values(
  file : @lib.Hdf5File,
  path : String,
  tile : Array[UInt64],
  target : @lib.Datatype,
) -> Array[Int64] raise @lib.Hdf5Error {
  let iterator = file.chunk_iterator_as(path, tile, target)
  let values : Array[Int64] = []
  while true {
    match iterator.next() {
      Some(chunk) => values.append(conversion_integer_values(chunk.values))
      None => break
    }
  }
  values
}

///|
fn conversion_collect_selection_values(
  file : @lib.Hdf5File,
  path : String,
  selection : @lib.DatasetSelection,
  page_size : Int32,
  target : @lib.Datatype,
) -> Array[Int64] raise @lib.Hdf5Error {
  let iterator = file.selection_iterator_as(path, selection, page_size, target)
  let values : Array[Int64] = []
  while true {
    match iterator.next() {
      Some(page) => values.append(conversion_integer_values(page.values))
      None => break
    }
  }
  values
}

///|
fn conversion_write_generated(
  writer : @lib.Hdf5Writer,
) -> Unit raise @lib.Hdf5Error {
  writer.add_group("/generated")
  let missing : @lib.Hdf5ConversionOptions = {
    mode: @lib.Hdf5ConversionMode::Strict,
    allow_missing_compound_members: true,
  }
  writer.add_dataset_typed(
    "/generated/narrowed",
    [3UL],
    @lib.Datatype::write_integer(false, 1),
    @lib.Int64Values([1L, 2L, 255L]),
  )
  writer.add_attribute_typed(
    "/generated/narrowed",
    "scale",
    [],
    @lib.Datatype::write_float(8),
    @lib.Int32Values([2]),
  )
  writer.add_dataset_typed(
    "/generated/lossy",
    [2UL],
    @lib.Datatype::write_integer(false, 1),
    @lib.Int64Values([300L, -1L]),
    options=@lib.Hdf5ConversionOptions::lossy(),
  )
  writer.add_dataset_typed(
    "/generated/float32",
    [3UL],
    @lib.Datatype::write_float(4),
    @lib.Float64Values([1.5, -2.25, 3.0]),
  )
  let half_values : Array[Double] = [
    0.0,
    -0.0,
    1.0 / 67108864.0,
    1.0 / 33554432.0,
    1.0 / 33554432.0 + 1.0 / 1152921504606846976.0,
    3.0 / 33554432.0,
    1.0 / 16777216.0,
    1.0 + 1.0 / 2048.0,
    1.0 + 1.0 / 2048.0 + 1.0 / 1099511627776.0,
    1.0 + 1.0 / 1024.0,
    65504.0,
    65519.0,
    65520.0,
    -65504.0,
    -65519.0,
    -65520.0,
  ]
  writer.add_dataset_typed(
    "/generated/float16_boundaries",
    [half_values.length().to_uint64()],
    @lib.Datatype::write_float(2),
    @lib.Float64Values(half_values),
    options=@lib.Hdf5ConversionOptions::lossy(),
  )
  writer.add_attribute_typed(
    "/generated/float16_boundaries",
    "edge_values",
    [5UL],
    @lib.Datatype::write_float(2),
    @lib.Float64Values(Array::makei(5, index => half_values[index])),
    options=@lib.Hdf5ConversionOptions::lossy(),
  )
  let extended_values : Array[Double] = [
    1.5,
    -2.25,
    1.0 / 3.0,
    0.0,
    -0.0,
    conversion_double(0x7FF0000000000000UL),
    conversion_double(0xFFF0000000000000UL),
    conversion_double(0x7FF8000000000000UL),
  ]
  writer.add_dataset_typed(
    "/generated/x87_16_le",
    [extended_values.length().to_uint64()],
    @lib.Datatype::write_float(16),
    @lib.Float64Values(extended_values),
    options=@lib.Hdf5ConversionOptions::lossy(),
  )
  writer.add_dataset_typed(
    "/generated/x87_16_be",
    [extended_values.length().to_uint64()],
    @lib.Datatype::write_float_with_order(16, false),
    @lib.Float64Values(extended_values),
    options=@lib.Hdf5ConversionOptions::lossy(),
  )
  writer.add_dataset_typed(
    "/generated/integer_be",
    [4UL],
    @lib.Datatype::write_integer_with_order(true, 2, false),
    @lib.Int64Values([-32768L, -1L, 0L, 32767L]),
  )
  writer.add_dataset_typed(
    "/generated/integer_padded",
    [4UL],
    @lib.Datatype::write_integer(true, 4, precision=12, offset=4),
    @lib.Int64Values([-2048L, -1L, 0L, 2047L]),
  )
  writer.add_dataset_typed(
    "/generated/integer_widened",
    [3UL],
    @lib.Datatype::write_integer(true, 16),
    @lib.Int64Values([-1L, 0L, 1L]),
  )
  writer.add_dataset_typed(
    "/generated/integer_unsigned_be",
    [3UL],
    @lib.Datatype::write_integer_with_order(false, 4, false),
    @lib.Int64Values([0L, 1L, 2147483647L]),
  )
  let opaque4 = @lib.Datatype::write_opaque(4, "SRC")
  let opaque6 = @lib.Datatype::write_opaque(6, "DST")
  let opaque2 = @lib.Datatype::write_opaque(2, "DST")
  let opaque_values : Array[@lib.DatasetValue] = [
    @lib.Opaque(b"ABCD"),
    @lib.Opaque(b"WXYZ"),
  ]
  writer.add_dataset_typed(
    "/generated/opaque_widened",
    [2UL],
    opaque6,
    @lib.StructuredValues(opaque4, opaque_values),
    options=@lib.Hdf5ConversionOptions::lossy(),
  )
  writer.add_dataset_typed(
    "/generated/opaque_truncated",
    [2UL],
    opaque2,
    @lib.StructuredValues(opaque4, opaque_values),
    options=@lib.Hdf5ConversionOptions::lossy(),
  )
  writer.add_dataset_typed(
    "/generated/wide",
    [4UL],
    @lib.Datatype::write_integer(true, 16),
    @lib.BigIntegerValues(
      [
        @bigint.BigInt::from_int(123),
        @bigint.BigInt::from_int(-456),
        conversion_large_integer(80),
        conversion_large_integer(80).neg(),
      ],
      12,
      true,
    ),
  )
  writer.add_dataset_typed(
    "/generated/time",
    [2UL],
    @lib.Datatype::write_time(8),
    @lib.Int64Values([0L, 1700000000L]),
  )
  writer.add_dataset_typed(
    "/generated/bitfield",
    [3UL],
    @lib.Datatype::write_bitfield(2, 12),
    @lib.Int64Values([0L, 240L, 4095L]),
  )
  writer.add_dataset_typed(
    "/generated/fixed_names",
    [3UL],
    @lib.Datatype::write_fixed_string(8),
    @lib.VariableStringValues(["alpha", "月亮", ""]),
  )
  writer.add_dataset_typed(
    "/generated/vlen_names",
    [2UL],
    @lib.Datatype::write_vlen_string(),
    @lib.FixedStringValues(["moon", "数据"], 8),
  )
  writer.add_dataset_typed(
    "/generated/states",
    [3UL],
    conversion_state_type(),
    @lib.Int64Values([1L, 2L, 0L]),
  )
  let source_record = @lib.Datatype::write_compound([
    ("id", @lib.Datatype::write_integer(true, 4)),
    ("weight", @lib.Datatype::write_float(8)),
    ("extra", @lib.Datatype::write_fixed_string(4)),
  ])
  writer.add_dataset_typed(
    "/generated/records",
    [2UL],
    conversion_generated_record_type(),
    @lib.StructuredValues(source_record, [
      @lib.Compound([
        { name: "id", value: @lib.SignedInteger(7L), },
        { name: "weight", value: @lib.FloatingPoint(1.5), },
        { name: "extra", value: @lib.Text("x"), },
      ]),
      @lib.Compound([
        { name: "id", value: @lib.SignedInteger(-2L), },
        { name: "weight", value: @lib.FloatingPoint(-2.25), },
        { name: "extra", value: @lib.Text("yz"), },
      ]),
    ]),
    options=missing,
  )
  let source_array = @lib.Datatype::write_array(
    @lib.Datatype::write_integer(true, 2),
    [2UL, 2UL],
  )
  writer.add_dataset_typed(
    "/generated/arrays",
    [2UL],
    @lib.Datatype::write_array(@lib.Datatype::write_float(8), [2UL, 2UL]),
    @lib.StructuredValues(source_array, [
      @lib.Elements([
        @lib.SignedInteger(1L),
        @lib.SignedInteger(2L),
        @lib.SignedInteger(3L),
        @lib.SignedInteger(4L),
      ]),
      @lib.Elements([
        @lib.SignedInteger(-1L),
        @lib.SignedInteger(-2L),
        @lib.SignedInteger(-3L),
        @lib.SignedInteger(-4L),
      ]),
    ]),
  )
  let source_sequence = @lib.Datatype::write_vlen_sequence(
    @lib.Datatype::write_integer(true, 2),
  )
  writer.add_dataset_typed(
    "/generated/sequences",
    [3UL],
    @lib.Datatype::write_vlen_sequence(@lib.Datatype::write_float(8)),
    @lib.StructuredValues(source_sequence, [
      @lib.Sequence([
        @lib.SignedInteger(1L),
        @lib.SignedInteger(-2L),
        @lib.SignedInteger(3L),
      ]),
      @lib.Sequence([]),
      @lib.Sequence([@lib.SignedInteger(300L), @lib.SignedInteger(400L)]),
    ]),
  )
  writer.add_dataset_typed(
    "/generated/complex_compound",
    [1UL],
    @lib.Datatype::write_compound([
      ("r", @lib.Datatype::write_float(8)),
      ("i", @lib.Datatype::write_float(8)),
    ]),
    @lib.Complex128Values([(5.0, -6.0)]),
  )
  writer.add_dataset_typed(
    "/generated/complex_float32",
    [2UL],
    @lib.Datatype::write_compound([
      ("r", @lib.Datatype::write_float(4)),
      ("i", @lib.Datatype::write_float(4)),
    ]),
    @lib.Complex128Values([(1.5, -2.25), (1.0 / 3.0, -7.0 / 9.0)]),
    options=@lib.Hdf5ConversionOptions::lossy(),
  )
  writer.add_dataset_typed(
    "/generated/complex_array",
    [2UL],
    @lib.Datatype::write_array(@lib.Datatype::write_float(8), [2UL]),
    @lib.Complex128Values([(7.0, -8.0), (9.0, 10.0)]),
  )
  writer.add_chunked_dataset_typed(
    "/generated/chunked",
    [5UL],
    [2U],
    @lib.Datatype::write_integer(true, 4),
    @lib.Int16Values([10, 11, 12, 13, 14]),
    fill_value=@lib.Int16Values([-1]),
    filters=[@lib.Deflate(6)],
  )
  writer.add_extensible_dataset_typed(
    "/generated/growing",
    [2UL],
    [None],
    [2U],
    @lib.Datatype::write_float(8),
    @lib.Int32Values([1, 2]),
  )
}

///|
fn conversion_write_imported(
  writer : @lib.Hdf5Writer,
  fixture : @lib.Hdf5File,
  numeric : @lib.Hdf5File?,
  wide : @lib.Hdf5File?,
) -> Unit raise @lib.Hdf5Error {
  writer.add_group("/imported")
  let missing : @lib.Hdf5ConversionOptions = {
    mode: @lib.Hdf5ConversionMode::Strict,
    allow_missing_compound_members: true,
  }
  let record_type = conversion_imported_record_type()
  let records = fixture.read_dataset_as(
    "/records",
    record_type,
    options=missing,
  )
  writer.add_dataset_typed(
    "/imported/records",
    [3UL],
    record_type,
    @lib.StructuredValues(
      record_type,
      conversion_structured_values(records.values),
    ),
  )
  let prototype = fixture.read_attribute_as(
    "/records",
    "prototype",
    record_type,
    options=missing,
  )
  writer.add_attribute_typed(
    "/imported",
    "prototype",
    [],
    record_type,
    @lib.StructuredValues(
      record_type,
      conversion_structured_values(prototype.values.unwrap()),
    ),
  )
  let points = fixture.read_dataset_selection_as(
    "/states",
    @lib.Points([[2UL], [0UL]]),
    @lib.Datatype::write_integer(true, 8),
  )
  writer.add_dataset_typed(
    "/imported/states_points",
    points.dimensions,
    @lib.Datatype::write_integer(true, 8),
    @lib.Int64Values(conversion_integer_values(points.values)),
  )
  let fixed_names = fixture.read_dataset_as(
    "/names",
    @lib.Datatype::write_fixed_string(16),
  )
  writer.add_dataset_typed(
    "/imported/names_fixed",
    [3UL],
    @lib.Datatype::write_fixed_string(16),
    @lib.VariableStringValues(conversion_string_values(fixed_names.values)),
  )
  let vector_type = @lib.Datatype::write_array(@lib.Datatype::write_float(8), [
    2UL,
  ])
  let vectors = fixture.read_dataset_as("/vectors", vector_type)
  writer.add_dataset_typed(
    "/imported/vectors",
    [2UL],
    vector_type,
    @lib.StructuredValues(
      vector_type,
      conversion_structured_values(vectors.values),
    ),
  )
  let sequence_type = conversion_sequence_type()
  let sequences = fixture.read_dataset_as("/sequences", sequence_type)
  writer.add_dataset_typed(
    "/imported/sequences",
    [2UL],
    sequence_type,
    @lib.StructuredValues(
      sequence_type,
      conversion_structured_values(sequences.values),
    ),
  )
  match numeric {
    Some(numeric_fixture) => {
      let int64_type = @lib.Datatype::write_integer(true, 8)
      let uint64_type = @lib.Datatype::write_integer(false, 8)
      let float64_type = @lib.Datatype::write_float(8)
      for
        (source, target) in [
          ("/offset", "/imported/numeric_offset"),
          ("/precision", "/imported/numeric_precision"),
          ("/big_endian", "/imported/numeric_big_endian"),
          ("/signed_20", "/imported/numeric_signed20"),
        ] {
        conversion_copy_dataset_typed(
          writer, target, numeric_fixture, source, int64_type,
        )
      }
      conversion_copy_dataset_typed(
        writer, "/imported/numeric_unsigned40", numeric_fixture, "/unsigned_40_be",
        uint64_type,
      )
      for
        (source, target) in [
          ("/half_special", "/imported/numeric_half"),
          ("/custom_binary_float", "/imported/numeric_custom_float"),
        ] {
        conversion_copy_dataset_typed(
          writer, target, numeric_fixture, source, float64_type,
        )
      }
    }
    None => ()
  }
  match wide {
    Some(wide_fixture) => {
      let int128_type = @lib.Datatype::write_integer(true, 16)
      let uint128_type = @lib.Datatype::write_integer(false, 16)
      let float64_type = @lib.Datatype::write_float(8)
      for
        (source, target, datatype) in [
          ("/int128_le", "/imported/wide_int128_le", int128_type),
          ("/int128_be", "/imported/wide_int128_be", int128_type),
          ("/uint128_le", "/imported/wide_uint128_le", uint128_type),
          ("/int96_offset_le", "/imported/wide_int96_offset", int128_type),
        ] {
        conversion_copy_dataset_typed(
          writer, target, wide_fixture, source, datatype,
        )
      }
      for
        (source, target) in [
          ("/float80_le", "/imported/wide_float80_le"),
          ("/float80_be", "/imported/wide_float80_be"),
        ] {
        conversion_copy_dataset_typed(
          writer,
          target,
          wide_fixture,
          source,
          float64_type,
          options=@lib.Hdf5ConversionOptions::lossy(),
        )
      }
    }
    None => ()
  }
}

///|
fn run(args : Array[String]) -> Int32 {
  if args.length() < 3 || args.length() > 5 {
    eprintln(
      "usage: moon run src/cmd/conversion_demo -- <output.h5> <datatypes.h5> [numeric-layouts.h5] [wide-numeric.h5]",
    )
    return 2
  }
  try {
    let writer = @lib.Hdf5Writer::new()
    conversion_write_generated(writer)
    let fixture = @lib.Hdf5File::parse(@fs.read_file_to_bytes(args[2]))
    let numeric = if args.length() > 3 {
      Some(@lib.Hdf5File::parse(@fs.read_file_to_bytes(args[3])))
    } else {
      None
    }
    let wide = if args.length() > 4 {
      Some(@lib.Hdf5File::parse(@fs.read_file_to_bytes(args[4])))
    } else {
      None
    }
    conversion_write_imported(writer, fixture, numeric, wide)

    let file = @lib.Hdf5File::parse(writer.encode())
    let int64_type = @lib.Datatype::write_integer(true, 8)
    let chunk_values = conversion_collect_chunk_values(
      file,
      "/generated/chunked",
      [2UL],
      int64_type,
    )
    let selection_values = conversion_collect_selection_values(
      file,
      "/generated/chunked",
      @lib.Points([[4UL], [0UL], [2UL]]),
      2,
      int64_type,
    )
    let with_chunk_bridge = file.add_dataset_typed(
      "/generated",
      "chunk_bridge",
      [5UL],
      int64_type,
      @lib.Int64Values(chunk_values),
    )
    let with_selection_bridge = @lib.Hdf5File::parse(with_chunk_bridge).add_dataset_typed(
      "/generated",
      "selection_bridge",
      [3UL],
      int64_type,
      @lib.Int64Values(selection_values),
    )
    let patched = @lib.Hdf5File::parse(with_selection_bridge).patch_contiguous_dataset_typed(
      "/generated/narrowed",
      @lib.Int64Values([4L, 5L, 6L]),
    )
    let with_dataset = @lib.Hdf5File::parse(patched).add_dataset_typed(
      "/generated",
      "file_added",
      [2UL],
      @lib.Datatype::write_integer(true, 2),
      @lib.UInt8Values([7, 8]),
    )
    let with_attribute = @lib.Hdf5File::parse(with_dataset).add_attribute_typed(
      "/generated/file_added",
      "labels",
      [2UL],
      @lib.Datatype::write_fixed_string(4),
      @lib.VariableStringValues(["x", "yz"]),
    )
    let with_patched_attribute = @lib.Hdf5File::parse(with_attribute).patch_attribute_values_typed(
      "/generated/narrowed",
      "scale",
      @lib.Int32Values([4]),
    )
    let appended = @lib.Hdf5File::parse(with_patched_attribute).append_dataset_typed(
      "/generated/growing",
      [2UL],
      @lib.Int32Values([3, 4]),
    )
    @fs.write_bytes_to_file(args[1], appended)
    println("wrote " + args[1])
    0
  } catch {
    error => {
      eprintln("conversion interop error: " + error.to_string())
      1
    }
  }
}

///|
fn main {
  @sys.exit(run(@env.args()))
}
