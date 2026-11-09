///|
test "contiguous datasets can be patched without changing metadata" {
  let writer = Hdf5Writer::new()
  writer.add_group("/group")
  writer.add_dataset("/group/values", [4UL], Int64Values([1L, 2L, 3L, 4L]))
  writer.add_dataset("/group/other", [2UL], Int64Values([9L, 10L]))
  writer.add_attribute(
    "/group/values",
    "units",
    [],
    FixedStringValues(["count"], 8),
  )
  let original = writer.encode()
  let file = Hdf5File::parse(original)
  let patched = file.patch_contiguous_dataset(
    "/group/values",
    Int64Values([100L, -200L, 300L, -400L]),
  )
  let patched_file = Hdf5File::parse(patched)
  assert_eq(integer_values(patched_file.read_dataset("/group/values")), [
    100L, -200L, 300L, -400L,
  ])
  assert_eq(integer_values(patched_file.read_dataset("/group/other")), [9L, 10L])
  let attributes = patched_file.attributes("/group/values")
  assert_eq(attributes.length(), 1)
  match attributes[0].values.unwrap() {
    Strings(values) => assert_eq(values, ["count"])
    _ => fail("expected the original string attribute")
  }
}

///|
test "contiguous patches support fixed strings and structured values" {
  let writer = Hdf5Writer::new()
  writer.add_dataset("/labels", [2UL], FixedStringValues(["left", "right"], 8))
  let record = Datatype::write_compound([
    ("id", Datatype::write_integer(true, 4)),
    ("value", Datatype::write_float(8)),
  ])
  writer.add_dataset(
    "/records",
    [1UL],
    StructuredValues(record, [
      Compound([
        { name: "id", value: SignedInteger(7L), },
        { name: "value", value: FloatingPoint(2.5), },
      ]),
    ]),
  )
  let file = Hdf5File::parse(writer.encode())
  let patched = file.patch_contiguous_dataset(
    "/labels",
    FixedStringValues(["alpha", "beta"], 8),
  )
  let patched_file = Hdf5File::parse(patched)
  assert_eq(string_values(patched_file.read_dataset("/labels")), [
    "alpha", "beta",
  ])
  let patched_records = patched_file.patch_contiguous_dataset(
    "/records",
    StructuredValues(record, [
      Compound([
        { name: "id", value: SignedInteger(9L), },
        { name: "value", value: FloatingPoint(-1.25), },
      ]),
    ]),
  )
  match Hdf5File::parse(patched_records).read_dataset("/records").values {
    Structured([Compound([id, value])]) => {
      assert_eq(id.value, SignedInteger(9L))
      assert_eq(value.value, FloatingPoint(-1.25))
    }
    _ => fail("expected patched compound value")
  }
}

///|
test "contiguous patches support half-precision floats" {
  let writer = Hdf5Writer::new()
  writer.add_dataset(
    "/half",
    [4UL],
    Float16Values([Float(1.0), Float(2.0), Float(3.0), Float(4.0)]),
  )
  let file = Hdf5File::parse(writer.encode())
  let patched = file.patch_contiguous_dataset(
    "/half",
    Float16Values([Float(-1.5), Float(0.25), Float(0.5), Float(65504.0)]),
  )
  let patched_file = Hdf5File::parse(patched)
  let metadata = patched_file.read_dataset_metadata("/half")
  assert_eq(metadata.datatype.size, 2)
  assert_eq(float_values(patched_file.read_dataset("/half")), [
    -1.5, 0.25, 0.5, 65504.0,
  ])
}

///|
test "patching rejects noncontiguous unallocated and incompatible storage" {
  let writer = Hdf5Writer::new()
  writer.add_chunked_dataset("/chunked", [2UL], [1U], Int64Values([1L, 2L]))
  writer.add_dataset("/fixed", [2UL], Int64Values([1L, 2L]))
  let file = Hdf5File::parse(writer.encode())
  let chunked = try {
    file.patch_contiguous_dataset("/chunked", Int64Values([3L, 4L])) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    chunked,
    Some(UnsupportedFeature("only contiguous storage can be patched")),
  )
  let wrong_type = try {
    file.patch_contiguous_dataset("/fixed", UInt64Values([1UL, 2UL])) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    wrong_type,
    Some(InvalidArgument("patched values must match the dataset datatype")),
  )
  let wrong_shape = try {
    file.patch_contiguous_dataset("/fixed", Int64Values([1L])) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    wrong_shape,
    Some(InvalidArgument("writer value count does not match dataspace")),
  )
  let compact = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/interop_earliest.h5"),
  )
  let compact_error = try {
    compact.patch_contiguous_dataset("/data/compact", Int64Values([1L]))
    |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    compact_error,
    Some(UnsupportedFeature("only contiguous storage can be patched")),
  )
  let fill = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/contiguous_fill_v3.h5"),
  )
  let fill_error = try {
    fill.patch_contiguous_dataset("/signed", Int32Values([1, 2, 3])) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    fill_error,
    Some(UnsupportedFeature("unallocated contiguous storage cannot be patched")),
  )
}

///|
test "patching supports big-endian numeric datasets and rejects external sources" {
  let writer = Hdf5Writer::new()
  writer.add_external_dataset("/external", [2UL], Int64Values([1L, 2L]), [
    { file_name: "patch-external.bin", file_offset: 0UL, data_size: 16UL, },
  ])
  let external_file = Hdf5File::parse(writer.encode())
  let external = try {
    external_file.patch_contiguous_dataset("/external", Int64Values([3L, 4L]))
    |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    external,
    Some(UnsupportedFeature("external storage cannot be patched")),
  )
  let big_endian = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/numeric_layouts.h5"),
  )
  let patched_be = big_endian.patch_contiguous_dataset(
    "/be_i4",
    Int32Values([1, 2]),
  )
  let patched_be_file = Hdf5File::parse(patched_be)
  assert_eq(integer_values(patched_be_file.read_dataset("/be_i4")), [1L, 2L])
  let patched_half = patched_be_file.patch_contiguous_dataset(
    "/be_f2",
    Float16Values([Float(1.5), Float(-2.25)]),
  )
  let patched_half_file = Hdf5File::parse(patched_half)
  assert_eq(float_values(patched_half_file.read_dataset("/be_f2")), [1.5, -2.25])
  let encoded = writer.encode()
  let sources : Array[&Hdf5Source] = [encoded]
  let sourced = Hdf5File::parse_source(sources[0])
  let source_error = try {
    sourced.patch_contiguous_dataset("/external", Int64Values([1L, 2L]))
    |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    source_error,
    Some(InvalidArgument("in-place patching requires a bytes-backed Hdf5File")),
  )
}

///|
test "patching rejects nested variable-length and reference datatypes" {
  let writer = Hdf5Writer::new()
  let reference_record = Datatype::write_compound([
    ("target", Datatype::write_object_reference()),
    ("id", Datatype::write_integer(true, 4)),
  ])
  writer.add_dataset(
    "/reference_record",
    [1UL],
    StructuredValues(reference_record, [
      Compound([
        { name: "target", value: ObjectReference(None), },
        { name: "id", value: SignedInteger(1L), },
      ]),
    ]),
  )
  let sequence_record = Datatype::write_compound([
    ("items", Datatype::write_vlen_sequence(Datatype::write_integer(true, 4))),
  ])
  writer.add_dataset(
    "/sequence_record",
    [1UL],
    StructuredValues(sequence_record, [
      Compound([{ name: "items", value: Sequence([SignedInteger(1L)]), }]),
    ]),
  )
  let file = Hdf5File::parse(writer.encode())
  let reference_error = try {
    file.patch_contiguous_dataset(
      "/reference_record",
      StructuredValues(reference_record, [
        Compound([
          { name: "target", value: ObjectReference(None), },
          { name: "id", value: SignedInteger(2L), },
        ]),
      ]),
    )
    |> ignore
    None
  } catch {
    error => Some(error)
  }
  let sequence_error = try {
    file.patch_contiguous_dataset(
      "/sequence_record",
      StructuredValues(sequence_record, [
        Compound([{ name: "items", value: Sequence([SignedInteger(2L)]), }]),
      ]),
    )
    |> ignore
    None
  } catch {
    error => Some(error)
  }
  let expected = Some(
    UnsupportedFeature(
      "variable-length and reference datatypes cannot be patched",
    ),
  )
  assert_eq(reference_error, expected)
  assert_eq(sequence_error, expected)
}
