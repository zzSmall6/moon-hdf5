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
test "compact datasets can be patched in place" {
  let earliest = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/interop_earliest.h5"),
  )
  let earliest_patched = earliest.patch_dataset_values(
    "/data/compact",
    Int32Values([31, 41]),
  )
  let earliest_file = Hdf5File::parse(earliest_patched)
  assert_eq(
    earliest_file.read_dataset_metadata("/data/compact").layout.kind(),
    Compact,
  )
  assert_eq(integer_values(earliest_file.read_dataset("/data/compact")), [
    31L, 41L,
  ])

  let v2 = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/attribute_storage.h5"),
  )
  let v2_patched = v2.patch_dataset_values("/compact", Int64Values([99L]))
  let v2_file = Hdf5File::parse(v2_patched)
  assert_eq(integer_values(v2_file.read_dataset("/compact")), [99L])
  assert_eq(v2_file.attributes("/compact").length(), 2)
  assert_eq(v2_file.attributes("/dense").length(), 12)
  assert_eq(integer_values(v2_file.read_dataset("/dense")), [42L])

  let layout_v4 = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/layout_v4.h5"),
  )
  let layout_patched = layout_v4.patch_dataset_values(
    "/compact",
    Int32Values([31, 41]),
  )
  let layout_file = Hdf5File::parse(layout_patched)
  assert_eq(integer_values(layout_file.read_dataset("/compact")), [31L, 41L])
  assert_eq(integer_values(layout_file.read_dataset("/contiguous")), [10L, 20L])
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

///|
fn patch_int64_sequence(
  count : Int32,
  first : Int64,
  step : Int64,
) -> Array[Int64] {
  let values = Array::make(count, 0L)
  for index in 0..<count {
    values[index] = first + index.to_int64() * step
  }
  values
}

///|
fn patch_int32_sequence(
  count : Int32,
  first : Int32,
  step : Int32,
) -> Array[Int32] {
  let values = Array::make(count, 0)
  for index in 0..<count {
    values[index] = first + index.to_int64().to_int() * step
  }
  values
}

///|
test "chunked patches cover v1 and v2 B-tree indexes" {
  let edge = Hdf5File::parse(@fs.read_file_to_bytes("testdata/chunked_edge.h5"))
  let expected = patch_int32_sequence(35, 100, 1)
  let edge_patched = edge.patch_dataset_values("/edge", Int32Values(expected))
  let edge_file = Hdf5File::parse(edge_patched)
  assert_eq(edge_file.read_dataset_metadata("/edge").layout.kind(), Chunked)
  assert_eq(
    integer_values(edge_file.read_dataset("/edge")),
    expected.map(value => value.to_int64()),
  )

  let btree = Hdf5File::parse(@fs.read_file_to_bytes("testdata/btree_v2.h5"))
  let deep_expected = patch_int32_sequence(10000, -5, 1)
  let deep_patched = btree.patch_dataset_values(
    "/raw/deep",
    Int32Values(deep_expected),
  )
  let deep_file = Hdf5File::parse(deep_patched)
  let deep_values = integer_values(deep_file.read_dataset("/raw/deep"))
  assert_eq(deep_values.length(), 10000)
  assert_eq(deep_values[0], -5L)
  assert_eq(deep_values[9999], 9994L)

  let sparse_error = try {
    btree.patch_dataset_values("/raw/sparse", Int32Values(Array::make(63, 1)))
    |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    sparse_error,
    Some(
      UnsupportedFeature(
        "chunked patching cannot allocate missing chunks for non-fill values",
      ),
    ),
  )
}

///|
test "chunked patches cover fixed and extensible array indexes" {
  let fixed = Hdf5File::parse(@fs.read_file_to_bytes("testdata/fixed_array.h5"))
  let raw_expected = patch_int32_sequence(35, 7, 2)
  let raw_file = Hdf5File::parse(
    fixed.patch_dataset_values("/raw", Int32Values(raw_expected)),
  )
  assert_eq(
    integer_values(raw_file.read_dataset("/raw")),
    raw_expected.map(value => value.to_int64()),
  )
  let resizable_expected = patch_int32_sequence(12, -3, 1)
  let resizable_file = Hdf5File::parse(
    fixed.patch_dataset_values("/resizable", Int32Values(resizable_expected)),
  )
  assert_eq(
    integer_values(resizable_file.read_dataset("/resizable")),
    resizable_expected.map(value => value.to_int64()),
  )
  let paged_error = try {
    fixed.patch_dataset_values("/paged", Int32Values(Array::make(4099, 1)))
    |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    paged_error,
    Some(
      UnsupportedFeature(
        "chunked patching cannot allocate missing chunks for non-fill values",
      ),
    ),
  )

  let extensible = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/extensible_array.h5"),
  )
  let grown_expected = patch_int32_sequence(420, 11, 3)
  let grown_file = Hdf5File::parse(
    extensible.patch_dataset_values("/raw/grown", Int32Values(grown_expected)),
  )
  assert_eq(
    integer_values(grown_file.read_dataset("/raw/grown")),
    grown_expected.map(value => value.to_int64()),
  )
  let swizzled_expected = patch_int32_sequence(210, 1, -1)
  let swizzled_file = Hdf5File::parse(
    extensible.patch_dataset_values(
      "/raw/swizzled",
      Int32Values(swizzled_expected),
    ),
  )
  assert_eq(
    integer_values(swizzled_file.read_dataset("/raw/swizzled")),
    swizzled_expected.map(value => value.to_int64()),
  )
}

///|
test "chunked patches cover v4 single and implicit indexes" {
  let file = Hdf5File::parse(@fs.read_file_to_bytes("testdata/layout_v4.h5"))
  let single_expected = patch_int32_sequence(12, 50, 1)
  let single_file = Hdf5File::parse(
    file.patch_dataset_values("/single/raw", Int32Values(single_expected)),
  )
  assert_eq(
    integer_values(single_file.read_dataset("/single/raw")),
    single_expected.map(value => value.to_int64()),
  )
  let implicit_expected = patch_int32_sequence(35, -1, -2)
  let implicit_file = Hdf5File::parse(
    file.patch_dataset_values("/implicit/edge", Int32Values(implicit_expected)),
  )
  assert_eq(
    integer_values(implicit_file.read_dataset("/implicit/edge")),
    implicit_expected.map(value => value.to_int64()),
  )
  let resizable_expected = patch_int32_sequence(12, 70, 1)
  let resizable_file = Hdf5File::parse(
    file.patch_dataset_values(
      "/implicit/resizable",
      Int32Values(resizable_expected),
    ),
  )
  assert_eq(
    integer_values(resizable_file.read_dataset("/implicit/resizable")),
    resizable_expected.map(value => value.to_int64()),
  )
  let filtered_error = try {
    file.patch_dataset_values(
      "/single/filtered",
      Int32Values(Array::make(12, 0)),
    )
    |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    filtered_error,
    Some(UnsupportedFeature("filtered storage cannot be patched")),
  )
}

///|
test "chunked patches map multidimensional chunks and edge padding" {
  let writer = Hdf5Writer::new()
  writer.add_chunked_dataset(
    "/matrix",
    [5UL, 7UL],
    [2U, 3U],
    Int64Values(patch_int64_sequence(35, 1L, 1L)),
  )
  let file = Hdf5File::parse(writer.encode())
  let expected = patch_int64_sequence(35, 1L, 3L)
  let patched = file.patch_dataset_values("/matrix", Int64Values(expected))
  let patched_file = Hdf5File::parse(patched)
  assert_eq(
    patched_file.read_dataset_metadata("/matrix").layout.kind(),
    Chunked,
  )
  assert_eq(integer_values(patched_file.read_dataset("/matrix")), expected)
}

///|
test "chunked patches preserve missing fill-only chunks" {
  let fixed = Hdf5File::parse(@fs.read_file_to_bytes("testdata/fixed_array.h5"))
  let paged_expected = Array::make(4099, -7)
  for index in [0, 1023, 2048, 4098] {
    paged_expected[index] = 1000 + index
  }
  let paged_file = Hdf5File::parse(
    fixed.patch_dataset_values("/paged", Int32Values(paged_expected)),
  )
  let paged_values = integer_values(paged_file.read_dataset("/paged"))
  assert_eq(paged_values.length(), 4099)
  assert_eq(paged_values[0], 1000L)
  assert_eq(paged_values[1], -7L)
  assert_eq(paged_values[1023], 2023L)
  assert_eq(paged_values[2048], 3048L)
  assert_eq(paged_values[4098], 5098L)

  let btree = Hdf5File::parse(@fs.read_file_to_bytes("testdata/btree_v2.h5"))
  let unallocated_expected = Array::make(8, -5)
  let unallocated_file = Hdf5File::parse(
    btree.patch_dataset_values(
      "/other/unallocated",
      Int32Values(unallocated_expected),
    ),
  )
  assert_eq(
    integer_values(unallocated_file.read_dataset("/other/unallocated")),
    Array::make(8, -5L),
  )
  let sparse_expected = Array::make(63, -7)
  sparse_expected[11] = 111
  sparse_expected[62] = 222
  let sparse_file = Hdf5File::parse(
    btree.patch_dataset_values("/raw/sparse", Int32Values(sparse_expected)),
  )
  let sparse_values = integer_values(sparse_file.read_dataset("/raw/sparse"))
  assert_eq(sparse_values.length(), 63)
  assert_eq(sparse_values[0], -7L)
  assert_eq(sparse_values[11], 111L)
  assert_eq(sparse_values[62], 222L)
}
