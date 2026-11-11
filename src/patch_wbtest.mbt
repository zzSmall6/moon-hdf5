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
  assert_eq(
    reference_error,
    Some(UnsupportedFeature("reference datatypes cannot be patched")),
  )
  assert_eq(
    sequence_error,
    Some(
      UnsupportedFeature("only variable-length string datatypes can be patched"),
    ),
  )
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
fn patch_pseudorandom_int32(count : Int32, seed : Int64) -> Array[Int32] {
  let values = Array::make(count, 0)
  let mut state = seed
  for index in 0..<count {
    state = (state * 1103515245L + 12345L) & 0x7FFFFFFFL
    values[index] = (state >> 8).to_int()
  }
  values
}

///|
fn patch_fixed_array_without_data_block(
  bytes : Bytes,
  path : String,
) -> Bytes raise Hdf5Error {
  let file = Hdf5File::parse(bytes)
  let metadata = file.read_dataset_metadata(path)
  guard metadata.layout.chunk_index is FixedArray(_) else {
    raise InvalidField("expected a fixed array dataset")
  }
  let base_address = file.superblock.base_address
  let offset_size = file.superblock.offset_size
  let header = address_to_offset(metadata.layout.data_address, base_address)
  let block_address_offset = header + 8 + file.superblock.length_size
  let undefined = if offset_size == 8 {
    0xFFFFFFFFFFFFFFFFUL
  } else {
    (1UL << (offset_size * 8)) - 1UL
  }
  let output = bytes.to_fixedarray()
  patch_little_endian_uint(
    output, block_address_offset, offset_size, undefined, "test fixed array data block address",
  )
  patch_metadata_checksum_in_place(
    output,
    header,
    block_address_offset + offset_size,
  )
  Bytes::from_array(output)
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

  let sparse_expected = Array::make(63, 1)
  let sparse_bytes = btree.patch_dataset_values(
    "/raw/sparse",
    Int32Values(sparse_expected),
  )
  let sparse_file = Hdf5File::parse(sparse_bytes)
  assert_true(
    sparse_file.read_dataset_metadata("/raw/sparse").layout.chunk_index
    is BTreeV2(_),
  )
  assert_eq(
    integer_values(sparse_file.read_dataset("/raw/sparse")),
    Array::make(63, 1L),
  )
  let rewritten_expected = patch_int32_sequence(63, 500, 3)
  let rewritten = Hdf5File::parse(
    sparse_file.patch_dataset_values(
      "/raw/sparse",
      Int32Values(rewritten_expected),
    ),
  )
  assert_eq(
    integer_values(rewritten.read_dataset("/raw/sparse")),
    rewritten_expected.map(value => value.to_int64()),
  )
}

///|
test "filtered growth relocation covers v1 v2 single and extensible indexes" {
  let v1_bytes = @fs.read_file_to_bytes("testdata/filters_v1.h5")
  let v1 = Hdf5File::parse(v1_bytes)
  let v1_expected = patch_pseudorandom_int32(35, 17L)
  let v1_patched_bytes = v1.patch_dataset_values(
    "/gzip",
    Int32Values(v1_expected),
  )
  assert_true(v1_patched_bytes.length() > v1_bytes.length())
  let v1_patched = Hdf5File::parse(v1_patched_bytes)
  assert_true(
    v1_patched.read_dataset_metadata("/gzip").layout.chunk_index is BTreeV1,
  )
  assert_eq(
    integer_values(v1_patched.read_dataset("/gzip")),
    v1_expected.map(value => value.to_int64()),
  )

  let writer = Hdf5Writer::new()
  writer.add_extensible_dataset(
    "/extensible",
    [64UL],
    [None],
    [16U],
    Int32Values(Array::make(64, 1)),
    filters=[Deflate(9)],
  )
  writer.add_extensible_dataset(
    "/btree",
    [4UL, 4UL],
    [None, None],
    [2U, 2U],
    Int32Values(Array::make(16, 1)),
    filters=[Deflate(9)],
  )
  let indexed_bytes = writer.encode()
  let indexed = Hdf5File::parse(indexed_bytes)
  let extensible_expected = patch_pseudorandom_int32(64, 29L)
  let extensible_bytes = indexed.patch_dataset_values(
    "/extensible",
    Int32Values(extensible_expected),
  )
  assert_true(extensible_bytes.length() > indexed_bytes.length())
  let extensible = Hdf5File::parse(extensible_bytes)
  assert_true(
    extensible.read_dataset_metadata("/extensible").layout.chunk_index
    is ExtensibleArray(_),
  )
  assert_eq(
    integer_values(extensible.read_dataset("/extensible")),
    extensible_expected.map(value => value.to_int64()),
  )
  let btree_expected = patch_pseudorandom_int32(16, 41L)
  let btree_bytes = extensible.patch_dataset_values(
    "/btree",
    Int32Values(btree_expected),
  )
  assert_true(btree_bytes.length() > extensible_bytes.length())
  let btree = Hdf5File::parse(btree_bytes)
  assert_true(
    btree.read_dataset_metadata("/btree").layout.chunk_index is BTreeV2(_),
  )
  assert_eq(
    integer_values(btree.read_dataset("/btree")),
    btree_expected.map(value => value.to_int64()),
  )

  let single_bytes = @fs.read_file_to_bytes("testdata/layout_v4.h5")
  let single = Hdf5File::parse(single_bytes)
  let single_expected = patch_pseudorandom_int32(12, 53L)
  let single_patched_bytes = single.patch_dataset_values(
    "/single/filtered",
    Int32Values(single_expected),
  )
  assert_true(single_patched_bytes.length() > single_bytes.length())
  let single_patched = Hdf5File::parse(single_patched_bytes)
  assert_true(
    single_patched.read_dataset_metadata("/single/filtered").layout.chunk_index
    is Single(_, _),
  )
  assert_eq(
    integer_values(single_patched.read_dataset("/single/filtered")),
    single_expected.map(value => value.to_int64()),
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
  let filtered_expected = patch_int32_sequence(12, 90, 2)
  let filtered_file = Hdf5File::parse(
    file.patch_dataset_values(
      "/single/filtered",
      Int32Values(filtered_expected),
    ),
  )
  assert_eq(
    integer_values(filtered_file.read_dataset("/single/filtered")),
    filtered_expected.map(value => value.to_int64()),
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

///|
test "filtered patches cover v1 and v2 B-tree indexes" {
  let compressed = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/compressed.h5"),
  )
  let gzip_expected = Array::make(256, 7)
  let gzip_file = Hdf5File::parse(
    compressed.patch_dataset_values("/gzip", Int32Values(gzip_expected)),
  )
  assert_eq(
    integer_values(gzip_file.read_dataset("/gzip")),
    Array::make(256, 7L),
  )

  let btree = Hdf5File::parse(@fs.read_file_to_bytes("testdata/btree_v2.h5"))
  let edge_expected = patch_int32_sequence(35, 101, -2)
  let edge_file = Hdf5File::parse(
    btree.patch_dataset_values("/filtered/edge", Int32Values(edge_expected)),
  )
  assert_eq(
    integer_values(edge_file.read_dataset("/filtered/edge")),
    edge_expected.map(value => value.to_int64()),
  )
  let deep_expected = patch_int32_sequence(10000, -1000, 1)
  let deep_file = Hdf5File::parse(
    btree.patch_dataset_values("/filtered/deep", Int32Values(deep_expected)),
  )
  assert_eq(
    integer_values(deep_file.read_dataset("/filtered/deep")),
    deep_expected.map(value => value.to_int64()),
  )
  let sparse_expected = Array::make(63, -7)
  sparse_expected[11] = 41
  sparse_expected[62] = 43
  let sparse_file = Hdf5File::parse(
    btree.patch_dataset_values("/filtered/sparse", Int32Values(sparse_expected)),
  )
  let sparse_values = integer_values(
    sparse_file.read_dataset("/filtered/sparse"),
  )
  assert_eq(sparse_values.length(), 63)
  assert_eq(sparse_values[0], -7L)
  assert_eq(sparse_values[11], 41L)
  assert_eq(sparse_values[62], 43L)
}

///|
test "filtered v2 sparse patches create missing B-tree records" {
  let bytes = @fs.read_file_to_bytes("testdata/btree_v2.h5")
  let file = Hdf5File::parse(bytes)
  let expected = patch_int32_sequence(63, 2000, 3)
  let patched_bytes = file.patch_dataset_values(
    "/filtered/sparse",
    Int32Values(expected),
  )
  assert_true(patched_bytes.length() > bytes.length())
  let patched = Hdf5File::parse(patched_bytes)
  assert_true(
    patched.read_dataset_metadata("/filtered/sparse").layout.chunk_index
    is BTreeV2(_),
  )
  assert_eq(
    integer_values(patched.read_dataset("/filtered/sparse")),
    expected.map(value => value.to_int64()),
  )
  let rewritten_expected = patch_int32_sequence(63, -300, 2)
  let rewritten = Hdf5File::parse(
    patched.patch_dataset_values(
      "/filtered/sparse",
      Int32Values(rewritten_expected),
    ),
  )
  assert_eq(
    integer_values(rewritten.read_dataset("/filtered/sparse")),
    rewritten_expected.map(value => value.to_int64()),
  )
}

///|
test "v1 sparse chunk patches create and rewrite B-tree records" {
  let bytes = @fs.read_file_to_bytes("testdata/interop_earliest.h5")
  let file = Hdf5File::parse(bytes)
  let metadata = file.read_dataset_metadata("/data/chunked_sparse")
  assert_true(metadata.layout.chunk_index is BTreeV1)
  let expected = patch_int32_sequence(63, 100, 1)
  let patched_bytes = file.patch_dataset_values(
    "/data/chunked_sparse",
    Int32Values(expected),
  )
  assert_true(patched_bytes.length() > bytes.length())
  let patched = Hdf5File::parse(patched_bytes)
  assert_true(
    patched.read_dataset_metadata("/data/chunked_sparse").layout.chunk_index
    is BTreeV1,
  )
  assert_eq(
    integer_values(patched.read_dataset("/data/chunked_sparse")),
    expected.map(value => value.to_int64()),
  )
  let rewritten_expected = patch_int32_sequence(63, -50, 2)
  let rewritten = Hdf5File::parse(
    patched.patch_dataset_values(
      "/data/chunked_sparse",
      Int32Values(rewritten_expected),
    ),
  )
  assert_eq(
    integer_values(rewritten.read_dataset("/data/chunked_sparse")),
    rewritten_expected.map(value => value.to_int64()),
  )
}

///|
test "filtered v1 sparse patches create missing B-tree records" {
  let bytes = @fs.read_file_to_bytes("testdata/filters_v1.h5")
  let file = Hdf5File::parse(bytes)
  assert_true(
    file.read_dataset_metadata("/special/partial").layout.chunk_index is BTreeV1,
  )
  let expected : Array[Int32] = [5, 6, 7, 8, 9, 10]
  let patched_bytes = file.patch_dataset_values(
    "/special/partial",
    Int32Values(expected),
  )
  assert_true(patched_bytes.length() > bytes.length())
  let patched = Hdf5File::parse(patched_bytes)
  assert_eq(
    integer_values(patched.read_dataset("/special/partial")),
    expected.map(value => value.to_int64()),
  )
}

///|
test "v2 sparse patches rebuild empty and internal B-tree levels" {
  let writer = Hdf5Writer::new()
  writer.add_extensible_dataset(
    "/sparse",
    [16UL, 16UL],
    [None, None],
    [1U, 1U],
    Int32Values(Array::make(256, -7)),
    fill_value=Int32Values([-7]),
  )
  let original = writer.encode()
  let file = Hdf5File::parse(original)
  let before = file.read_dataset("/sparse")
  assert_true(before.layout.chunk_index is BTreeV2(_))
  assert_eq(integer_values(before), Array::make(256, -7L))
  let expected = patch_int32_sequence(256, -20, 1)
  let patched_bytes = file.patch_dataset_values(
    "/sparse",
    Int32Values(expected),
  )
  assert_true(patched_bytes.length() > original.length())
  let patched = Hdf5File::parse(patched_bytes)
  let dataset = patched.read_dataset("/sparse")
  assert_true(dataset.layout.chunk_index is BTreeV2(_))
  let header = address_to_offset(
    dataset.layout.data_address,
    patched.superblock.base_address,
  )
  assert_true(
    Reader::at(patched_bytes, header + 12).read_u16_le().to_int() >= 1,
  )
  assert_eq(integer_values(dataset), expected.map(value => value.to_int64()))
}

///|
test "filtered patches cover fixed and extensible arrays including pages" {
  let fixed = Hdf5File::parse(@fs.read_file_to_bytes("testdata/fixed_array.h5"))
  let fixed_expected = patch_int32_sequence(35, 100, 1)
  let fixed_file = Hdf5File::parse(
    fixed.patch_dataset_values("/filtered", Int32Values(fixed_expected)),
  )
  assert_eq(
    integer_values(fixed_file.read_dataset("/filtered")),
    fixed_expected.map(value => value.to_int64()),
  )
  let paged_expected = Array::make(4099, -7)
  for index in [0, 1023, 2048, 4098] {
    paged_expected[index] = 2000 + index
  }
  let paged_file = Hdf5File::parse(
    fixed.patch_dataset_values("/paged_filtered", Int32Values(paged_expected)),
  )
  let paged_values = integer_values(paged_file.read_dataset("/paged_filtered"))
  assert_eq(paged_values.length(), 4099)
  for index in [0, 1023, 2048, 4098] {
    assert_eq(paged_values[index], (2000 + index).to_int64())
  }
  assert_eq(paged_values[1], -7L)

  let extensible = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/extensible_array.h5"),
  )
  let grown_expected = Array::make(420, 11)
  let grown_file = Hdf5File::parse(
    extensible.patch_dataset_values(
      "/filtered/grown",
      Int32Values(grown_expected),
    ),
  )
  assert_eq(
    integer_values(grown_file.read_dataset("/filtered/grown")),
    Array::make(420, 11L),
  )
  let swizzled_expected = Array::make(210, 13)
  let swizzled_file = Hdf5File::parse(
    extensible.patch_dataset_values(
      "/filtered/swizzled",
      Int32Values(swizzled_expected),
    ),
  )
  assert_eq(
    integer_values(swizzled_file.read_dataset("/filtered/swizzled")),
    Array::make(210, 13L),
  )
  let positions = [
    0, 3, 4, 19, 20, 51, 52, 115, 116, 243, 244, 499, 500, 1011, 1012, 65523, 65524,
    131060, 133108, 199999,
  ]
  let extensible_paged_expected = Array::make(200000, -7)
  for index in positions {
    extensible_paged_expected[index] = 300000 + index
  }
  let extensible_paged_file = Hdf5File::parse(
    extensible.patch_dataset_values(
      "/filtered/paged",
      Int32Values(extensible_paged_expected),
    ),
  )
  let extensible_paged_values = integer_values(
    extensible_paged_file.read_dataset("/filtered/paged"),
  )
  assert_eq(extensible_paged_values.length(), 200000)
  for index in positions {
    assert_eq(extensible_paged_values[index], (300000 + index).to_int64())
  }
  assert_eq(extensible_paged_values[1], -7L)
}

///|
test "filtered patches refresh fixed extensible and v2 B-tree metadata" {
  let writer = Hdf5Writer::new()
  writer.add_chunked_dataset(
    "/fixed",
    [8UL],
    [2U],
    Int32Values([0, 1, 2, 3, 4, 5, 6, 7]),
    filters=[ScaleOffset(0)],
  )
  writer.add_extensible_dataset(
    "/extensible",
    [8UL],
    [None],
    [2U],
    Int32Values([0, 1, 2, 3, 4, 5, 6, 7]),
    filters=[ScaleOffset(0)],
  )
  writer.add_extensible_dataset(
    "/btree",
    [4UL, 4UL],
    [None, None],
    [2U, 2U],
    Int32Values([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
    filters=[ScaleOffset(0)],
  )
  let file = Hdf5File::parse(writer.encode())
  assert_true(file.read_dataset("/fixed").layout.chunk_index is FixedArray(_))
  assert_true(
    file.read_dataset("/extensible").layout.chunk_index is ExtensibleArray(_),
  )
  assert_true(file.read_dataset("/btree").layout.chunk_index is BTreeV2(_))
  let patched = Hdf5File::parse(
    file.patch_dataset_values("/fixed", Int32Values(Array::make(8, 500))),
  )
  assert_eq(
    integer_values(patched.read_dataset("/fixed")),
    Array::make(8, 500L),
  )
  let patched = Hdf5File::parse(
    patched.patch_dataset_values(
      "/extensible",
      Int32Values(Array::make(8, 600)),
    ),
  )
  assert_eq(
    integer_values(patched.read_dataset("/extensible")),
    Array::make(8, 600L),
  )
  let patched = Hdf5File::parse(
    patched.patch_dataset_values("/btree", Int32Values(Array::make(16, 700))),
  )
  assert_eq(
    integer_values(patched.read_dataset("/btree")),
    Array::make(16, 700L),
  )
}

///|
test "filtered patches support multi-filter pipelines and relocate growth" {
  let writer = Hdf5Writer::new()
  writer.add_chunked_dataset(
    "/multi",
    [64UL],
    [16U],
    Int32Values(patch_int32_sequence(64, 0, 1)),
    filters=[Shuffle, Deflate(6), Fletcher32],
  )
  let file = Hdf5File::parse(writer.encode())
  let expected = Array::make(64, 17)
  let patched = Hdf5File::parse(
    file.patch_dataset_values("/multi", Int32Values(expected)),
  )
  assert_eq(
    integer_values(patched.read_dataset("/multi")),
    Array::make(64, 17L),
  )

  let growth_writer = Hdf5Writer::new()
  growth_writer.add_chunked_dataset(
    "/zeros",
    [64UL],
    [64U],
    Int32Values(Array::make(64, 1)),
    filters=[Deflate(9)],
  )
  let growth_bytes = growth_writer.encode()
  let growth_file = Hdf5File::parse(growth_bytes)
  let random = Array::make(64, 0)
  let mut state = 1L
  for index in 0..<64 {
    state = (state * 1103515245L + 12345L) & 0x7FFFFFFFL
    random[index] = (state >> 8).to_int()
  }
  let patched_bytes = growth_file.patch_dataset_values(
    "/zeros",
    Int32Values(random),
  )
  assert_true(patched_bytes.length() > growth_bytes.length())
  let patched_file = Hdf5File::parse(patched_bytes)
  assert_eq(
    integer_values(patched_file.read_dataset("/zeros")),
    random.map(value => value.to_int64()),
  )
  assert_eq(
    patched_file.superblock.end_of_file_address,
    patched_bytes.length().to_uint64() - patched_file.superblock.base_address,
  )
  let second = Array::make(64, 0)
  for index in 0..<64 {
    second[index] = random[index] ^ (index + 1)
  }
  let second_file = Hdf5File::parse(
    patched_file.patch_dataset_values("/zeros", Int32Values(second)),
  )
  assert_eq(
    integer_values(second_file.read_dataset("/zeros")),
    second.map(value => value.to_int64()),
  )
}

///|
test "filtered patches cover external scale-offset byte orders and pipelines" {
  let file = Hdf5File::parse(@fs.read_file_to_bytes("testdata/scaleoffset.h5"))

  let auto_expected = patch_int32_sequence(20, 100, 1)
  for index in 10..<20 {
    auto_expected[index] = -7
  }
  let auto = Hdf5File::parse(
    file.patch_dataset_values("/int32_auto", Int32Values(auto_expected)),
  )
  assert_eq(
    integer_values(auto.read_dataset("/int32_auto")),
    auto_expected.map(value => value.to_int64()),
  )
  let fixed_expected = patch_int32_sequence(16, 200, 1)
  let fixed = Hdf5File::parse(
    file.patch_dataset_values("/int32_fixed", Int32Values(fixed_expected)),
  )
  assert_eq(
    integer_values(fixed.read_dataset("/int32_fixed")),
    fixed_expected.map(value => value.to_int64()),
  )
  let int16_expected = [-301, -250, -199, -150, -99, -49]
  let int16 = Hdf5File::parse(
    file.patch_dataset_values("/int16_be", Int16Values(int16_expected)),
  )
  assert_eq(
    integer_values(int16.read_dataset("/int16_be")),
    int16_expected.map(value => value.to_int64()),
  )
  let float32_expected = [2.25, 3.5, -2.75, 8.5, -9.5, 0.25]
  let float32 = Hdf5File::parse(
    file.patch_dataset_values(
      "/float32_be",
      Float32Values([
        Float(2.25),
        Float(3.5),
        Float(-2.75),
        Float(8.5),
        Float(-9.5),
        Float(0.25),
      ]),
    ),
  )
  assert_eq(float_values(float32.read_dataset("/float32_be")), float32_expected)

  let pipeline_expected = [10.25, 10.25, 10.25, -9.5, -9.5]
  let pipeline = Hdf5File::parse(
    file.patch_dataset_values(
      "/float32_pipeline",
      Float32Values([
        Float(10.25),
        Float(10.25),
        Float(10.25),
        Float(-9.5),
        Float(-9.5),
      ]),
    ),
  )
  assert_eq(
    float_values(pipeline.read_dataset("/float32_pipeline")),
    pipeline_expected,
  )
  let full_expected = [-1, -2, -3, -4]
  let full = Hdf5File::parse(
    file.patch_dataset_values("/int32_full", Int32Values(full_expected)),
  )
  assert_eq(
    integer_values(full.read_dataset("/int32_full")),
    full_expected.map(value => value.to_int64()),
  )
}

///|
test "patch_dataset_values allocates unallocated contiguous storage" {
  for
    path in ["testdata/contiguous_fill_v2.h5", "testdata/contiguous_fill_v3.h5"] {
    let bytes = @fs.read_file_to_bytes(path)
    let file = Hdf5File::parse(bytes)
    let strict = try {
      file.patch_contiguous_dataset("/signed", Int32Values([1, 2, 3])) |> ignore
      None
    } catch {
      error => Some(error)
    }
    assert_eq(
      strict,
      Some(
        UnsupportedFeature("unallocated contiguous storage cannot be patched"),
      ),
    )
    let patched_bytes = file.patch_dataset_values(
      "/signed",
      Int32Values([11, -12, 13]),
    )
    assert_true(patched_bytes.length() > bytes.length())
    let patched = Hdf5File::parse(patched_bytes)
    assert_eq(integer_values(patched.read_dataset("/signed")), [11L, -12L, 13L])
    assert_eq(
      patched.superblock.end_of_file_address,
      patched_bytes.length().to_uint64() - patched.superblock.base_address,
    )
    let second = Hdf5File::parse(
      patched.patch_dataset_values("/signed", Int32Values([21, 22, 23])),
    )
    assert_eq(integer_values(second.read_dataset("/signed")), [21L, 22L, 23L])
  }
}

///|
test "missing single and fixed-array chunks allocate at the file tail" {
  let layout_bytes = @fs.read_file_to_bytes("testdata/layout_v4.h5")
  let layout = Hdf5File::parse(layout_bytes)
  let single_expected = patch_int32_sequence(12, 40, 3)
  let single_bytes = layout.patch_dataset_values(
    "/single/unallocated",
    Int32Values(single_expected),
  )
  assert_true(single_bytes.length() > layout_bytes.length())
  let single = Hdf5File::parse(single_bytes)
  assert_true(
    single.read_dataset_metadata("/single/unallocated").layout.chunk_index
    is Single(_, _),
  )
  assert_eq(
    integer_values(single.read_dataset("/single/unallocated")),
    single_expected.map(value => value.to_int64()),
  )

  let fixed_bytes = @fs.read_file_to_bytes("testdata/fixed_array.h5")
  let fixed = Hdf5File::parse(fixed_bytes)
  let fixed_expected = patch_int32_sequence(8, 100, 10)
  let patched_bytes = fixed.patch_dataset_values(
    "/skipped",
    Int32Values(fixed_expected),
  )
  assert_true(patched_bytes.length() > fixed_bytes.length())
  let patched = Hdf5File::parse(patched_bytes)
  assert_eq(
    integer_values(patched.read_dataset("/skipped")),
    fixed_expected.map(value => value.to_int64()),
  )
}

///|
test "filtered missing fixed and extensible chunks allocate existing slots" {
  let fixed_writer = Hdf5Writer::new()
  fixed_writer.add_chunked_dataset(
    "/fixed",
    [8UL],
    [2U],
    Int32Values(Array::make(8, -7)),
    fill_value=Int32Values([-7]),
    filters=[Deflate(6)],
  )
  let fixed_bytes = fixed_writer.encode()
  let fixed = Hdf5File::parse(fixed_bytes)
  let fixed_expected = patch_int32_sequence(8, 200, 2)
  let fixed_patched_bytes = fixed.patch_dataset_values(
    "/fixed",
    Int32Values(fixed_expected),
  )
  assert_true(fixed_patched_bytes.length() > fixed_bytes.length())
  let fixed_patched = Hdf5File::parse(fixed_patched_bytes)
  assert_eq(
    integer_values(fixed_patched.read_dataset("/fixed")),
    fixed_expected.map(value => value.to_int64()),
  )

  let extensible_writer = Hdf5Writer::new()
  let extensible_values = Array::make(8, -7)
  extensible_values[7] = 99
  extensible_writer.add_extensible_dataset(
    "/extensible",
    [8UL],
    [None],
    [2U],
    Int32Values(extensible_values),
    fill_value=Int32Values([-7]),
    filters=[Deflate(6)],
  )
  let extensible_bytes = extensible_writer.encode()
  let extensible = Hdf5File::parse(extensible_bytes)
  assert_true(
    extensible.read_dataset_metadata("/extensible").layout.chunk_index
    is ExtensibleArray(_),
  )
  let extensible_expected = patch_int32_sequence(8, 300, 3)
  let extensible_patched_bytes = extensible.patch_dataset_values(
    "/extensible",
    Int32Values(extensible_expected),
  )
  assert_true(extensible_patched_bytes.length() > extensible_bytes.length())
  let extensible_patched = Hdf5File::parse(extensible_patched_bytes)
  assert_eq(
    integer_values(extensible_patched.read_dataset("/extensible")),
    extensible_expected.map(value => value.to_int64()),
  )
}

///|
test "unfiltered missing fixed and extensible slots refresh index checksums" {
  let fixed_writer = Hdf5Writer::new()
  let fixed_values = Array::make(8, -7)
  fixed_values[4] = 40
  fixed_values[5] = 41
  fixed_values[6] = 42
  fixed_values[7] = 43
  fixed_writer.add_chunked_dataset(
    "/fixed",
    [8UL],
    [2U],
    Int32Values(fixed_values),
    fill_value=Int32Values([-7]),
  )
  let fixed_bytes = fixed_writer.encode()
  let fixed = Hdf5File::parse(fixed_bytes)
  let fixed_expected = patch_int32_sequence(8, -100, 1)
  let fixed_patched_bytes = fixed.patch_dataset_values(
    "/fixed",
    Int32Values(fixed_expected),
  )
  assert_true(fixed_patched_bytes.length() > fixed_bytes.length())
  let fixed_patched = Hdf5File::parse(fixed_patched_bytes)
  assert_eq(
    integer_values(fixed_patched.read_dataset("/fixed")),
    fixed_expected.map(value => value.to_int64()),
  )

  let extensible_writer = Hdf5Writer::new()
  let extensible_values = Array::make(8, -7)
  extensible_values[6] = 60
  extensible_values[7] = 61
  extensible_writer.add_extensible_dataset(
    "/extensible",
    [8UL],
    [None],
    [2U],
    Int32Values(extensible_values),
    fill_value=Int32Values([-7]),
  )
  let extensible_bytes = extensible_writer.encode()
  let extensible = Hdf5File::parse(extensible_bytes)
  let extensible_expected = patch_int32_sequence(8, -200, 1)
  let extensible_patched_bytes = extensible.patch_dataset_values(
    "/extensible",
    Int32Values(extensible_expected),
  )
  assert_true(extensible_patched_bytes.length() > extensible_bytes.length())
  let extensible_patched = Hdf5File::parse(extensible_patched_bytes)
  assert_eq(
    integer_values(extensible_patched.read_dataset("/extensible")),
    extensible_expected.map(value => value.to_int64()),
  )
}

///|
test "missing fixed-array pages allocate inside an existing data block" {
  let bytes = @fs.read_file_to_bytes("testdata/fixed_array.h5")
  let file = Hdf5File::parse(bytes)
  for path in ["/paged", "/paged_filtered"] {
    let expected = Array::make(4099, -7)
    expected[1200] = 1200
    expected[1500] = 1500
    let patched_bytes = file.patch_dataset_values(path, Int32Values(expected))
    assert_true(patched_bytes.length() > bytes.length())
    let patched = Hdf5File::parse(patched_bytes)
    assert_true(
      patched.read_dataset_metadata(path).layout.chunk_index is FixedArray(_),
    )
    assert_eq(
      integer_values(patched.read_dataset(path)),
      expected.map(value => value.to_int64()),
    )
  }
}

///|
test "missing fixed-array data blocks allocate new index storage" {
  let bytes = @fs.read_file_to_bytes("testdata/fixed_array.h5")
  let raw_expected = patch_int32_sequence(35, 500, 2)
  let filtered_expected = patch_int32_sequence(35, -900, 3)
  let paged_expected = Array::make(4099, -7)
  for index in [0, 1023, 2048, 4098] {
    paged_expected[index] = 7000 + index
  }
  let cases : Array[(String, Array[Int32])] = [
    ("/raw", raw_expected),
    ("/filtered", filtered_expected),
    ("/paged", paged_expected),
    ("/paged_filtered", paged_expected),
  ]
  for (path, expected) in cases {
    let missing_bytes = patch_fixed_array_without_data_block(bytes, path)
    let missing = Hdf5File::parse(missing_bytes)
    assert_eq(
      integer_values(missing.read_dataset(path)).length(),
      expected.length(),
    )
    let patched_bytes = missing.patch_dataset_values(
      path,
      Int32Values(expected),
    )
    assert_true(patched_bytes.length() > missing_bytes.length())
    let patched = Hdf5File::parse(patched_bytes)
    assert_true(
      patched.read_dataset_metadata(path).layout.chunk_index is FixedArray(_),
    )
    assert_eq(
      integer_values(patched.read_dataset(path)),
      expected.map(value => value.to_int64()),
    )
    assert_eq(
      patched.superblock.end_of_file_address,
      patched_bytes.length().to_uint64() - patched.superblock.base_address,
    )
    let revised = expected.copy()
    let revised_index = if expected.length() > 1000 { 1500 } else { 1 }
    revised[revised_index] = 123456
    let revised_bytes = patched.patch_dataset_values(path, Int32Values(revised))
    let revised_file = Hdf5File::parse(revised_bytes)
    let revised_values = integer_values(revised_file.read_dataset(path))
    assert_eq(revised_values[revised_index], 123456L)
    assert_eq(revised_values.length(), expected.length())
  }
}

///|
test "missing extensible-array pages allocate inside existing data blocks" {
  let bytes = @fs.read_file_to_bytes("testdata/extensible_array.h5")
  let file = Hdf5File::parse(bytes)
  let existing = [
    0, 3, 4, 19, 20, 51, 52, 115, 116, 243, 244, 499, 500, 1011, 1012, 65523, 65524,
    131060, 133108, 199999,
  ]
  for path in ["/raw/paged", "/filtered/paged"] {
    let expected = Array::make(200000, -7)
    for index in existing {
      expected[index] = index
    }
    for index in [132500, 132700, 134500, 199500] {
      expected[index] = 400000 + index
    }
    let patched_bytes = file.patch_dataset_values(path, Int32Values(expected))
    assert_true(patched_bytes.length() > bytes.length())
    let patched = Hdf5File::parse(patched_bytes)
    assert_true(
      patched.read_dataset_metadata(path).layout.chunk_index
      is ExtensibleArray(_),
    )
    let values = integer_values(patched.read_dataset(path))
    assert_eq(values.length(), expected.length())
    for index in [0, 132500, 132700, 134500, 199500, 199999] {
      assert_eq(values[index], expected[index].to_int64())
    }
    assert_eq(values[132499], -7L)
    assert_eq(values[134501], -7L)
  }
}

///|
fn clear_extensible_index_address(
  bytes : Bytes,
  path : String,
  clear_maximum_set~ : Bool = false,
) -> Bytes raise Hdf5Error {
  let file = Hdf5File::parse(bytes)
  let metadata = file.read_dataset_metadata(path)
  let info = read_extensible_array_header_info(
    bytes,
    metadata.layout,
    file.superblock.offset_size,
    file.superblock.length_size,
    file.superblock.base_address,
  )
  guard info.index_address != 0xFFFFFFFFFFFFFFFFUL &&
    info.index_address_width == file.superblock.offset_size else {
    raise InvalidField("test file has no writable extensible-array index")
  }
  let output = bytes.to_fixedarray()
  patch_little_endian_uint(
    output,
    info.index_address_offset,
    info.index_address_width,
    0xFFFFFFFFFFFFFFFFUL,
    "extensible array test index address",
  )
  if clear_maximum_set {
    patch_little_endian_uint(
      output,
      info.maximum_set_offset,
      file.superblock.length_size,
      0UL,
      "extensible array test maximum set",
    )
  }
  patch_metadata_checksum_in_place(
    output,
    address_to_offset(
      metadata.layout.data_address,
      file.superblock.base_address,
    ),
    info.header_checksum_offset,
  )
  Bytes::from_array(output)
}

///|
fn clear_extensible_super_pointer(
  bytes : Bytes,
  path : String,
  target_slot : Int64,
) -> Bytes raise Hdf5Error {
  let file = Hdf5File::parse(bytes)
  let metadata = file.read_dataset_metadata(path)
  let config = match metadata.layout.chunk_index {
    ExtensibleArray(config) => config
    _ => raise InvalidField("test dataset is not an extensible array")
  }
  let info = read_extensible_array_header_info(
    bytes,
    metadata.layout,
    file.superblock.offset_size,
    file.superblock.length_size,
    file.superblock.base_address,
  )
  let index_offset = address_to_offset(
    info.index_address,
    file.superblock.base_address,
  )
  let offset_size = file.superblock.offset_size
  let entry_start = 6 + offset_size
  let direct_count = 2 * (config.min_pointers - 1)
  let first_super = 2 * array_log2(config.min_pointers)
  let levels = 1 + config.max_bits - array_log2(config.min_elements)
  let direct_start = entry_start + config.index_elements * info.entry_size
  let super_start = direct_start + direct_count * offset_size
  let index_content = super_start + (levels - first_super) * offset_size
  let mut level_start = 0L
  let mut pointer_offset = -1
  for level in 0..<levels {
    let blocks = 1L << (level / 2)
    let count = config.min_elements.to_int64() << ((level + 1) / 2)
    if level >= first_super {
      let start = config.index_elements.to_int64() + level_start
      if target_slot >= start && target_slot < start + blocks * count {
        pointer_offset = index_offset +
          super_start +
          (level - first_super) * offset_size
        break
      }
    }
    level_start = level_start + blocks * count
  }
  guard pointer_offset >= 0 else {
    raise InvalidField("test target is not in an extensible-array super block")
  }
  guard Reader::at(bytes, pointer_offset).read_uint_le(offset_size) !=
    0xFFFFFFFFFFFFFFFFUL else {
    raise InvalidField("test super block pointer is already undefined")
  }
  let output = bytes.to_fixedarray()
  patch_little_endian_uint(
    output, pointer_offset, offset_size, 0xFFFFFFFFFFFFFFFFUL, "extensible array test super address",
  )
  patch_metadata_checksum_in_place(
    output,
    index_offset,
    index_offset + index_content,
  )
  Bytes::from_array(output)
}

///|
fn patch_extensible_missing_block_case(
  extent : Int32,
  seed_positions : Array[Int32],
  write_positions : Array[Int32],
  filters : Array[Hdf5WriteFilter],
) -> Unit raise {
  patch_extensible_structure_case(
    extent,
    seed_positions,
    write_positions,
    filters,
    bytes => bytes,
  )
}

///|
fn patch_extensible_structure_case(
  extent : Int32,
  seed_positions : Array[Int32],
  write_positions : Array[Int32],
  filters : Array[Hdf5WriteFilter],
  mutate : (Bytes) -> Bytes raise Hdf5Error,
) -> Unit raise {
  let original = Array::make(extent, -7)
  for position in seed_positions {
    original[position] = 1000 + position
  }
  let writer = Hdf5Writer::new()
  writer.add_extensible_dataset(
    "/data",
    [extent.to_uint64()],
    [None],
    [1U],
    Int32Values(original),
    fill_value=Int32Values([-7]),
    filters~,
  )
  let bytes = mutate(writer.encode())
  let file = Hdf5File::parse(bytes)
  assert_true(
    file.read_dataset_metadata("/data").layout.chunk_index is ExtensibleArray(_),
  )
  let original_values = integer_values(file.read_dataset("/data"))
  assert_eq(original_values.length().to_int64(), extent.to_int64())
  for position in write_positions {
    assert_eq(original_values[position], -7L)
  }
  let expected = original.copy()
  for position in write_positions {
    expected[position] = 500000 + position
  }
  let patched_bytes = file.patch_dataset_values("/data", Int32Values(expected))
  assert_true(patched_bytes.length() > bytes.length())
  let patched = Hdf5File::parse(patched_bytes)
  assert_true(
    patched.read_dataset_metadata("/data").layout.chunk_index
    is ExtensibleArray(_),
  )
  let patched_values = integer_values(patched.read_dataset("/data"))
  assert_eq(patched_values, expected.map(value => value.to_int64()))
  assert_eq(
    patched.superblock.end_of_file_address,
    patched_bytes.length().to_uint64() - patched.superblock.base_address,
  )
  let revised = expected.copy()
  let revised_position = write_positions[write_positions.length() - 1]
  revised[revised_position] = 987654
  let revised_bytes = patched.patch_dataset_values(
    "/data",
    Int32Values(revised),
  )
  let revised_file = Hdf5File::parse(revised_bytes)
  assert_eq(
    integer_values(revised_file.read_dataset("/data")),
    revised.map(value => value.to_int64()),
  )
}

///|
test "missing extensible-array data blocks allocate direct and super blocks" {
  let cases : Array[(Int32, Array[Int32], Array[Int32], Array[Hdf5WriteFilter])] = [
    (300, [10, 299], [200, 220], []),
    (300, [10, 299], [200, 220], [ScaleOffset(0)]),
    (5000, [4090, 4700, 4999], [4400, 4500], []),
    (5000, [4090, 4700, 4999], [4400, 4500], [ScaleOffset(0)]),
    (138000, [131100, 136000, 137900], [134000, 134500], []),
    (138000, [131100, 136000, 137900], [134000, 134500], [ScaleOffset(0)]),
  ]
  for (extent, seeds, writes, filters) in cases {
    patch_extensible_missing_block_case(extent, seeds, writes, filters)
  }
}

///|
test "missing extensible-array index blocks are recreated" {
  let cases : Array[(Int32, Array[Int32], Array[Int32], Array[Hdf5WriteFilter])] = [
    (4, [1, 3], [2], []),
    (4, [1, 3], [2], [ScaleOffset(0)]),
    (300, [10, 299], [200, 220], []),
    (300, [10, 299], [200, 220], [ScaleOffset(0)]),
    (5000, [4090, 4700, 4999], [4400, 4500], []),
    (5000, [4090, 4700, 4999], [4400, 4500], [ScaleOffset(0)]),
    (138000, [131100, 136000, 137900], [134000, 134500], []),
    (138000, [131100, 136000, 137900], [134000, 134500], [ScaleOffset(0)]),
  ]
  for (extent, seeds, writes, filters) in cases {
    patch_extensible_structure_case(extent, seeds, writes, filters, bytes => {
      clear_extensible_index_address(bytes, "/data")
    })
  }
}

///|
test "missing extensible-array index and maximum set are recreated" {
  let cases : Array[(Int32, Array[Int32], Array[Int32], Array[Hdf5WriteFilter])] = [
    (300, [10, 299], [200, 220], []),
    (300, [10, 299], [200, 220], [ScaleOffset(0)]),
    (5000, [4090, 4700, 4999], [4400, 4500], []),
    (5000, [4090, 4700, 4999], [4400, 4500], [ScaleOffset(0)]),
  ]
  for (extent, seeds, writes, filters) in cases {
    patch_extensible_structure_case(extent, seeds, writes, filters, bytes => {
      clear_extensible_index_address(bytes, "/data", clear_maximum_set=true)
    })
  }
}

///|
test "missing extensible-array super blocks are recreated" {
  let cases : Array[
    (Int32, Array[Int32], Array[Int32], Int64, Array[Hdf5WriteFilter]),
  ] = [
    (5000, [4090, 4700, 4999], [4400, 4500], 4400L, []),
    (5000, [4090, 4700, 4999], [4400, 4500], 4400L, [ScaleOffset(0)]),
    (138000, [131100, 136000, 137900], [134000, 134500], 134000L, []),
    (
      138000,
      [131100, 136000, 137900],
      [134000, 134500],
      134000L,
      [ScaleOffset(0)],
    ),
  ]
  for (extent, seeds, writes, super_slot, filters) in cases {
    patch_extensible_structure_case(extent, seeds, writes, filters, bytes => {
      clear_extensible_super_pointer(bytes, "/data", super_slot)
    })
  }
}
