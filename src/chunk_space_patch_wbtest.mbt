///|
fn patch_space_noise(count : Int32) -> Array[Int32] {
  let mut state = 123456789U
  Array::makei(count, _index => {
    state = state ^ (state << 13)
    state = state ^ (state >> 17)
    state = state ^ (state << 5)
    state.reinterpret_as_int()
  })
}

///|
fn patch_space_stored(file : Hdf5File, path : String) -> UInt64 raise Hdf5Error {
  let mut total = 0UL
  for record in relocation_chunk_records(file, path) {
    total = total + record.size
  }
  total
}

///|
test "filtered chunk growth reuses reserve and shrink returns unused bytes across indexes" {
  let noise = patch_space_noise(1024)
  for kind in 0..<4 {
    let writer = Hdf5Writer::new()
    writer.set_file_space_reserve(65536)
    writer.set_user_block_content(b"patch-space")
    let values = Int32Values(Array::make(1024, 1))
    let filters : Array[Hdf5WriteFilter] = [Shuffle, Deflate(4), Fletcher32]
    match kind {
      0 =>
        writer.add_chunked_dataset(
          "/data",
          [32UL, 32UL],
          [32U, 32U],
          values,
          filters~,
        )
      1 =>
        writer.add_chunked_dataset(
          "/data",
          [32UL, 32UL],
          [8U, 8U],
          values,
          filters~,
        )
      2 =>
        writer.add_extensible_dataset(
          "/data",
          [32UL, 32UL],
          [None, Some(32UL)],
          [8U, 8U],
          values,
          filters~,
        )
      _ =>
        writer.add_extensible_dataset(
          "/data",
          [32UL, 32UL],
          [None, None],
          [8U, 8U],
          values,
          filters~,
        )
    }
    let file = Hdf5File::parse(writer.encode())
    let old_stored = patch_space_stored(file, "/data")
    let grown = Hdf5File::parse(
      file.patch_dataset_values("/data", Int32Values(noise)),
    )
    assert_eq(
      integer_values(grown.read_dataset("/data")),
      noise.map(value => value.to_int64()),
    )
    assert_eq(
      allocation_free_total(grown),
      65536UL + old_stored - patch_space_stored(grown, "/data"),
    )
    let reserve = file.free_space_managers()[0].sections[0]
    for record in relocation_chunk_records(grown, "/data") {
      assert_true(
        record.address >= reserve.address &&
        record.address + record.size <= reserve.address + reserve.size,
      )
    }
    let shrunk = Hdf5File::parse(
      grown.patch_dataset_values("/data", Int32Values(Array::make(1024, 2))),
    )
    assert_eq(
      integer_values(shrunk.read_dataset("/data")),
      Array::make(1024, 2L),
    )
    assert_eq(
      allocation_free_total(shrunk),
      65536UL + old_stored - patch_space_stored(shrunk, "/data"),
    )
    let repeated = Hdf5File::parse(
      shrunk.patch_dataset_values("/data", Int32Values(noise)),
    )
    assert_eq(
      integer_values(repeated.read_dataset("/data")),
      noise.map(value => value.to_int64()),
    )
    assert_eq(
      allocation_free_total(repeated),
      65536UL + old_stored - patch_space_stored(repeated, "/data"),
    )
  }
}

///|
test "chunk growth appends when reserve is insufficient but returns old storage" {
  let writer = Hdf5Writer::new()
  writer.set_file_space_reserve(1)
  writer.add_chunked_dataset(
    "/data",
    [1024UL],
    [256U],
    Int32Values(Array::make(1024, 1)),
    filters=[Deflate(4)],
  )
  let original = Hdf5File::parse(writer.encode())
  let old = patch_space_stored(original, "/data")
  let noise = patch_space_noise(1024)
  let patched = Hdf5File::parse(
    original.patch_dataset_values("/data", Int32Values(noise)),
  )
  assert_eq(
    integer_values(patched.read_dataset("/data")),
    noise.map(value => value.to_int64()),
  )
  assert_eq(allocation_free_total(patched), 1UL + old)
  for record in relocation_chunk_records(patched, "/data") {
    assert_true(record.address >= original.bytes.length().to_uint64())
  }
}

///|
test "missing chunks consume reserved raw space without losing fill slots" {
  for kind in 0..<3 {
    let writer = Hdf5Writer::new()
    writer.set_file_space_reserve(4096)
    let values = Int32Values(Array::make(16, -1))
    match kind {
      0 =>
        writer.add_chunked_dataset(
          "/data",
          [4UL, 4UL],
          [2U, 2U],
          values,
          fill_value=Int32Values([-1]),
        )
      1 =>
        writer.add_extensible_dataset(
          "/data",
          [4UL, 4UL],
          [None, Some(4UL)],
          [2U, 2U],
          values,
          fill_value=Int32Values([-1]),
        )
      _ =>
        writer.add_extensible_dataset(
          "/data",
          [4UL, 4UL],
          [None, None],
          [2U, 2U],
          values,
          fill_value=Int32Values([-1]),
        )
    }
    let file = Hdf5File::parse(writer.encode())
    let values = Array::make(16, -1)
    values[0] = 42
    let patched = Hdf5File::parse(
      file.patch_dataset_values("/data", Int32Values(values)),
    )
    assert_eq(
      integer_values(patched.read_dataset("/data")),
      values.map(value => value.to_int64()),
    )
    assert_eq(allocation_free_total(patched), 4096UL - 16UL)
    assert_eq(relocation_chunk_records(patched, "/data").length(), 1)
  }
}

///|
test "variable-length chunk patching keeps heaps valid while recycling raw bytes" {
  let writer = Hdf5Writer::new()
  writer.set_file_space_reserve(16384)
  writer.add_chunked_dataset(
    "/names",
    [4UL],
    [2U],
    VariableStringValues(["a", "b", "", "a"]),
    filters=[Shuffle, Deflate(4)],
  )
  let original = Hdf5File::parse(writer.encode())
  let before = patch_space_stored(original, "/names")
  let values = ["a much longer value", "月亮", "new", ""]
  let patched = Hdf5File::parse(
    original.patch_dataset_values("/names", VariableStringValues(values)),
  )
  match patched.read_dataset("/names").values {
    Strings(actual) => assert_eq(actual, values)
    _ => abort("expected patched strings")
  }
  assert_eq(
    allocation_free_total(patched),
    16384UL + before - patch_space_stored(patched, "/names"),
  )
}

///|
test "chunk reuse rejects live storage overlapping a recorded free section" {
  let writer = Hdf5Writer::new()
  writer.set_file_space_reserve(4096)
  writer.add_chunked_dataset(
    "/data",
    [8UL],
    [4U],
    Int32Values(Array::make(8, 1)),
    filters=[Deflate(4)],
  )
  let file = Hdf5File::parse(writer.encode())
  let record = relocation_chunk_records(file, "/data")[0]
  let free = file.free_space_managers()[0].sections[0]
  let output = file.bytes.to_fixedarray()
  patch_chunk_address_field(output, record, free.address)
  patch_metadata_checksum_in_place(
    output,
    record.metadata_start,
    record.metadata_checksum_offset,
  )
  let corrupt = Hdf5File::parse(output.unsafe_reinterpret_as_bytes())
  assert_true(
    try {
      corrupt.patch_dataset_values("/data", Int32Values(Array::make(8, 42)))
      |> ignore
      false
    } catch {
      InvalidField(message) => message.contains("overlaps recorded free space")
      _ => false
    },
  )
}
