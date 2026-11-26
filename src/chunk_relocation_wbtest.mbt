///|
fn relocation_chunk_records(
  file : Hdf5File,
  path : String,
) -> Array[RawChunkRecord] raise Hdf5Error {
  let header = file.object_header_at(file.resolve(path).address)
  let metadata = file.read_dataset_metadata(path)
  let filters = match header.first(msg_filter_pipeline) {
    Some(message) => parse_filters(message)
    None => []
  }
  if metadata.layout.chunk_index is Single(_, _) {
    return [
      single_chunk_slot_record(
        header,
        metadata,
        chunk_byte_size(
          metadata.layout.chunk_dimensions,
          metadata.datatype.size,
        ),
        file.superblock.length_size,
      ),
    ]
  }
  let absent = btree_v1_undefined_address(file.superblock.offset_size)
  collect_raw_chunk_records(
    file.bytes,
    metadata.layout,
    metadata.dataspace,
    metadata.datatype,
    filters,
    file.superblock.offset_size,
    file.superblock.length_size,
    file.superblock.base_address,
  ).filter(record => record.address != absent)
}

///|
test "chunk relocation preserves raw bytes masks values and free totals across indexes" {
  for filtered in [false, true] {
    let writer = Hdf5Writer::new()
    writer.set_file_space_reserve(65536)
    writer.set_user_block_content(b"chunk-relocation")
    let filters : Array[Hdf5WriteFilter] = if filtered {
      [Shuffle, Deflate(4), Fletcher32]
    } else {
      []
    }
    let values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    writer.add_chunked_dataset(
      "/single",
      [3UL, 4UL],
      [3U, 4U],
      Int32Values(values),
      filters~,
    )
    writer.add_chunked_dataset(
      "/fixed",
      [3UL, 4UL],
      [2U, 2U],
      Int32Values(values),
      filters~,
    )
    writer.add_extensible_dataset(
      "/array",
      [3UL, 4UL],
      [None, Some(4UL)],
      [2U, 2U],
      Int32Values(values),
      filters~,
    )
    writer.add_extensible_dataset(
      "/tree",
      [3UL, 4UL],
      [None, None],
      [2U, 2U],
      Int32Values(values),
      filters~,
    )
    if filtered {
      writer.set_chunk_filter_mask("/fixed", 0, 7U)
    }
    let mut file = Hdf5File::parse(
      writer.encode() catch {
        error => abort("encode: " + error.message())
      },
    )
    for path in ["/single", "/fixed", "/array", "/tree"] {
      let before = relocation_chunk_records(file, path) catch {
        error => abort(path + " records: " + error.message())
      }
      assert_true(before.length() > 0)
      let total = allocation_free_total(file)
      let moved = Hdf5File::parse(
        file.relocate_dataset_storage(path) catch {
          error => abort(path + " relocation: " + error.message())
        },
      )
      assert_eq(
        allocation_free_total(moved) catch {
          error => abort(path + " manager: " + error.message())
        },
        total,
      )
      let after = relocation_chunk_records(moved, path) catch {
        error => abort(path + " moved records: " + error.message())
      }
      assert_eq(before.length(), after.length())
      for index in 0..<before.length() {
        assert_true(before[index].address != after[index].address)
        assert_eq(after[index].address % 8UL, 0UL)
        assert_eq(after[index].size, before[index].size)
        assert_eq(after[index].mask, before[index].mask)
        assert_eq(
          file.space_read(before[index].address, before[index].size.to_int()),
          moved.space_read(after[index].address, after[index].size.to_int()),
        )
      }
      match
        (moved.read_dataset(path) catch {
          error => abort(path + " read: " + error.message())
        }).values {
        Integers(actual) =>
          assert_eq(actual, values.map(value => value.to_int64()))
        _ => abort("expected relocated integers")
      }
      let repeated = Hdf5File::parse(moved.relocate_dataset_storage(path))
      assert_eq(allocation_free_total(repeated), total)
      file = repeated
    }
  }
}

///|
test "chunk relocation fails atomically when reserve cannot fit every chunk" {
  let writer = Hdf5Writer::new()
  writer.set_file_space_reserve(24)
  writer.add_chunked_dataset(
    "/data",
    [8UL],
    [4U],
    Int32Values([1, 2, 3, 4, 5, 6, 7, 8]),
  )
  let bytes = writer.encode()
  let file = Hdf5File::parse(bytes)
  assert_true(
    try {
      file.relocate_dataset_storage("/data") |> ignore
      false
    } catch {
      UnsupportedFeature(message) => message.contains("insufficient")
      _ => false
    },
  )
  assert_eq(allocation_free_total(file), 24UL)
  assert_eq(file.bytes, bytes)
}

///|
test "chunk relocation preserves sparse fill slots and variable-length heaps" {
  let writer = Hdf5Writer::new()
  writer.set_file_space_reserve(16384)
  writer.add_chunked_dataset(
    "/sparse",
    [8UL],
    [2U],
    Int32Values([42, 1, -1, -1, -1, -1, 7, 8]),
    fill_value=Int32Values([-1]),
    filters=[Deflate(4)],
  )
  writer.add_chunked_dataset(
    "/empty",
    [8UL],
    [2U],
    Int32Values(Array::make(8, -1)),
    fill_value=Int32Values([-1]),
  )
  writer.add_chunked_dataset(
    "/names",
    [5UL],
    [2U],
    VariableStringValues(["alpha", "月亮", "", "beta", "alpha"]),
    filters=[Deflate(4)],
  )
  let original = Hdf5File::parse(writer.encode())
  let sparse = Hdf5File::parse(original.relocate_dataset_storage("/sparse"))
  assert_eq(relocation_chunk_records(sparse, "/sparse").length(), 2)
  match sparse.read_dataset("/sparse").values {
    Integers(values) => assert_eq(values, [42L, 1L, -1L, -1L, -1L, -1L, 7L, 8L])
    _ => abort("expected sparse integers")
  }
  assert_eq(original.relocate_dataset_storage("/empty"), original.bytes)
  let names = Hdf5File::parse(sparse.relocate_dataset_storage("/names"))
  match names.read_dataset("/names").values {
    Strings(values) =>
      assert_eq(values, ["alpha", "月亮", "", "beta", "alpha"])
    _ => abort("expected relocated strings")
  }
  assert_eq(allocation_free_total(names), 16384UL)
}

///|
test "uncompressed multi-level v2 chunk trees retain valid node checksums" {
  let writer = Hdf5Writer::new()
  writer.set_file_space_reserve(65536)
  let values = Array::make(400, 0)
  for index in 0..<values.length() {
    values[index] = index + 1
  }
  writer.add_extensible_dataset(
    "/tree",
    [20UL, 20UL],
    [None, None],
    [1U, 1U],
    Int32Values(values),
  )
  let file = Hdf5File::parse(writer.encode())
  let moved = Hdf5File::parse(file.relocate_dataset_storage("/tree"))
  let records = relocation_chunk_records(moved, "/tree")
  assert_eq(records.length(), 400)
  assert_true(
    records.all(record => {
      record.metadata_start >= 0 && record.metadata_checksum_offset >= 0
    }),
  )
  match moved.read_dataset("/tree").values {
    Integers(actual) => assert_eq(actual, values.map(value => value.to_int64()))
    _ => abort("expected v2 tree integers")
  }
  assert_eq(allocation_free_total(moved), 65536UL)
}

///|
test "chunk relocation rejects implicit indexes and unsupported space strategies" {
  let implicit = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/layout_v4.h5"),
  )
  assert_true(
    try {
      implicit.relocate_dataset_storage("/implicit/edge") |> ignore
      false
    } catch {
      UnsupportedFeature(message) => message.contains("explicit-index")
      _ => false
    },
  )
  let writer = Hdf5Writer::new()
  writer.add_chunked_dataset("/chunks", [4UL], [2U], Int32Values([1, 2, 3, 4]))
  let file = Hdf5File::parse(writer.encode())
  assert_true(
    try {
      file.relocate_dataset_storage("/chunks") |> ignore
      false
    } catch {
      UnsupportedFeature(message) => message.contains("persistent aggregate")
      _ => false
    },
  )
}
