///|
test "new writers can create a persistent raw-space reserve" {
  for reserve in [1, 64, 8192] {
    let writer = Hdf5Writer::new()
    writer.set_file_space_reserve(reserve)
    writer.add_dataset("/values", [4UL], Int32Values([1, 2, 3, 4]))
    let bytes = writer.encode()
    let file = Hdf5File::parse(bytes)
    let info = file.file_space_info().unwrap()
    assert_true(info.strategy == FsmAggregate && info.persistent)
    assert_eq(info.threshold, 1UL)
    let managers = file.free_space_managers()
    assert_eq(managers.length(), 1)
    assert_eq(managers[0].address, info.manager_addresses[2])
    assert_eq(managers[0].total_space, reserve.to_uint64())
    assert_eq(managers[0].sections.length(), 1)
    let section = managers[0].sections[0]
    assert_eq(section.size, reserve.to_uint64())
    let absolute = address_to_offset(
      section.address,
      file.superblock.base_address,
    )
    for index in 0..<reserve {
      assert_eq(bytes[absolute + index], b'\x00')
    }
    assert_eq(
      Hdf5File::parse_source(bytes).free_space_managers()[0].total_space,
      reserve.to_uint64(),
    )
  }
}

///|
test "new reserved storage supports relocation and preserves its free total" {
  let writer = Hdf5Writer::new()
  writer.set_file_space_reserve(8192)
  writer.add_dataset("/values", [4UL], Int32Values([1, 2, 3, 4]))
  let original = Hdf5File::parse(writer.encode())
  let moved = Hdf5File::parse(original.relocate_dataset_storage("/values"))
  assert_true(
    moved.read_dataset_metadata("/values").layout.data_address !=
    original.read_dataset_metadata("/values").layout.data_address,
  )
  assert_eq(allocation_free_total(moved), 8192UL)
  match moved.read_dataset("/values").values {
    Integers(values) => assert_eq(values, [1L, 2L, 3L, 4L])
    _ => abort("expected integers")
  }
}

///|
test "reserve options reject invalid sizes and disabling preserves default output" {
  let writer = Hdf5Writer::new()
  writer.add_dataset("/values", [1UL], Int32Values([42]))
  let original = writer.encode()
  writer.set_file_space_reserve(1024)
  assert_true(writer.encode().length() > original.length())
  writer.set_file_space_reserve(0)
  assert_eq(writer.encode(), original)
  for size in [-1, 0x7FFFFFFF] {
    assert_true(
      try {
        writer.set_file_space_reserve(size)
        false
      } catch {
        InvalidArgument(_) => true
        _ => false
      },
    )
  }
}

///|
test "reserved storage works with user blocks filters and source output" {
  let writer = Hdf5Writer::new()
  writer.set_file_space_reserve(8192)
  writer.set_user_block_content(b"space-reserve")
  for path in ["/first", "/second", "/third"] {
    writer.add_chunked_dataset(
      path,
      [4UL],
      [2U],
      Int32Values([1, 2, 3, 4]),
      fill_value=Int32Values([-1]),
      filters=[Shuffle, Deflate(4)],
    )
    writer.add_attribute(path, "unit", [], FixedStringValues(["items"], 8))
  }
  let source = TypedInt32Source::{
    dimensions: [4UL],
    values: [5, 6, 7, 8],
    typed_reads: 0,
    typed_largest: 0,
  }
  writer.add_dataset_from_source(
    "/streamed",
    [4UL],
    [2U],
    Int32Values([]),
    source,
  )
  let sink = Hdf5MemorySink::new()
  writer.write_to(sink)
  let bytes = sink.to_bytes()
  assert_eq(bytes, writer.encode())
  let file = Hdf5File::parse(bytes)
  assert_eq(file.superblock.base_address, 512UL)
  assert_eq(allocation_free_total(file), 8192UL)
  for path in ["/first", "/second", "/third"] {
    match file.read_dataset(path).values {
      Integers(values) => assert_eq(values, [1L, 2L, 3L, 4L])
      _ => abort("expected shared dataset integers")
    }
    assert_eq(file.attributes(path)[0].name, "unit")
  }
  let moved = Hdf5File::parse(file.relocate_dataset_storage("/streamed"))
  assert_eq(allocation_free_total(moved), 8192UL)
  match moved.read_dataset("/streamed").values {
    Integers(values) => assert_eq(values, [5L, 6L, 7L, 8L])
    _ => abort("expected streamed integers")
  }
}

///|
test "reserved storage rejects shared-message output before touching the sink" {
  for kind in 0..<3 {
    let writer = Hdf5Writer::new()
    writer.set_file_space_reserve(8192)
    if kind == 0 {
      writer.set_attribute_sharing(true)
    }
    if kind == 1 {
      writer.set_fill_sharing(true)
    }
    if kind == 2 {
      writer.set_filter_sharing(true)
    }
    let sink = Hdf5MemorySink::new()
    sink.write_at(0UL, b"unchanged")
    assert_true(
      try {
        writer.write_to(sink)
        false
      } catch {
        UnsupportedFeature(_) => true
        _ => false
      },
    )
    assert_eq(sink.to_bytes(), b"unchanged")
    writer.set_attribute_sharing(false)
    writer.set_fill_sharing(false)
    writer.set_filter_sharing(false)
    assert_eq(allocation_free_total(Hdf5File::parse(writer.encode())), 8192UL)
  }
}
