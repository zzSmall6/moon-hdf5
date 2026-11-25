///|
fn allocation_free_total(file : Hdf5File) -> UInt64 raise Hdf5Error {
  let mut total = 0UL
  for manager in file.free_space_managers() {
    total = total + manager.total_space
  }
  total
}

///|
test "simple free-space allocation splits aligns and selects best fit" {
  let sections = [
    { address: 1UL, size: 100UL, section_type: 0, },
    { address: 201UL, size: 39UL, section_type: 0, },
  ]
  let (address, rest) = space_allocate(sections, 16UL, 8UL).unwrap()
  assert_eq(address, 208UL)
  assert_eq(rest, [
    sections[0],
    { address: 201UL, size: 7UL, section_type: 0, },
    { address: 224UL, size: 16UL, section_type: 0, },
  ])
  let (exact, remaining) = space_allocate([sections[1]], 39UL, 1UL).unwrap()
  assert_eq(exact, 201UL)
  assert_eq(remaining.length(), 0)
  assert_true(space_allocate(sections, 1000UL, 8UL) is None)
  assert_eq(sections[1].size, 39UL)
}

///|
test "free-space release merges neighbors and rejects overlaps and overflow" {
  assert_eq(
    space_coalesce(
      [
        { address: 32UL, size: 16UL, section_type: 0, },
        { address: 0UL, size: 16UL, section_type: 0, },
        { address: 16UL, size: 16UL, section_type: 0, },
      ],
      100UL,
    ),
    [{ address: 0UL, size: 48UL, section_type: 0, }],
  )
  assert_true(
    try {
      space_coalesce(
        [
          { address: 10UL, size: 20UL, section_type: 0, },
          { address: 20UL, size: 20UL, section_type: 0, },
        ],
        100UL,
      )
      |> ignore
      false
    } catch {
      InvalidField(_) => true
      _ => false
    },
  )
  assert_true(
    try {
      space_allocate(
        [{ address: 0xFFFFFFFFFFFFFFFFUL, size: 1UL, section_type: 0, }],
        1UL,
        8UL,
      )
      |> ignore
      false
    } catch {
      InvalidField(_) => true
      _ => false
    },
  )
  assert_true(
    try {
      space_allocate([], 1UL, 3UL) |> ignore
      false
    } catch {
      InvalidArgument(_) => true
      _ => false
    },
  )
}

///|
test "contiguous relocation preserves values and returns old storage" {
  for name in ["fsm", "width4"] {
    let bytes = @fs.read_file_to_bytes("testdata/free_space_" + name + ".h5")
    let file = Hdf5File::parse(bytes)
    let original = file.read_dataset_metadata("/data_0").layout
    let total = allocation_free_total(file)
    let edited = Hdf5File::parse(file.relocate_dataset_storage("/data_0"))
    let layout = edited.read_dataset_metadata("/data_0").layout
    assert_true(layout.data_address != original.data_address)
    assert_eq(layout.data_address % 8UL, 0UL)
    assert_eq(layout.data_size, original.data_size)
    assert_eq(
      edited.resolve("/data_0").address,
      file.resolve("/data_0").address,
    )
    assert_eq(allocation_free_total(edited), total)
    assert_true(
      edited
      .free_space_managers()
      .any(manager => {
        manager.sections.any(section => {
          section.address <= original.data_address &&
          section.address + section.size >=
          original.data_address + original.data_size
        })
      }),
    )
    match
      (
        edited.read_dataset("/data_0").values,
        file.read_dataset("/data_0").values,
      ) {
      (Integers(actual), Integers(expected)) => assert_eq(actual, expected)
      _ => abort("expected integer data")
    }
    let repeated = Hdf5File::parse(edited.relocate_dataset_storage("/data_0"))
    assert_eq(allocation_free_total(repeated), total)
    for path in ["/data_0", "/data_2", "/data_4", "/data_6", "/data_7"] {
      match
        (repeated.read_dataset(path).values, file.read_dataset(path).values) {
        (Integers(actual), Integers(expected)) => assert_eq(actual, expected)
        _ => abort("expected integer data")
      }
    }
  }
}

///|
test "unallocated contiguous patch consumes persisted raw free space" {
  let bytes = @fs.read_file_to_bytes("testdata/free_space_fsm.h5")
  let file = Hdf5File::parse(bytes)
  let layout = file.read_dataset_metadata("/data_0").layout
  let header = file.object_header_at(file.resolve("/data_0").address)
  let message = header.first(msg_data_layout).unwrap()
  let output = bytes.to_fixedarray()
  patch_little_endian_uint(
    output,
    layout.data_address_offset,
    layout.data_address_width,
    0xFFFFFFFFFFFFFFFFUL,
    "unallocated test address",
  )
  patch_metadata_checksum_in_place(
    output,
    message.chunk_start,
    message.checksum_offset,
  )
  let unallocated = Hdf5File::parse(output.unsafe_reinterpret_as_bytes())
  let values = Array::make(600, 42)
  let edited = Hdf5File::parse(
    unallocated.patch_dataset_values("/data_0", Int32Values(values)),
  )
  assert_eq(allocation_free_total(edited), allocation_free_total(file) - 2400UL)
  match edited.read_dataset("/data_0").values {
    Integers(actual) => assert_eq(actual, values.map(value => value.to_int64()))
    _ => abort("expected patched integers")
  }
}

///|
test "relocation rejects unsupported strategies sources and alignment" {
  for name in ["page", "transient", "none"] {
    let file = Hdf5File::parse(
      @fs.read_file_to_bytes("testdata/free_space_" + name + ".h5"),
    )
    assert_true(
      try {
        file.relocate_dataset_storage("/data_0") |> ignore
        false
      } catch {
        UnsupportedFeature(_) => true
        _ => false
      },
    )
  }
  let bytes = @fs.read_file_to_bytes("testdata/free_space_fsm.h5")
  assert_true(
    try {
      Hdf5File::parse_source(bytes).relocate_dataset_storage("/data_0")
      |> ignore
      false
    } catch {
      InvalidArgument(_) => true
      _ => false
    },
  )
  assert_true(
    try {
      Hdf5File::parse(bytes).relocate_dataset_storage("/data_0", alignment=3UL)
      |> ignore
      false
    } catch {
      InvalidArgument(_) => true
      _ => false
    },
  )
}

///|
test "consuming every raw section clears its serialized allocation" {
  let mut bytes = @fs.read_file_to_bytes("testdata/free_space_fsm.h5")
  let original = Hdf5File::parse(bytes)
  let address = original.file_space_info().unwrap().manager_addresses[2]
  let mut sizes : Array[Int32] = []
  for manager in original.free_space_managers() {
    if manager.address == address {
      sizes = manager.sections.map(section => section.size.to_int())
    }
  }
  assert_true(sizes.length() > 0)
  for size in sizes {
    let file = Hdf5File::parse(bytes)
    let output = bytes.to_fixedarray()
    let tail = @buffer.Buffer::Buffer()
    let payload = FixedArray::make(size, b'\x2a').unsafe_reinterpret_as_bytes()
    assert_true(
      file.reuse_contiguous_space(payload, output, tail, None, 1UL) is Some(_),
    )
    bytes = file.finish_appended_image(output, tail)
  }
  let file = Hdf5File::parse(bytes)
  for manager in file.free_space_managers() {
    if manager.address == address {
      assert_eq(manager.total_space, 0UL)
      assert_eq(manager.sections.length(), 0)
    }
  }
  let start = address.to_int()
  assert_eq(read_uint_at(bytes, start + 54, 8, true), 0xFFFFFFFFFFFFFFFFUL)
  assert_eq(read_uint_at(bytes, start + 62, 8, true), 0UL)
  assert_eq(read_uint_at(bytes, start + 70, 8, true), 0UL)
}

///|
test "a live dataset overlapping recorded free space is not released" {
  let bytes = @fs.read_file_to_bytes("testdata/free_space_fsm.h5")
  let file = Hdf5File::parse(bytes)
  let raw_address = file.file_space_info().unwrap().manager_addresses[2]
  let mut free : Hdf5FreeSection? = None
  for manager in file.free_space_managers() {
    if manager.address == raw_address {
      free = Some(manager.sections[0])
    }
  }
  let header = file.object_header_at(file.resolve("/data_0").address)
  let layout = file.read_dataset_metadata("/data_0").layout
  let message = header.first(msg_data_layout).unwrap()
  let output = bytes.to_fixedarray()
  patch_little_endian_uint(
    output,
    layout.data_address_offset,
    layout.data_address_width,
    free.unwrap().address,
    "overlap test",
  )
  patch_metadata_checksum_in_place(
    output,
    message.chunk_start,
    message.checksum_offset,
  )
  assert_true(
    try {
      Hdf5File::parse(output.unsafe_reinterpret_as_bytes()).relocate_dataset_storage(
        "/data_0",
      )
      |> ignore
      false
    } catch {
      InvalidField(message) => message.contains("overlaps recorded free space")
      _ => false
    },
  )
}
