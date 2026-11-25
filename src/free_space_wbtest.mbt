///|
test "persistent free-space managers match HDF5 1.14.6 totals" {
  for
    (name, expected, strategy) in [
      ("fsm", 9727UL, FsmAggregate),
      ("page", 21220UL, Paged),
      ("width4", 9817UL, FsmAggregate),
      ("mixed", 21400UL, Paged),
      ("userblock", 21220UL, Paged),
    ] {
    let bytes = @fs.read_file_to_bytes("testdata/free_space_" + name + ".h5")
    for file in [Hdf5File::parse(bytes), Hdf5File::parse_source(bytes)] {
      let info = file.file_space_info().unwrap()
      assert_true(info.strategy == strategy)
      assert_true(info.persistent)
      assert_eq(info.threshold, 1UL)
      assert_eq(info.page_size, 4096UL)
      assert_eq(info.manager_addresses.length(), 12)
      let managers = file.free_space_managers() catch {
        error => abort(name + ": " + error.message())
      }
      assert_true(managers.length() > 0)
      let mut total = 0UL
      for manager in managers {
        assert_eq(manager.ghost_sections, 0UL)
        assert_eq(manager.total_sections, manager.sections.length().to_uint64())
        let mut section_total = 0UL
        for section in manager.sections {
          section_total = section_total + section.size
        }
        assert_eq(section_total, manager.total_space)
        total = total + manager.total_space
      }
      assert_eq(total, expected)
    }
  }
}

///|
test "transient allocation policies expose no persisted free sections" {
  for
    (name, strategy) in [
      ("transient", FsmAggregate),
      ("aggr", Aggregate),
      ("none", Unmanaged),
    ] {
    let bytes = @fs.read_file_to_bytes("testdata/free_space_" + name + ".h5")
    let file = Hdf5File::parse(bytes)
    if file.file_space_info() is Some(info) {
      assert_true(info.strategy == strategy)
      assert_true(!info.persistent)
      assert_eq(info.manager_addresses.length(), 0)
    } else {
      assert_eq(name, "transient")
    }
    assert_eq(file.free_space_managers().length(), 0)
  }
}

///|
test "files without explicit allocation settings expose no managers" {
  let writer = Hdf5Writer::new()
  writer.add_dataset("/values", [2UL], Int32Values([1, 2]))
  let bytes = writer.encode()
  for file in [Hdf5File::parse(bytes), Hdf5File::parse_source(bytes)] {
    assert_true(file.file_space_info() is None)
    assert_eq(file.free_space_managers().length(), 0)
  }
  let legacy = Hdf5File::parse(@fs.read_file_to_bytes("testdata/basic_v0.h5"))
  assert_true(legacy.file_space_info() is None)
}

///|
fn space_test_patch(
  bytes : Bytes,
  start : Int32,
  checksum : Int32,
  field : Int32,
  width : Int32,
  value : UInt64,
) -> Bytes raise Hdf5Error {
  let output = FixedArray::make(bytes.length(), b'\x00')
  output.blit_from_bytes(0, bytes, 0, bytes.length())
  patch_little_endian_uint(output, field, width, value, "free-space test")
  patch_metadata_checksum_in_place(output, start, checksum)
  output.unsafe_reinterpret_as_bytes()
}

///|
test "free-space metadata rejects corrupt checksums and truncated sources" {
  let bytes = @fs.read_file_to_bytes("testdata/free_space_fsm.h5")
  let file = Hdf5File::parse(bytes)
  let address = file.free_space_managers()[0].address.to_int()
  let raw = FixedArray::make(bytes.length(), b'\x00')
  raw.blit_from_bytes(0, bytes, 0, bytes.length())
  raw[address + 6] = raw[address + 6] ^ b'\x01'
  for
    parsed in [
      Hdf5File::parse(raw.unsafe_reinterpret_as_bytes()),
      Hdf5File::parse_source(raw.unsafe_reinterpret_as_bytes()),
    ] {
    assert_true(
      try {
        parsed.free_space_managers() |> ignore
        false
      } catch {
        InvalidField(message) => message.contains("checksum")
        _ => false
      },
    )
  }
  let short = bytes.get_view(start=0, end=address + 10).unwrap().to_owned()
  assert_true(
    try {
      Hdf5File::parse_source(short).free_space_managers() |> ignore
      false
    } catch {
      UnexpectedEof(_) => true
      _ => false
    },
  )
}

///|
test "checksummed manager headers reject invalid dimensions and counts" {
  let bytes = @fs.read_file_to_bytes("testdata/free_space_fsm.h5")
  let file = Hdf5File::parse(bytes)
  let start = file.free_space_managers()[0].address.to_int()
  for
    (field, width, value) in [
      (0, 4, 0UL),
      (14, 8, 0UL),
      (22, 8, 0UL),
      (30, 8, 1UL),
      (38, 2, 0UL),
      (38, 2, 4UL),
      (44, 2, 0UL),
      (44, 2, 65UL),
      (46, 8, 0UL),
      (70, 8, 0UL),
      (70, 8, 0xFFFFFFFFFFFFFFFFUL),
    ] {
    let patched = space_test_patch(
      bytes,
      start,
      start + 78,
      start + field,
      width,
      value,
    )
    assert_true(
      try {
        Hdf5File::parse(patched).free_space_managers() |> ignore
        false
      } catch {
        InvalidField(_) => true
        _ => false
      },
    )
  }
  for (field, value) in [(4, 1UL), (5, 0UL)] {
    let patched = space_test_patch(
      bytes,
      start,
      start + 78,
      start + field,
      1,
      value,
    )
    assert_true(
      try {
        Hdf5File::parse(patched).free_space_managers() |> ignore
        false
      } catch {
        UnsupportedFeature(_) => true
        _ => false
      },
    )
  }
}

///|
test "serialized free sections reject invalid owners classes ranges and overlap" {
  let bytes = @fs.read_file_to_bytes("testdata/free_space_fsm.h5")
  let file = Hdf5File::parse(bytes)
  let address = file.free_space_managers()[0].address.to_int()
  let start = read_uint_at(bytes, address + 54, 8, true).to_int()
  let size = read_uint_at(bytes, address + 62, 8, true).to_int()
  let first = read_uint_at(bytes, start + 22, 8, true)
  for
    (field, width, value) in [
      (0, 4, 0UL),
      (5, 8, 0UL),
      (13, 1, 0UL),
      (13, 1, 255UL),
      (14, 8, 0UL),
      (22, 8, 0UL),
      (22, 8, 0xFFFFFFFFFFFFFFFFUL),
      (30, 1, 3UL),
      (40, 8, first),
    ] {
    let patched = space_test_patch(
      bytes,
      start,
      start + size - 4,
      start + field,
      width,
      value,
    )
    assert_true(
      try {
        Hdf5File::parse(patched).free_space_managers() |> ignore
        false
      } catch {
        InvalidField(_) => true
        _ => false
      },
    )
  }
}

///|
test "file-space settings reject unknown versions and persistence flags" {
  let bytes = @fs.read_file_to_bytes("testdata/free_space_fsm.h5")
  let file = Hdf5File::parse(bytes)
  let message = file
    .object_header_at(file.superblock.superblock_extension_address)
    .first(0x17)
    .unwrap()
  for (field, value) in [(1, 4UL), (2, 2UL), (1, 3UL)] {
    let patched = space_test_patch(
      bytes,
      message.chunk_start,
      message.checksum_offset,
      message.data_offset + field,
      1,
      value,
    )
    assert_true(
      try {
        Hdf5File::parse(patched).file_space_info() |> ignore
        false
      } catch {
        InvalidField(_) => true
        _ => false
      },
    )
  }
  let patched = space_test_patch(
    bytes,
    message.chunk_start,
    message.checksum_offset,
    message.data_offset,
    1,
    2UL,
  )
  assert_true(
    try {
      Hdf5File::parse(patched).file_space_info() |> ignore
      false
    } catch {
      UnsupportedFeature(_) => true
      _ => false
    },
  )
}

///|
test "legacy file-space settings map strategies and six manager slots" {
  let bytes = @fs.read_file_to_bytes("testdata/free_space_fsm.h5")
  let file = Hdf5File::parse(bytes)
  let message = file
    .object_header_at(file.superblock.superblock_extension_address)
    .first(0x17)
    .unwrap()
  let addresses = file.file_space_info().unwrap().manager_addresses
  for
    (encoded, strategy, persistent) in [
      (1UL, FsmAggregate, true),
      (2UL, FsmAggregate, false),
      (3UL, Aggregate, false),
      (4UL, Unmanaged, false),
    ] {
    let output = FixedArray::make(bytes.length(), b'\x00')
    output.blit_from_bytes(0, bytes, 0, bytes.length())
    output[message.data_offset] = b'\x00'
    output[message.data_offset + 1] = encoded.to_byte()
    patch_little_endian_uint(
      output,
      message.data_offset + 2,
      8,
      1UL,
      "threshold",
    )
    for index in 0..<6 {
      patch_little_endian_uint(
        output,
        message.data_offset + 10 + index * 8,
        8,
        addresses[index],
        "legacy manager address",
      )
    }
    patch_metadata_checksum_in_place(
      output,
      message.chunk_start,
      message.checksum_offset,
    )
    let info = Hdf5File::parse(output.unsafe_reinterpret_as_bytes())
      .file_space_info()
      .unwrap()
    assert_true(info.strategy == strategy)
    assert_eq(info.persistent, persistent)
    assert_eq(info.threshold, 1UL)
    assert_eq(info.page_size, 4096UL)
    assert_eq(info.manager_addresses.length(), if persistent { 6 } else { 0 })
  }
}

///|
test "source free-space inspection reads metadata rather than the whole file" {
  let source = CountingSource::{
    bytes: @fs.read_file_to_bytes("testdata/free_space_page.h5"),
    reads: 0,
    requested: 0UL,
    largest_read: 0,
  }
  let file = Hdf5File::parse_source(source)
  let managers = file.free_space_managers()
  assert_true(managers.length() > 0)
  assert_true(source.largest_read < 1024)
  assert_true(source.requested < source.bytes.length().to_uint64())
}

///|
test "free sections cannot overlap another manager's header or section allocation" {
  let bytes = @fs.read_file_to_bytes("testdata/free_space_fsm.h5")
  let file = Hdf5File::parse(bytes)
  let managers = file.free_space_managers()
  assert_true(managers.length() >= 2)
  let first = managers[0].address.to_int()
  let second = managers[1].address
  let start = read_uint_at(bytes, first + 54, 8, true).to_int()
  let size = read_uint_at(bytes, first + 62, 8, true).to_int()
  let other_sections = read_uint_at(bytes, second.to_int() + 54, 8, true)
  for address in [second, other_sections] {
    let patched = space_test_patch(
      bytes,
      start,
      start + size - 4,
      start + 22,
      8,
      address,
    )
    for parsed in [Hdf5File::parse(patched), Hdf5File::parse_source(patched)] {
      assert_true(
        try {
          parsed.free_space_managers() |> ignore
          false
        } catch {
          InvalidField(message) => message.contains("manager metadata")
          _ => false
        },
      )
    }
  }
}
