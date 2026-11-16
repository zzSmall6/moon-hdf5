///|
test "groups can be created below existing groups" {
  let writer = Hdf5Writer::new()
  writer.add_group("/existing")
  let file = Hdf5File::parse(writer.encode())

  let created = file.add_group("/", "created") catch {
    error => fail("create group: " + error.message())
  }
  let created_file = Hdf5File::parse(created)
  let node = created_file.resolve("/created")
  assert_eq(node.kind, Group)
  let header = created_file.object_header_at(node.address)
  assert_true(header.first(msg_link_info) is Some(_))
  assert_eq(created_file.links(header).length(), 0)

  let nested = created_file.add_group("/created", "nested") catch {
    error => fail("create nested group: " + error.message())
  }
  let nested_file = Hdf5File::parse(nested)
  assert_eq(nested_file.resolve("/created/nested").kind, Group)
  assert_eq(nested_file.walk().map(item => item.path), [
    "/existing", "/created", "/created/nested",
  ])
}

///|
test "object creation preserves an existing user block" {
  let original = @fs.read_file_to_bytes("testdata/userblock.h5")
  let file = Hdf5File::parse(original)
  let created = file.add_group("/", "created") catch {
    error => fail("create group behind user block: " + error.message())
  }
  assert_true(created.length() > original.length())
  let edited = Hdf5File::parse(created)
  assert_eq(edited.superblock._superblock_offset, 512)
  assert_eq(edited.superblock.base_address, 512UL)
  assert_eq(edited.superblock.end_of_file_address, created.length().to_uint64())
  assert_eq(edited.resolve("/created").kind, Group)
  assert_eq(integer_values(edited.read_dataset("/x")), [0L, 1L, 2L, 3L])
  let patched = edited.patch_dataset_values("/x", Int32Values([10, 20, 30, 40])) catch {
    error => fail("patch dataset behind user block: " + error.message())
  }
  let patched_file = Hdf5File::parse(patched)
  assert_eq(
    patched_file.superblock.end_of_file_address,
    patched.length().to_uint64(),
  )
  assert_eq(integer_values(patched_file.read_dataset("/x")), [
    10L, 20L, 30L, 40L,
  ])
}

///|
test "group creation converts symbol-table parents and honors four-byte addresses" {
  let file = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/symbol_table_32.h5"),
  )
  assert_eq(file.superblock.offset_size, 4)
  let created = file.add_group("/group", "created") catch {
    error => fail("create group in symbol-table parent: " + error.message())
  }
  let edited = Hdf5File::parse(created)
  assert_eq(edited.superblock.offset_size, 4)
  let parent = edited.resolve("/group")
  let header = edited.object_header_at(parent.address)
  assert_true(header.first(msg_symbol_table) is None)
  assert_true(header.first(msg_link_info) is Some(_))
  let node = edited.resolve("/group/created")
  assert_eq(node.kind, Group)
  assert_eq(edited.links(edited.object_header_at(node.address)).length(), 0)
}

///|
test "group creation rejects duplicate and non-group parents" {
  let writer = Hdf5Writer::new()
  writer.add_group("/group")
  writer.add_dataset("/dataset", [], UInt64Values([1UL]))
  let file = Hdf5File::parse(writer.encode())
  let duplicate = try {
    let created = file.add_group("/group", "item")
    ignore(Hdf5File::parse(created).add_group("/group", "item"))
    None
  } catch {
    error => Some(error)
  }
  assert_eq(duplicate, Some(InvalidArgument("link name already exists: item")))
  let invalid_parent = try {
    ignore(file.add_group("/dataset", "child"))
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    invalid_parent,
    Some(InvalidArgument("link parent is not a group: /dataset")),
  )
}

///|
test "fixed-size datasets can be created and read" {
  let writer = Hdf5Writer::new()
  writer.add_group("/existing")
  let file = Hdf5File::parse(writer.encode())

  let created = file.add_dataset(
    "/existing",
    "matrix",
    [2UL, 3UL],
    Int32Values([1, 2, 3, 4, 5, 6]),
    fill_value=Int32Values([-1]),
  ) catch {
    error => fail("create dataset: " + error.message())
  }
  let created_file = Hdf5File::parse(created)
  let node = created_file.resolve("/existing/matrix")
  assert_eq(node.kind, Dataset)
  let metadata = created_file.read_dataset_metadata("/existing/matrix")
  assert_eq(metadata.dataspace.dimensions, [2UL, 3UL])
  assert_eq(metadata.layout.kind(), Contiguous)
  let header = created_file.object_header_at(node.address)
  assert_true(header.first(msg_fill_value) is Some(_))
  match created_file.read_dataset("/existing/matrix").values {
    Integers(values) => assert_eq(values, [1L, 2L, 3L, 4L, 5L, 6L])
    _ => fail("expected integer dataset values")
  }

  let scalar = created_file.add_dataset(
    "/existing",
    "scalar",
    [],
    Float64Values([2.5]),
  ) catch {
    error => fail("create scalar dataset: " + error.message())
  }
  let scalar_file = Hdf5File::parse(scalar)
  assert_eq(
    scalar_file.read_dataset_metadata("/existing/scalar").layout.kind(),
    Contiguous,
  )
  match scalar_file.read_dataset("/existing/scalar").values {
    Floats(values) => assert_eq(values, [2.5])
    _ => fail("expected floating-point scalar values")
  }

  let empty = scalar_file.add_dataset(
    "/existing",
    "empty",
    [0UL],
    UInt64Values([]),
  ) catch {
    error => fail("create empty dataset: " + error.message())
  }
  let empty_file = Hdf5File::parse(empty)
  let empty_metadata = empty_file.read_dataset_metadata("/existing/empty")
  assert_eq(empty_metadata.dataspace.dimensions, [0UL])
  assert_eq(empty_metadata.layout.kind(), Contiguous)
  match empty_file.read_dataset("/existing/empty").values {
    UnsignedIntegers(values) => assert_eq(values, [])
    _ => fail("expected empty unsigned integer values")
  }
}

///|
test "dataset creation works with symbol-table parents and four-byte addresses" {
  let file = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/symbol_table_32.h5"),
  )
  let created = file.add_dataset(
    "/group",
    "created_values",
    [2UL, 2UL],
    Int64Values([7L, 8L, 9L, 10L]),
  ) catch {
    error => fail("create 4-byte dataset: " + error.message())
  }
  let edited = Hdf5File::parse(created)
  let metadata = edited.read_dataset_metadata("/group/created_values")
  assert_eq(metadata.dataspace.dimensions, [2UL, 2UL])
  assert_eq(metadata.layout.kind(), Contiguous)
  match edited.read_dataset("/group/created_values").values {
    Integers(values) => assert_eq(values, [7L, 8L, 9L, 10L])
    _ => fail("expected 4-byte integer dataset values")
  }
  let header = edited.object_header_at(edited.resolve("/group").address)
  assert_true(header.first(msg_symbol_table) is None)
  assert_true(header.first(msg_link_info) is Some(_))
  assert_eq(
    edited.attributes("/group/created_values").map(attribute => attribute.name),
    [],
  )

  let empty = edited.add_dataset(
    "/group",
    "empty_values",
    [0UL],
    UInt64Values([]),
  ) catch {
    error => fail("create empty 4-byte dataset: " + error.message())
  }
  let empty_file = Hdf5File::parse(empty)
  let empty_metadata = empty_file.read_dataset_metadata("/group/empty_values")
  assert_eq(empty_metadata.dataspace.dimensions, [0UL])
  assert_eq(empty_metadata.layout.kind(), Contiguous)
  match empty_file.read_dataset("/group/empty_values").values {
    UnsignedIntegers(values) => assert_eq(values, [])
    _ => fail("expected empty 4-byte unsigned integer values")
  }
}

///|
test "compact datasets preserve values references and variable-length data" {
  for
    (fixture, reference_size) in [
      ("testdata/symbol_table_32.h5", 4),
      ("testdata/references.h5", 8),
    ] {
    let file = Hdf5File::parse(@fs.read_file_to_bytes(fixture))
    let values = file.add_compact_dataset(
      "/group",
      "compact_values",
      [2UL, 3UL],
      Int32Values([1, 2, 3, 4, 5, 6]),
    ) catch {
      error => fail("create compact dataset: " + error.message())
    }
    let labels = Hdf5File::parse(values).add_compact_dataset(
      "/group",
      "compact_labels",
      [2UL],
      VariableStringValues(["moon", "数据"]),
    ) catch {
      error => fail("create compact string dataset: " + error.message())
    }
    let references = Hdf5File::parse(labels).add_compact_dataset(
      "/group",
      "compact_references",
      [2UL],
      ObjectReferenceValues([Some("/matrix"), None]),
    ) catch {
      error => fail("create compact reference dataset: " + error.message())
    }
    let edited = Hdf5File::parse(references)
    let metadata = edited.read_dataset_metadata("/group/compact_values")
    assert_eq(metadata.dataspace.dimensions, [2UL, 3UL])
    assert_eq(metadata.layout.kind(), Compact)
    match edited.read_dataset("/group/compact_values").values {
      Integers(items) => assert_eq(items, [1L, 2L, 3L, 4L, 5L, 6L])
      _ => fail("expected compact integer values")
    }
    match edited.read_dataset("/group/compact_labels").values {
      Strings(items) => assert_eq(items, ["moon", "数据"])
      _ => fail("expected compact variable-length strings")
    }
    let refs = match edited.read_dataset("/group/compact_references").values {
      Structured(items) => items
      _ => fail("expected compact object references")
    }
    let target = match refs[0] {
      Reference(bytes) => bytes
      _ => fail("expected a compact object reference")
    }
    assert_eq(target.length(), reference_size)
    assert_eq(
      edited.resolve_object_reference(target).unwrap().path,
      Some("/matrix"),
    )
    assert_eq(refs[1], Reference(Bytes::make(reference_size, b'\x00')))
    let attributed = edited.add_attribute(
      "/group/compact_values",
      "unit",
      [],
      FixedStringValues(["count"], 8),
    ) catch {
      error => fail("attribute a compact dataset: " + error.message())
    }
    assert_eq(
      Hdf5File::parse(attributed).attributes("/group/compact_values")[0].name,
      "unit",
    )
  }
}

///|
test "compact dataset creation stores fills in four- and eight-byte files" {
  for fixture in ["testdata/symbol_table_32.h5", "testdata/references.h5"] {
    let file = Hdf5File::parse(@fs.read_file_to_bytes(fixture))
    let created = file.add_compact_dataset(
      "/group",
      "compact_filled",
      [2UL],
      Int32Values([1, 2]),
      fill_value=Int32Values([-9]),
    ) catch {
      error =>
        fail(fixture + ": create filled compact dataset: " + error.message())
    }
    let edited = Hdf5File::parse(created)
    let metadata = edited.read_dataset_metadata("/group/compact_filled")
    assert_eq(metadata.layout.kind(), Compact)
    let header = edited.object_header_at(
      edited.resolve("/group/compact_filled").address,
    )
    assert_eq(
      dataset_fill_value(header, metadata.datatype),
      b"\xf7\xff\xff\xff",
    )
    match edited.read_dataset("/group/compact_filled").values {
      Integers(values) => assert_eq(values, [1L, 2L])
      _ => fail(fixture + ": expected compact filled values")
    }

    let reference_fill = edited.add_compact_dataset(
      "/group",
      "compact_reference_fill",
      [1UL],
      ObjectReferenceValues([None]),
      fill_value=ObjectReferenceValues([Some("/matrix")]),
    ) catch {
      error =>
        fail(fixture + ": create compact reference fill: " + error.message())
    }
    let reference_file = Hdf5File::parse(reference_fill)
    let reference_metadata = reference_file.read_dataset_metadata(
      "/group/compact_reference_fill",
    )
    let reference_header = reference_file.object_header_at(
      reference_file.resolve("/group/compact_reference_fill").address,
    )
    let fill = dataset_fill_value(reference_header, reference_metadata.datatype)
    assert_eq(fill.length(), reference_file.superblock.offset_size)
    assert_eq(
      reference_file.resolve_object_reference(fill).map(node => node.path),
      Some(Some("/matrix")),
    )
  }
}

///|
test "compact dataset creation validates storage and duplicate names" {
  let writer = Hdf5Writer::new()
  writer.add_group("/group")
  let file = Hdf5File::parse(writer.encode())
  let boundary = file.add_compact_dataset(
    "/group",
    "boundary",
    [65524UL],
    UInt8Values(Array::make(65524, 1U)),
  ) catch {
    error => fail("create boundary compact dataset: " + error.message())
  }
  let boundary_file = Hdf5File::parse(boundary)
  assert_eq(
    boundary_file.read_dataset_metadata("/group/boundary").layout.kind(),
    Compact,
  )
  match boundary_file.read_dataset("/group/boundary").values {
    UnsignedIntegers(items) => {
      assert_eq(items.length(), 65524)
      assert_eq(items[0], 1UL)
      assert_eq(items[65523], 1UL)
    }
    _ => fail("expected boundary compact integer values")
  }
  let too_large = try {
    ignore(
      file.add_compact_dataset(
        "/group",
        "large",
        [65525UL],
        UInt8Values(Array::make(65525, 0U)),
      ),
    )
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    too_large,
    Some(InvalidArgument("compact dataset values exceed 65524 bytes")),
  )
  let created = file.add_compact_dataset(
    "/group",
    "created",
    [2UL],
    UInt16Values([1U, 2U]),
  ) catch {
    error => fail("create compact dataset: " + error.message())
  }
  let duplicate = try {
    ignore(
      Hdf5File::parse(created).add_compact_dataset(
        "/group",
        "created",
        [1UL],
        UInt16Values([3U]),
      ),
    )
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    duplicate,
    Some(InvalidArgument("link name already exists: created")),
  )
}

///|
test "compact dataset boundary works in four- and eight-byte files" {
  for fixture in ["testdata/symbol_table_32.h5", "testdata/references.h5"] {
    let file = Hdf5File::parse(@fs.read_file_to_bytes(fixture))
    let created = file.add_compact_dataset(
      "/group",
      "boundary",
      [65524UL],
      UInt8Values(Array::make(65524, 1U)),
    ) catch {
      error =>
        fail(fixture + ": create boundary compact dataset: " + error.message())
    }
    let edited = Hdf5File::parse(created)
    assert_eq(
      edited.read_dataset_metadata("/group/boundary").layout.kind(),
      Compact,
    )
    match edited.read_dataset("/group/boundary").values {
      UnsignedIntegers(items) => assert_eq(items.length(), 65524)
      _ => fail(fixture + ": expected boundary compact integer values")
    }
  }
}

///|
test "committed datatypes can be added and edited in existing files" {
  let writer = Hdf5Writer::new()
  writer.add_group("/types")
  let file = Hdf5File::parse(writer.encode())
  let record = Datatype::write_compound([
    ("id", Datatype::write_integer(false, 4)),
    ("name", Datatype::write_fixed_string(8)),
    ("target", Datatype::write_object_reference()),
  ])
  let created = file.add_committed_datatype("/types", "record", record) catch {
    error => fail("create committed datatype: " + error.message())
  }
  let edited = Hdf5File::parse(created)
  let node = edited.resolve("/types/record")
  assert_eq(node.kind, Datatype)
  let header = edited.object_header_at(node.address)
  let message = match header.first(msg_datatype) {
    Some(message) => message
    None => fail("committed datatype has no datatype message")
  }
  let parsed = parse_datatype(message)
  assert_eq(parsed.class, record.class)
  assert_eq(parsed.size, record.size)
  assert_eq(
    edited.attributes("/types/record").map(attribute => attribute.name),
    [],
  )
  let attributed = edited.add_attribute(
    "/types/record",
    "category",
    [],
    FixedStringValues(["record"], 8),
  ) catch {
    error => fail("attribute a committed datatype: " + error.message())
  }
  let linked = Hdf5File::parse(attributed).add_hard_link(
    "/types", "record_alias", "/types/record",
  ) catch {
    error => fail("link a committed datatype: " + error.message())
  }
  let final_file = Hdf5File::parse(linked)
  assert_eq(final_file.resolve("/types/record_alias").kind, Datatype)
  assert_eq(
    final_file
    .attributes("/types/record_alias")
    .map(attribute => attribute.name),
    ["category"],
  )
}

///|
test "committed datatypes use the target file address width" {
  let file = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/symbol_table_32.h5"),
  )
  let datatype = Datatype::write_compound([
    ("name", Datatype::write_vlen_string_with_address_size(4)),
    ("target", Datatype::write_object_reference_with_address_size(4)),
  ])
  let created = file.add_committed_datatype("/group", "wide_record", datatype) catch {
    error => fail("create 4-byte committed datatype: " + error.message())
  }
  let edited = Hdf5File::parse(created)
  let node = edited.resolve("/group/wide_record")
  assert_eq(node.kind, Datatype)
  let header = edited.object_header_at(node.address)
  let message = match header.first(msg_datatype) {
    Some(message) => message
    None => fail("committed datatype has no datatype message")
  }
  assert_eq(
    message.data,
    encode_writer_datatype(datatype, reference_address_size=4),
  )
  let duplicate = try {
    ignore(edited.add_committed_datatype("/group", "wide_record", datatype))
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    duplicate,
    Some(InvalidArgument("link name already exists: wide_record")),
  )
}

///|
test "created datasets can receive attributes and links" {
  let writer = Hdf5Writer::new()
  writer.add_group("/targets")
  let file = Hdf5File::parse(writer.encode())
  let created = file.add_dataset(
    "/targets",
    "data",
    [3UL],
    UInt16Values([10U, 20U, 30U]),
  ) catch {
    error => fail("create attributed dataset: " + error.message())
  }
  let edited = Hdf5File::parse(created).add_attribute(
    "/targets/data",
    "unit",
    [],
    FixedStringValues(["count"], 8),
  ) catch {
    error => fail("add dataset attribute: " + error.message())
  }
  let linked = Hdf5File::parse(edited).add_hard_link(
    "/targets", "alias", "/targets/data",
  ) catch {
    error => fail("link created dataset: " + error.message())
  }
  let final_file = Hdf5File::parse(linked)
  match final_file.read_dataset("/targets/alias").values {
    UnsignedIntegers(values) => assert_eq(values, [10UL, 20UL, 30UL])
    _ => fail("expected unsigned integer values")
  }
  match final_file.attributes("/targets/data")[0].values {
    Some(Strings(values)) => assert_eq(values, ["count"])
    _ => fail("expected dataset attribute to survive edits")
  }
}

///|
test "variable-length and object-reference datasets can be created" {
  let cases : Array[(String, String, Int32)] = [
    ("testdata/symbol_table_32.h5", "/group/values", 4),
    ("testdata/references.h5", "/group/child", 8),
  ]
  for (fixture, second_target, reference_size) in cases {
    let file = Hdf5File::parse(@fs.read_file_to_bytes(fixture))
    let strings = file.add_dataset(
      "/group",
      "labels",
      [4UL],
      VariableStringValues(["moon", "数据", "", "moon"]),
    ) catch {
      error => fail("create variable-length strings: " + error.message())
    }
    let references = Hdf5File::parse(strings).add_dataset(
      "/group",
      "references",
      [3UL],
      ObjectReferenceValues([Some("/matrix"), None, Some(second_target)]),
    ) catch {
      error => fail("create object references: " + error.message())
    }
    let edited = Hdf5File::parse(references)
    let labels = edited.read_dataset("/group/labels")
    assert_eq(labels.datatype.class, 9)
    assert_eq(labels.datatype.size, 4 + reference_size + 4)
    match labels.values {
      Strings(values) => assert_eq(values, ["moon", "数据", "", "moon"])
      _ => fail("expected variable-length strings")
    }
    let refs = edited.read_dataset("/group/references")
    assert_eq(refs.datatype.class, 7)
    assert_eq(refs.datatype.size, reference_size)
    let values = match refs.values {
      Structured(values) => values
      _ => fail("expected structured object references")
    }
    let matrix_bytes = match values[0] {
      Reference(bytes) => bytes
      _ => fail("expected object-reference bytes")
    }
    assert_eq(
      edited.resolve_object_reference(matrix_bytes).unwrap().path,
      Some("/matrix"),
    )
    assert_eq(values[1], Reference(Bytes::make(reference_size, b'\x00')))
    let target_bytes = match values[2] {
      Reference(bytes) => bytes
      _ => fail("expected object-reference bytes")
    }
    assert_eq(
      edited.resolve_object_reference(target_bytes).unwrap().address,
      edited.resolve(second_target).address,
    )
  }
}

///|
test "dataset creation rejects invalid references and shapes" {
  let writer = Hdf5Writer::new()
  let file = Hdf5File::parse(writer.encode())
  let missing = try {
    ignore(
      file.add_dataset(
        "/",
        "refs",
        [1UL],
        ObjectReferenceValues([Some("/missing")]),
      ),
    )
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    missing,
    Some(InvalidArgument("object reference target does not exist: /missing")),
  )
  let mismatched = try {
    ignore(file.add_dataset("/", "bad", [2UL], UInt64Values([1UL])))
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    mismatched,
    Some(InvalidArgument("writer value count does not match dataspace")),
  )
}

///|
test "chunked and filtered datasets can be created with file-width indexes" {
  let cases : Array[(String, String, Int32)] = [
    ("testdata/symbol_table_32.h5", "/group/values", 4),
    ("testdata/references.h5", "/group/child", 8),
  ]
  for (fixture, second_target, reference_size) in cases {
    let file = Hdf5File::parse(@fs.read_file_to_bytes(fixture))
    let chunked = file.add_chunked_dataset(
      "/group",
      "chunked",
      [4UL, 3UL],
      [2U, 3U],
      UInt32Values([0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U, 10U, 11U]),
      fill_value=UInt32Values([99U]),
      filters=[Shuffle, Deflate(6)],
    ) catch {
      error => fail("create chunked dataset: " + error.message())
    }
    let names = Hdf5File::parse(chunked).add_chunked_dataset(
      "/group",
      "chunked_names",
      [4UL],
      [2U],
      VariableStringValues(["moon", "数据", "", "moon"]),
      filters=[Deflate(3)],
    ) catch {
      error => fail("create chunked string dataset: " + error.message())
    }
    let references = Hdf5File::parse(names).add_chunked_dataset(
      "/group",
      "chunked_references",
      [3UL],
      [2U],
      ObjectReferenceValues([Some("/matrix"), None, Some(second_target)]),
    ) catch {
      error => fail("create chunked reference dataset: " + error.message())
    }
    let edited = Hdf5File::parse(references)
    let metadata = edited.read_dataset_metadata("/group/chunked")
    assert_eq(metadata.dataspace.dimensions, [4UL, 3UL])
    assert_eq(metadata.layout.kind(), Chunked)
    assert_eq(metadata.layout.chunk_dimensions, [2U, 3U])
    let header = edited.object_header_at(
      edited.resolve("/group/chunked").address,
    )
    assert_true(header.first(msg_filter_pipeline) is Some(_))
    match edited.read_dataset("/group/chunked").values {
      UnsignedIntegers(values) =>
        assert_eq(values, [
          0UL, 1UL, 2UL, 3UL, 4UL, 5UL, 6UL, 7UL, 8UL, 9UL, 10UL, 11UL,
        ])
      _ => fail("expected chunked integer values")
    }
    match edited.read_dataset("/group/chunked_names").values {
      Strings(values) => assert_eq(values, ["moon", "数据", "", "moon"])
      _ => fail("expected chunked variable-length strings")
    }
    let refs = match edited.read_dataset("/group/chunked_references").values {
      Structured(values) => values
      _ => fail("expected chunked object references")
    }
    let matrix_bytes = match refs[0] {
      Reference(bytes) => bytes
      _ => fail("expected object-reference bytes")
    }
    assert_eq(matrix_bytes.length(), reference_size)
    assert_eq(
      edited.resolve_object_reference(matrix_bytes).unwrap().path,
      Some("/matrix"),
    )
    assert_eq(refs[1], Reference(Bytes::make(reference_size, b'\x00')))
  }
}

///|
test "external raw-data datasets can be created with file-width metadata" {
  let cases : Array[(String, Int32)] = [
    ("testdata/symbol_table_32.h5", 4),
    ("testdata/references.h5", 8),
  ]
  for (fixture, offset_size) in cases {
    let file = Hdf5File::parse(@fs.read_file_to_bytes(fixture))
    let external = file.add_external_dataset(
      "/group",
      "external",
      [6UL],
      Int32Values([10, 20, 30, 40, 50, 60]),
      [
        {
          file_name: "created_external.bin",
          file_offset: 4UL,
          data_size: 24UL,
        },
      ],
      fill_value=Int32Values([-1]),
    ) catch {
      error => fail("create external dataset: " + error.message())
    }
    let edited = Hdf5File::parse(external)
    assert_eq(edited.superblock.offset_size, offset_size)
    let metadata = edited.read_dataset_metadata("/group/external")
    assert_eq(metadata.layout.kind(), Contiguous)
    assert_eq(metadata.layout.data_size, 24UL)
    let source = @buffer.Buffer::Buffer()
    writer_write_zeros(source, 4)
    for value in [10U, 20U, 30U, 40U, 50U, 60U] {
      source.write_uint_le(value)
    }
    let resolver = Hdf5Resolver::new(external)
    resolver.add_source("created_external.bin", source.to_bytes())
    match resolver.read_dataset("/group/external").values {
      Integers(values) => assert_eq(values, [10L, 20L, 30L, 40L, 50L, 60L])
      _ => fail("expected external integer values")
    }
    match
      resolver.read_dataset_selection("/group/external", Points([[5UL], [1UL]])).values {
      Integers(values) => assert_eq(values, [60L, 20L])
      _ => fail("expected external point selection")
    }
  }
}

///|
fn external_source_with_prefix(payload : Bytes, offset : Int32) -> Bytes {
  let source = @buffer.Buffer::Buffer()
  writer_write_zeros(source, offset)
  writer_write_bytes(source, payload)
  source.to_bytes()
}

///|
test "external raw-data datasets store fixed-width object references" {
  let cases : Array[(String, Int32, String)] = [
    ("testdata/symbol_table_32.h5", 4, "/group/values"),
    ("testdata/references.h5", 8, "/group/child"),
  ]
  for (fixture, reference_size, second_target) in cases {
    let file = Hdf5File::parse(@fs.read_file_to_bytes(fixture))
    let values = ObjectReferenceValues([
      Some("/matrix"),
      None,
      Some(second_target),
    ])
    let created = file.add_external_dataset(
      "/group",
      "external_references",
      [3UL],
      values,
      [
        {
          file_name: "created_external_references.bin",
          file_offset: 4UL,
          data_size: (3 * reference_size).to_uint64(),
        },
      ],
      fill_value=ObjectReferenceValues([Some("/matrix")]),
    ) catch {
      error =>
        fail(fixture + ": create external references: " + error.message())
    }
    let edited = Hdf5File::parse(created)
    let metadata = edited.read_dataset_metadata("/group/external_references")
    assert_eq(metadata.datatype.class, 7)
    assert_eq(metadata.datatype.size, reference_size)
    let fill = dataset_fill_value(
      edited.object_header_at(
        edited.resolve("/group/external_references").address,
      ),
      metadata.datatype,
    )
    assert_eq(
      edited.resolve_object_reference(fill).map(target => target.path),
      Some(Some("/matrix")),
    )

    let objects = edited.created_reference_objects()
    let payload = encode_write_values(
        values,
        objects~,
        reference_address_size=reference_size,
      ).payload
    let resolver = Hdf5Resolver::new(created)
    resolver.add_source(
      "created_external_references.bin",
      external_source_with_prefix(payload, 4),
    )
    let references = match
      resolver.read_dataset("/group/external_references").values {
      Structured(items) => items
      _ => fail(fixture + ": expected external object references")
    }
    let matrix = match references[0] {
      Reference(bytes) => bytes
      _ => fail(fixture + ": expected a matrix reference")
    }
    assert_eq(matrix.length(), reference_size)
    assert_eq(
      resolver.resolve_object_reference(matrix).unwrap().target.path,
      Some("/matrix"),
    )
    assert_eq(references[1], Reference(Bytes::make(reference_size, b'\x00')))
    let second = match references[2] {
      Reference(bytes) => bytes
      _ => fail(fixture + ": expected a second-target reference")
    }
    assert_eq(
      resolver.resolve_object_reference(second).unwrap().target.address,
      edited.resolve(second_target).address,
    )
  }
}

///|
test "external raw-data datasets store nested fixed-width object references" {
  let cases : Array[(String, Int32, String)] = [
    ("testdata/symbol_table_32.h5", 4, "/group/values"),
    ("testdata/references.h5", 8, "/group/child"),
  ]
  for (fixture, reference_size, second_target) in cases {
    let object_type = Datatype::write_object_reference_with_address_size(
      reference_size,
    )
    let record_type = Datatype::write_compound([
      ("id", Datatype::write_integer(false, 4)),
      ("target", object_type),
      ("pair", Datatype::write_array(object_type, [2UL])),
    ])
    let first = Compound([
      { name: "id", value: UnsignedInteger(1UL), },
      { name: "target", value: ObjectReference(Some("/matrix")), },
      {
        name: "pair",
        value: Elements([
          ObjectReference(Some("/matrix")),
          ObjectReference(Some(second_target)),
        ]),
      },
    ])
    let second = Compound([
      { name: "id", value: UnsignedInteger(2UL), },
      { name: "target", value: ObjectReference(None), },
      {
        name: "pair",
        value: Elements([ObjectReference(None), ObjectReference(None)]),
      },
    ])
    let values = StructuredValues(record_type, [first, second])
    let file = Hdf5File::parse(@fs.read_file_to_bytes(fixture))
    let fill = if reference_size == 8 {
      Some(StructuredValues(record_type, [first]))
    } else {
      let rejected = try {
        ignore(
          file.add_external_dataset(
            "/group",
            "external_record",
            [2UL],
            values,
            [{ file_name: "records.bin", file_offset: 0UL, data_size: 32UL, }],
            fill_value=StructuredValues(record_type, [first]),
          ),
        )
        None
      } catch {
        error => Some(error)
      }
      assert_eq(
        rejected,
        Some(
          UnsupportedFeature(
            "custom fills for structured reference values require eight-byte file addresses",
          ),
        ),
      )
      None
    }
    let created = file.add_external_dataset(
      "/group",
      "external_records",
      [2UL],
      values,
      [
        {
          file_name: "created_external_records.bin",
          file_offset: 4UL,
          data_size: (2 * record_type.size).to_uint64(),
        },
      ],
      fill_value?=fill,
    ) catch {
      error => fail(fixture + ": create external records: " + error.message())
    }
    let edited = Hdf5File::parse(created)
    let objects = edited.created_reference_objects()
    let payload = encode_write_values(
        values,
        objects~,
        reference_address_size=reference_size,
      ).payload
    let resolver = Hdf5Resolver::new(created)
    resolver.add_source(
      "created_external_records.bin",
      external_source_with_prefix(payload, 4),
    )
    let records = match
      resolver.read_dataset("/group/external_records").values {
      Structured(items) => items
      _ => fail(fixture + ": expected external compound records")
    }
    let first_fields = match records[0] {
      Compound(fields) => fields
      _ => fail(fixture + ": expected a compound record")
    }
    let target = match first_fields[1].value {
      Reference(bytes) => bytes
      _ => fail(fixture + ": expected a nested target reference")
    }
    assert_eq(
      resolver.resolve_object_reference(target).unwrap().target.path,
      Some("/matrix"),
    )
    let pair = match first_fields[2].value {
      Elements(items) => items
      _ => fail(fixture + ": expected a nested reference pair")
    }
    let second_pair = match pair[1] {
      Reference(bytes) => bytes
      _ => fail(fixture + ": expected a nested second-target reference")
    }
    assert_eq(
      resolver.resolve_object_reference(second_pair).unwrap().target.address,
      edited.resolve(second_target).address,
    )
    if reference_size == 8 {
      let metadata = edited.read_dataset_metadata("/group/external_records")
      let fill_bytes = dataset_fill_value(
        edited.object_header_at(
          edited.resolve("/group/external_records").address,
        ),
        metadata.datatype,
      )
      let target_member = record_type.compound_members[1]
      let fill_target = Bytes::makei(reference_size, index => {
        fill_bytes[target_member.offset + index]
      })
      assert_eq(
        edited.resolve_object_reference(fill_target).map(target => target.path),
        Some(Some("/matrix")),
      )
    }
  }
}

///|
test "external raw-data datasets reject indirect references" {
  let file = Hdf5File::parse(Hdf5Writer::new().encode())
  let values = StructuredValues(Datatype::write_region_reference(), [
    RegionReference(None, RegionEmpty),
  ])
  let rejected = try {
    ignore(
      file.add_external_dataset("/", "external_regions", [1UL], values, [
        { file_name: "regions.bin", file_offset: 0UL, data_size: 12UL, },
      ]),
    )
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    rejected,
    Some(
      UnsupportedFeature(
        "external storage supports only fixed-width legacy object references",
      ),
    ),
  )
}

///|
test "chunked and external creation validate filters and slots" {
  let file = Hdf5File::parse(Hdf5Writer::new().encode())
  let bad_chunk = try {
    ignore(
      file.add_chunked_dataset("/", "bad", [2UL], [0U], UInt32Values([1U, 2U])),
    )
    None
  } catch {
    error => Some(error)
  }
  assert_eq(bad_chunk, Some(InvalidArgument("chunk dimension is invalid")))
  let bad_filter = try {
    ignore(
      file.add_chunked_dataset(
        "/",
        "bad",
        [2UL],
        [2U],
        UInt32Values([1U, 2U]),
        filters=[Deflate(10)],
      ),
    )
    None
  } catch {
    error => Some(error)
  }
  assert_eq(bad_filter, Some(InvalidArgument("invalid deflate writer filter")))
  let bad_slots = try {
    ignore(
      file.add_external_dataset("/", "bad", [2UL], Int32Values([1, 2]), [
        { file_name: "data.bin", file_offset: 0UL, data_size: 4UL, },
      ]),
    )
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    bad_slots,
    Some(InvalidArgument("external data slots do not match the dataset size")),
  )
}

///|
test "resizable datasets can be created with maximum dimensions" {
  let cases : Array[(String, Int32)] = [
    ("testdata/symbol_table_32.h5", 4),
    ("testdata/references.h5", 8),
  ]
  for (fixture, offset_size) in cases {
    let file = Hdf5File::parse(@fs.read_file_to_bytes(fixture))
    let one_axis = file.add_chunked_dataset(
      "/group",
      "resizable",
      [4UL, 3UL],
      [2U, 3U],
      Int32Values([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
      maximum_dimensions=[None, Some(3UL)],
      fill_value=Int32Values([-1]),
      filters=[Shuffle, Deflate(4)],
    ) catch {
      error => fail("create one-axis resizable dataset: " + error.message())
    }
    let many_axes = Hdf5File::parse(one_axis).add_chunked_dataset(
      "/group",
      "growable",
      [3UL, 4UL],
      [2U, 2U],
      UInt16Values([1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U, 10U, 11U, 12U]),
      maximum_dimensions=[None, None],
      fill_value=UInt16Values([99U]),
      filters=[Deflate(3)],
    ) catch {
      error => fail("create multi-axis resizable dataset: " + error.message())
    }
    let edited = Hdf5File::parse(many_axes) catch {
      error => fail("parse created file: " + error.message())
    }
    assert_eq(edited.superblock.offset_size, offset_size)
    let one_metadata = edited.read_dataset_metadata("/group/resizable") catch {
      error => fail("read one-axis metadata: " + error.message())
    }
    assert_eq(one_metadata.dataspace.maximum_dimensions.length(), 2)
    assert_eq(
      one_metadata.dataspace.maximum_dimensions[0],
      0xFFFFFFFFFFFFFFFFUL,
    )
    assert_eq(one_metadata.dataspace.maximum_dimensions[1], 3UL)
    assert_eq(one_metadata.layout.kind(), Chunked)
    let many_metadata = edited.read_dataset_metadata("/group/growable") catch {
      error => fail("read multi-axis metadata: " + error.message())
    }
    assert_eq(many_metadata.dataspace.maximum_dimensions, [
      0xFFFFFFFFFFFFFFFFUL, 0xFFFFFFFFFFFFFFFFUL,
    ])
    let one_values = edited.read_dataset("/group/resizable") catch {
      error => fail("read one-axis values: " + error.message())
    }
    match one_values.values {
      Integers(values) =>
        assert_eq(values, [0L, 1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, 10L, 11L])
      _ => fail("expected one-axis resizable integer values")
    }
    let many_values = edited.read_dataset("/group/growable") catch {
      error => fail("read multi-axis values: " + error.message())
    }
    match many_values.values {
      UnsignedIntegers(values) =>
        assert_eq(values, [
          1UL, 2UL, 3UL, 4UL, 5UL, 6UL, 7UL, 8UL, 9UL, 10UL, 11UL, 12UL,
        ])
      _ => fail("expected multi-axis resizable values")
    }
  }
}

///|
fn created_nested_record_type(
  reference_size : Int32,
) -> Datatype raise Hdf5Error {
  let text = Datatype::write_vlen_string_with_address_size(reference_size)
  let object_reference = Datatype::write_object_reference_with_address_size(
    reference_size,
  )
  Datatype::write_compound([
    ("name", text),
    (
      "counts",
      Datatype::write_vlen_sequence_with_address_size(
        Datatype::write_integer(true, 4),
        reference_size,
      ),
    ),
    ("labels", Datatype::write_array(text, [2UL])),
    ("target", object_reference),
    ("pair", Datatype::write_array(object_reference, [2UL])),
    (
      "region",
      Datatype::write_region_reference_with_address_size(reference_size),
    ),
  ])
}

///|
fn created_nested_record(
  name : String,
  counts : Array[DatasetValue],
  labels : Array[String],
  target : String?,
  pair : Array[String?],
  region_path : String?,
) -> DatasetValue {
  Compound([
    { name: "name", value: Text(name), },
    { name: "counts", value: Sequence(counts), },
    { name: "labels", value: Elements(labels.map(label => Text(label))), },
    { name: "target", value: ObjectReference(target), },
    { name: "pair", value: Elements(pair.map(path => ObjectReference(path))), },
    {
      name: "region",
      value: RegionReference(region_path, RegionPoints([[0UL, 0UL]])),
    },
  ])
}

///|
fn created_nested_fields(
  file : Hdf5File,
  path : String,
) -> Array[Array[CompoundFieldValue]] raise Hdf5Error {
  let values = match file.read_dataset(path).values {
    Structured(values) => values
    _ => raise InvalidField("expected structured created values")
  }
  let records : Array[Array[CompoundFieldValue]] = []
  for value in values {
    match value {
      Compound(fields) => records.push(fields)
      _ => raise InvalidField("expected created compound values")
    }
  }
  records
}

///|
test "nested variable-length and reference values can be created" {
  let cases : Array[(String, Int32)] = [
    ("testdata/symbol_table_32.h5", 4),
    ("testdata/references.h5", 8),
  ]
  for (fixture, reference_size) in cases {
    let record_type = created_nested_record_type(reference_size)
    let first = created_nested_record(
      "moon",
      [SignedInteger(1L), SignedInteger(-2L)],
      ["一", ""],
      Some("/matrix"),
      [Some("/matrix"), None],
      Some("/matrix"),
    )
    let second = created_nested_record(
      "",
      [],
      ["", "二"],
      None,
      [None, Some("/matrix")],
      None,
    )
    let file = Hdf5File::parse(@fs.read_file_to_bytes(fixture))
    let contiguous = file.add_dataset(
      "/group",
      "nested_records",
      [2UL],
      StructuredValues(record_type, [first, second]),
    ) catch {
      error => fail("create nested records: " + error.message())
    }
    let chunked = if reference_size == 8 {
      Hdf5File::parse(contiguous).add_chunked_dataset(
        "/group",
        "nested_chunked",
        [3UL],
        [2U],
        StructuredValues(record_type, [first, second, first]),
        fill_value=StructuredValues(record_type, [second]),
        filters=[Shuffle, Deflate(3)],
      ) catch {
        error => fail("create nested chunked records: " + error.message())
      }
    } else {
      Hdf5File::parse(contiguous).add_chunked_dataset(
        "/group",
        "nested_chunked",
        [3UL],
        [2U],
        StructuredValues(record_type, [first, second, first]),
        filters=[Shuffle, Deflate(3)],
      ) catch {
        error => fail("create nested chunked records: " + error.message())
      }
    }
    let resizable = if reference_size == 8 {
      Hdf5File::parse(chunked).add_chunked_dataset(
        "/group",
        "nested_resizable",
        [1UL],
        [1U],
        StructuredValues(record_type, [first]),
        maximum_dimensions=[None],
        fill_value=StructuredValues(record_type, [second]),
        filters=[Deflate(3)],
      ) catch {
        error => fail("create nested resizable records: " + error.message())
      }
    } else {
      Hdf5File::parse(chunked).add_chunked_dataset(
        "/group",
        "nested_resizable",
        [1UL],
        [1U],
        StructuredValues(record_type, [first]),
        maximum_dimensions=[None],
        filters=[Deflate(3)],
      ) catch {
        error => fail("create nested resizable records: " + error.message())
      }
    }
    let edited = Hdf5File::parse(resizable) catch {
      error => fail("parse nested records: " + error.message())
    }
    assert_eq(edited.superblock.offset_size, reference_size)
    let records = created_nested_fields(edited, "/group/nested_records")
    assert_eq(records.length(), 2)
    assert_eq(records[0][0].value, Text("moon"))
    assert_eq(
      records[0][1].value,
      Sequence([SignedInteger(1L), SignedInteger(-2L)]),
    )
    assert_eq(records[0][2].value, Elements([Text("一"), Text("")]))
    let target_bytes = match records[0][3].value {
      Reference(bytes) => bytes
      _ => fail("expected a nested object reference")
    }
    assert_eq(target_bytes.length(), reference_size)
    assert_eq(
      edited.resolve_object_reference(target_bytes).unwrap().path,
      Some("/matrix"),
    )
    let pair = match records[0][4].value {
      Elements(items) => items
      _ => fail("expected a nested object-reference array")
    }
    match pair[1] {
      Reference(bytes) => assert_eq(bytes, Bytes::make(reference_size, b'\x00'))
      _ => fail("expected a null nested object reference")
    }
    let region_bytes = match records[0][5].value {
      Reference(bytes) => bytes
      _ => fail("expected a nested region reference")
    }
    assert_eq(
      edited.resolve_region_reference(region_bytes).unwrap().path,
      Some("/matrix"),
    )

    let chunked_records = created_nested_fields(edited, "/group/nested_chunked")
    assert_eq(chunked_records.length(), 3)
    assert_eq(chunked_records[1][2].value, Elements([Text(""), Text("二")]))
    let resizable_metadata = edited.read_dataset_metadata(
      "/group/nested_resizable",
    )
    assert_eq(resizable_metadata.layout.kind(), Chunked)
    assert_eq(
      resizable_metadata.dataspace.maximum_dimensions[0],
      0xFFFFFFFFFFFFFFFFUL,
    )
    let header = edited.object_header_at(
      edited.resolve("/group/nested_resizable").address,
    )
    if reference_size == 8 {
      let fill = dataset_fill_value(header, resizable_metadata.datatype)
      assert_eq(fill.length(), record_type.size)
    } else {
      assert_true(header.first(msg_fill_value) is None)
      let rejected = try {
        ignore(
          Hdf5File::parse(resizable).add_dataset(
            "/group",
            "unsafe_nested_fill",
            [1UL],
            StructuredValues(record_type, [second]),
            fill_value=StructuredValues(record_type, [second]),
          ),
        )
        None
      } catch {
        error => Some(error)
      }
      assert_eq(
        rejected,
        Some(
          UnsupportedFeature(
            "custom fills for structured variable-length values require eight-byte file addresses",
          ),
        ),
      )
    }
  }
}
