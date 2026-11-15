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
