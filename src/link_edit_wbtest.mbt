///|
fn edited_group_links(
  bytes : Bytes,
  path : String,
) -> Array[GroupLink] raise Hdf5Error {
  let file = Hdf5File::parse(bytes)
  let node = file.resolve(path)
  let header = file.object_header_at(node.address)
  file.links(header)
}

///|
test "links can be added renamed and deleted" {
  let writer = Hdf5Writer::new()
  writer.add_group("/links")
  writer.add_dataset("/target", [2UL], Int64Values([3L, 4L]))
  writer.add_dataset("/other", [], UInt64Values([9UL]))
  writer.add_attribute("/links", "note", [], FixedStringValues(["keep"], 4))
  let file = Hdf5File::parse(writer.encode())

  let hard = file.add_hard_link("/links", "hard", "/target") catch {
    error => fail("add hard link: " + error.message())
  }
  let soft = Hdf5File::parse(hard).add_soft_link("/links", "soft", "../target") catch {
    error => fail("add soft link: " + error.message())
  }
  let external = Hdf5File::parse(soft).add_external_link(
    "/links", "external", "other.h5", "/value",
  ) catch {
    error => fail("add external link: " + error.message())
  }
  let edited = Hdf5File::parse(external)
  let links = edited_group_links(external, "/links")
  assert_eq(links.map(link => link.name), ["hard", "soft", "external"])
  assert_eq(
    edited.resolve("/links/hard").address,
    edited.resolve("/target").address,
  )
  assert_eq(
    edited.resolve("/links/soft").address,
    edited.resolve("/target").address,
  )
  match links.search_by(link => link.name == "external") {
    Some(index) => {
      assert_eq(links[index].cache_type, 64U)
      assert_eq(links[index].link_target, Some("other.h5"))
      assert_eq(links[index].external_path, Some("/value"))
    }
    None => fail("expected external link")
  }
  match edited.attributes("/links")[0].values {
    Some(Strings(values)) => assert_eq(values, ["keep"])
    _ => fail("expected parent attribute to survive link edits")
  }

  let renamed = edited.rename_link("/links", "hard", "renamed") catch {
    error => fail("rename link: " + error.message())
  }
  let renamed_file = Hdf5File::parse(renamed)
  assert_eq(edited_group_links(renamed, "/links").map(link => link.name), [
    "renamed", "soft", "external",
  ])
  assert_eq(integer_values(renamed_file.read_dataset("/links/renamed")), [
    3L, 4L,
  ])

  let deleted = renamed_file.delete_link("/links", "soft") catch {
    error => fail("delete link: " + error.message())
  }
  let deleted_file = Hdf5File::parse(deleted)
  assert_eq(edited_group_links(deleted, "/links").map(link => link.name), [
    "renamed", "external",
  ])
  let missing = try {
    ignore(deleted_file.resolve("/links/soft"))
    None
  } catch {
    error => Some(error)
  }
  assert_true(missing is Some(_))
}

///|
test "link editing converts between compact and dense storage" {
  let writer = Hdf5Writer::new()
  writer.add_group("/many")
  for index in 0..<8 {
    writer.add_dataset(
      "/many/value_" + index.to_string(),
      [],
      UInt64Values([index.to_uint64()]),
    )
  }
  writer.add_dataset("/target", [], UInt64Values([42UL]))
  let file = Hdf5File::parse(writer.encode())
  let dense_bytes = file.add_hard_link("/many", "target", "/target")
  let dense_file = Hdf5File::parse(dense_bytes)
  let dense_header = dense_file.object_header_at(
    dense_file.resolve("/many").address,
  )
  assert_true(
    !dense_header.messages.any(message => message.msg_type == msg_link),
  )
  let dense_links = dense_file.links(dense_header)
  assert_eq(dense_links.length(), 9)
  assert_eq(dense_links[8].name, "target")
  assert_eq(unsigned_values(dense_file.read_dataset("/many/target")), [42UL])

  let renamed = dense_file.rename_link("/many", "value_4", "renamed")
  let renamed_file = Hdf5File::parse(renamed)
  assert_eq(
    renamed_file
    .links(renamed_file.object_header_at(renamed_file.resolve("/many").address))
    .map(link => link.name)[4],
    "renamed",
  )
  assert_eq(unsigned_values(renamed_file.read_dataset("/many/renamed")), [4UL])

  let compact_bytes = renamed_file.delete_link("/many", "value_7")
  let compact_file = Hdf5File::parse(compact_bytes)
  let compact_header = compact_file.object_header_at(
    compact_file.resolve("/many").address,
  )
  assert_true(
    compact_header.messages.any(message => message.msg_type == msg_link),
  )
  let compact_links = compact_file.links(compact_header)
  assert_eq(compact_links.length(), 8)
  assert_eq(compact_links.map(link => link.name), [
    "value_0", "value_1", "value_2", "value_3", "renamed", "value_5", "value_6",
    "target",
  ])
}

///|
test "link editing rejects conflicts and invalid names" {
  let writer = Hdf5Writer::new()
  writer.add_group("/links")
  writer.add_dataset("/target", [], UInt64Values([1UL]))
  let file = Hdf5File::parse(writer.encode())

  let duplicate = try {
    let added = file.add_hard_link("/links", "item", "/target")
    ignore(Hdf5File::parse(added).add_hard_link("/links", "item", "/target"))
    None
  } catch {
    error => Some(error)
  }
  assert_eq(duplicate, Some(InvalidArgument("link name already exists: item")))
  let invalid = try {
    ignore(file.add_soft_link("/links", "bad/name", "/target"))
    None
  } catch {
    error => Some(error)
  }
  assert_eq(invalid, Some(InvalidArgument("link name is invalid")))
  let missing = try {
    ignore(file.add_hard_link("/links", "missing", "/does-not-exist"))
    None
  } catch {
    error => Some(error)
  }
  assert_true(missing is Some(_))
  let missing_parent = try {
    ignore(file.add_soft_link("/target", "child", "/target"))
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    missing_parent,
    Some(InvalidArgument("link parent is not a group: /target")),
  )
  let oversized_external = try {
    ignore(
      file.add_external_link(
        "/links",
        "long",
        "f".repeat(40000),
        "p".repeat(30000),
      ),
    )
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    oversized_external,
    Some(InvalidArgument("external link information is too long")),
  )
}

///|
test "link editing rebuilds four-byte address groups" {
  let bytes = @fs.read_file_to_bytes("testdata/interop_32.h5")
  let source = Hdf5File::parse(bytes)
  assert_eq(source.superblock.offset_size, 4)
  let mut edited = bytes
  for index in 0..<7 {
    let file = Hdf5File::parse(edited)
    edited = file.add_hard_link("/", "alias_" + index.to_string(), "/raw") catch {
      error => fail("add four-byte hard link: " + error.message())
    }
  }
  let file = Hdf5File::parse(edited)
  let header = file.object_header_at(file.resolve("/").address)
  let (_, heap_address, name_index_address, order_index_address) = editable_link_storage(
    header, 4,
  )
  assert_true(heap_address != heap_undefined_address(4))
  assert_true(name_index_address != heap_undefined_address(4))
  assert_true(order_index_address != heap_undefined_address(4))
  let records = read_v2_btree_records(
    edited,
    name_index_address,
    4,
    file.superblock.length_size,
    file.superblock.base_address,
    [5, 6],
    "four-byte edited link index",
  )
  assert_eq(records.records.length(), 9)
  let heap = parse_fractal_heap(
    edited,
    heap_address,
    file.superblock.offset_size,
    file.superblock.length_size,
    file.superblock.base_address,
  )
  assert_true(heap.blocks.length() > 0)
  let links = edited_group_links(edited, "/")
  assert_eq(links.length(), 9)
  for index in 0..<7 {
    assert_eq(
      file.resolve("/alias_" + index.to_string()).address,
      file.resolve("/raw").address,
    )
  }
  let renamed = file.rename_link("/", "alias_0", "alias_renamed") catch {
    error => fail("rename four-byte hard link: " + error.message())
  }
  let renamed_file = Hdf5File::parse(renamed)
  assert_eq(
    renamed_file.resolve("/alias_renamed").address,
    renamed_file.resolve("/raw").address,
  )
}

///|
test "link editing spans dense heap direct blocks" {
  let writer = Hdf5Writer::new()
  writer.add_group("/large")
  let target = "/" + "t".repeat(60000)
  let mut bytes = writer.encode()
  for index in 0..<9 {
    let file = Hdf5File::parse(bytes)
    bytes = file.add_soft_link("/large", "link_" + index.to_string(), target) catch {
      error => fail("add large soft link: " + error.message())
    }
  }
  let file = Hdf5File::parse(bytes)
  let header = file.object_header_at(file.resolve("/large").address)
  let (_, heap_address, _, _) = editable_link_storage(
    header,
    file.superblock.offset_size,
  )
  assert_true(heap_address != heap_undefined_address(8))
  let heap = parse_fractal_heap(
    bytes,
    heap_address,
    file.superblock.offset_size,
    file.superblock.length_size,
    file.superblock.base_address,
  )
  assert_true(heap.blocks.length() > 1)
  let links = edited_group_links(bytes, "/large")
  assert_eq(links.length(), 9)
  for link in links {
    assert_eq(link.cache_type, 2U)
    assert_eq(link.link_target, Some(target))
  }
}

///|
test "dense link creation order uses the full link-info field" {
  let writer = Hdf5Writer::new()
  writer.add_group("/ordered")
  for index in 0..<9 {
    writer.add_dataset(
      "/ordered/value_" + index.to_string(),
      [],
      UInt64Values([index.to_uint64()]),
    )
  }
  let file = Hdf5File::parse(writer.encode())
  let node = file.resolve("/ordered")
  let header = file.object_header_at(node.address)
  let links = file.editable_links(header)
  links[8] = { link: links[8].link, creation_order: 70000UL, }
  let rebuilt = file.write_rebuilt_links(
    node.address,
    header,
    links,
    @buffer.Buffer::Buffer(),
  )
  let rebuilt_file = Hdf5File::parse(rebuilt)
  let rebuilt_header = rebuilt_file.object_header_at(node.address)
  let parsed = rebuilt_file.editable_links(rebuilt_header)
  assert_eq(parsed[8].creation_order, 70000UL)
}

///|
test "link editing stores huge external links in the huge-object index" {
  let writer = Hdf5Writer::new()
  writer.add_group("/huge")
  writer.add_dataset("/target", [], UInt64Values([1UL]))
  let mut bytes = writer.encode()
  for index in 0..<8 {
    let file = Hdf5File::parse(bytes)
    bytes = file.add_hard_link("/huge", "alias_" + index.to_string(), "/target") catch {
      error => fail("add huge-group hard link: " + error.message())
    }
  }
  let target_file = "f".repeat(32755)
  let target_path = "/" + "p".repeat(32776)
  let file = Hdf5File::parse(bytes)
  bytes = file.add_external_link(
    "/huge", "huge_external", target_file, target_path,
  ) catch {
    error => fail("add huge external link: " + error.message())
  }
  let file = Hdf5File::parse(bytes)
  let links = edited_group_links(bytes, "/huge")
  assert_eq(links.length(), 9)
  let external = match links.search_by(link => link.name == "huge_external") {
    Some(index) => links[index]
    None => fail("expected the huge external link")
  }
  assert_eq(external.cache_type, 64U)
  assert_eq(external.link_target, Some(target_file))
  assert_eq(external.external_path, Some(target_path))
  let header = file.object_header_at(file.resolve("/huge").address)
  let (_, heap_address, _, _) = editable_link_storage(
    header,
    file.superblock.offset_size,
  )
  let heap = parse_fractal_heap(
    bytes,
    heap_address,
    file.superblock.offset_size,
    file.superblock.length_size,
    file.superblock.base_address,
  )
  assert_true(heap.huge_btree_address != heap_undefined_address(8))
  let records = read_v2_btree_records(
    bytes,
    heap.huge_btree_address,
    file.superblock.offset_size,
    file.superblock.length_size,
    file.superblock.base_address,
    [1],
    "edited huge link index",
  )
  assert_eq(records.records.length(), 1)
  let record = Reader::new(records.records[0])
  let address = record.read_uint_le(file.superblock.offset_size)
  let stored_size = record.read_uint_le(file.superblock.length_size)
  let object_id = record.read_uint_le(file.superblock.length_size)
  assert_true(stored_size > 65535UL)
  assert_eq(object_id, 1UL)
  assert_eq(Reader::at(bytes, address.to_int()).read_u8(), b'\x01')
}
