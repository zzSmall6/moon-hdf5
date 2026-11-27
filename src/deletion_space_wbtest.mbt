///|
test "deleting fixed-width datasets returns contiguous and chunk raw space" {
  let writer = Hdf5Writer::new()
  writer.set_file_space_reserve(32768)
  writer.add_group("/edits")
  writer.add_dataset("/contiguous", [256UL], Int32Values(Array::make(256, 1)))
  writer.add_chunked_dataset(
    "/chunked",
    [16UL, 16UL],
    [4U, 4U],
    Int32Values(Array::make(256, 2)),
    filters=[Deflate(4)],
  )
  let file = Hdf5File::parse(writer.encode())
  let free = allocation_free_total(file)
  let contiguous = file.read_dataset_metadata("/contiguous").layout.data_size
  let chunked = patch_space_stored(file, "/chunked")
  let after_contiguous = Hdf5File::parse(file.delete_object("/contiguous"))
  assert_eq(allocation_free_total(after_contiguous), free + contiguous)
  assert_true(
    try {
      after_contiguous.resolve("/contiguous") |> ignore
      false
    } catch {
      _ => true
    },
  )
  let after_chunked = Hdf5File::parse(
    after_contiguous.delete_object("/chunked"),
  )
  assert_eq(allocation_free_total(after_chunked), free + contiguous + chunked)
  let reopened = Hdf5File::parse(after_chunked.add_group("/edits", "new"))
  assert_eq(allocation_free_total(reopened), free + contiguous + chunked)
}

///|
test "deleting a hard link keeps the target and does not reclaim storage" {
  let writer = Hdf5Writer::new()
  writer.set_file_space_reserve(8192)
  writer.add_group("/edits")
  writer.add_dataset("/values", [8UL], Int32Values([1, 2, 3, 4, 5, 6, 7, 8]))
  let file = Hdf5File::parse(
    Hdf5File::parse(writer.encode()).add_hard_link("/edits", "alias", "/values"),
  )
  let free = allocation_free_total(file)
  let edited = Hdf5File::parse(file.delete_link("/", "values"))
  assert_eq(allocation_free_total(edited), free)
  assert_eq(integer_values(edited.read_dataset("/edits/alias")), [
    1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L,
  ])
}

///|
test "references and variable-length values protect deleted raw storage" {
  let writer = Hdf5Writer::new()
  writer.set_file_space_reserve(16384)
  writer.add_dataset("/target", [2UL], Int32Values([1, 2]))
  writer.add_dataset("/names", [2UL], VariableStringValues(["alpha", "月亮"]))
  writer.add_dataset(
    "/refs",
    [1UL],
    StructuredValues(Datatype::write_object_reference(), [
      ObjectReference(Some("/target")),
    ]),
  )
  let file = Hdf5File::parse(writer.encode())
  let free = allocation_free_total(file)
  assert_eq(
    allocation_free_total(Hdf5File::parse(file.delete_object("/names"))),
    free,
  )
  assert_eq(
    allocation_free_total(Hdf5File::parse(file.delete_object("/target"))),
    free,
  )
}

///|
test "deletion rejects malformed overlap before returning a changed image" {
  let writer = Hdf5Writer::new()
  writer.set_file_space_reserve(4096)
  writer.add_dataset("/values", [8UL], Int32Values(Array::make(8, 1)))
  let file = Hdf5File::parse(writer.encode())
  let layout = file.read_dataset_metadata("/values").layout
  let header = file.object_header_at(file.resolve("/values").address)
  let manager = file.free_space_managers()[0].sections[0]
  let output = file.bytes.to_fixedarray()
  patch_little_endian_uint(
    output,
    layout.data_address_offset,
    layout.data_address_width,
    manager.address,
    "overlap test",
  )
  let message = header.first(msg_data_layout).unwrap()
  patch_metadata_checksum_in_place(
    output,
    message.chunk_start,
    message.checksum_offset,
  )
  let corrupt = Hdf5File::parse(output.unsafe_reinterpret_as_bytes())
  assert_true(
    try {
      corrupt.delete_object("/values") |> ignore
      false
    } catch {
      InvalidField(message) => message.contains("overlaps recorded free space")
      _ => false
    },
  )
}

///|
test "object deletion removes every hard alias before reclaiming a payload once" {
  let writer = Hdf5Writer::new()
  writer.set_file_space_reserve(8192)
  writer.add_group("/edits")
  writer.add_dataset("/values", [8UL], Int32Values(Array::make(8, 42)))
  let original = Hdf5File::parse(writer.encode())
  let aliased = Hdf5File::parse(
    original.add_hard_link("/edits", "alias", "/values"),
  )
  let edited = Hdf5File::parse(aliased.delete_object("/values"))
  assert_true(!edited.exists("/values") && !edited.exists("/edits/alias"))
  assert_eq(allocation_free_total(edited), 8192UL + 32UL)
  assert_true(
    try {
      edited.delete_object("/values") |> ignore
      false
    } catch {
      _ => true
    },
  )
}

///|
test "vlen and group deletion retain storage while reference attributes block reclamation" {
  let writer = Hdf5Writer::new()
  writer.set_file_space_reserve(8192)
  writer.add_group("/group")
  writer.add_dataset(
    "/group/names",
    [2UL],
    VariableStringValues(["alpha", "月亮"]),
  )
  let file = Hdf5File::parse(writer.encode())
  assert_eq(
    allocation_free_total(Hdf5File::parse(file.delete_object("/group/names"))),
    8192UL,
  )
  assert_eq(
    allocation_free_total(Hdf5File::parse(file.delete_object("/group"))),
    8192UL,
  )
  let writer = Hdf5Writer::new()
  writer.set_file_space_reserve(8192)
  writer.add_dataset("/values", [2UL], Int32Values([1, 2]))
  writer.add_attribute(
    "/",
    "reference",
    [1UL],
    ObjectReferenceValues([Some("/values")]),
  )
  let file = Hdf5File::parse(writer.encode())
  assert_eq(
    allocation_free_total(Hdf5File::parse(file.delete_object("/values"))),
    8192UL,
  )
}
