///|
priv struct TestDeflateFilter {
  marker : Int32
}

///|
impl Hdf5CustomFilter for TestDeflateFilter with fn id(self : TestDeflateFilter) -> Int32 {
  self.marker |> ignore
  42
}

///|
impl Hdf5CustomFilter for TestDeflateFilter with fn decode(
  self : TestDeflateFilter,
  data : Bytes,
  parameters : Array[UInt32],
  element_size : Int32,
  expected_size : Int32,
) -> Bytes raise Hdf5Error {
  self.marker |> ignore
  element_size |> ignore
  guard parameters.length() == 1 && parameters[0] <= 9U else {
    raise InvalidField("invalid deflate parameters")
  }
  @zlib.unzlib(data[:], checksum=true, limit=expected_size) catch {
    _ => raise InvalidField("invalid custom deflate chunk")
  }
}

///|
priv struct BuiltinTestFilter {}

///|
impl Hdf5CustomFilter for BuiltinTestFilter with fn id(self : BuiltinTestFilter) -> Int32 {
  self |> ignore
  1
}

///|
impl Hdf5CustomFilter for BuiltinTestFilter with fn decode(
  self : BuiltinTestFilter,
  data : Bytes,
  parameters : Array[UInt32],
  element_size : Int32,
  expected_size : Int32,
) -> Bytes raise Hdf5Error {
  self |> ignore
  parameters |> ignore
  element_size |> ignore
  expected_size |> ignore
  data
}

///|
priv struct OversizedTestFilter {}

///|
impl Hdf5CustomFilter for OversizedTestFilter with fn id(
  self : OversizedTestFilter,
) -> Int32 {
  self |> ignore
  43
}

///|
impl Hdf5CustomFilter for OversizedTestFilter with fn decode(
  self : OversizedTestFilter,
  data : Bytes,
  parameters : Array[UInt32],
  element_size : Int32,
  expected_size : Int32,
) -> Bytes raise Hdf5Error {
  self |> ignore
  data |> ignore
  parameters |> ignore
  element_size |> ignore
  expected_size |> ignore
  FixedArray::make(1024, b'\x00').unsafe_reinterpret_as_bytes()
}

///|
fn custom_filter_find(bytes : Bytes, pattern : Bytes) -> Int32 {
  let mut offset = 0
  while offset + pattern.length() <= bytes.length() {
    let mut matches = true
    for index in 0..<pattern.length() {
      if bytes[offset + index] != pattern[index] {
        matches = false
        break
      }
    }
    if matches {
      return offset
    }
    offset = offset + 1
  }
  -1
}

///|
fn custom_filter_fixture() -> Bytes raise Hdf5Error {
  let writer = Hdf5Writer::new()
  writer.add_chunked_dataset(
    "/values",
    [8UL],
    [4U],
    Int64Values([1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L]),
    filters=[Deflate(6)],
  )
  let encoded = writer.encode()
  let pattern = b"\x02\x01\x01\x00\x00\x00\x01\x00\x06\x00\x00\x00"
  let position = custom_filter_find(encoded, pattern)
  guard position >= 0 else { raise InvalidField("filter pipeline not found") }
  let original = Hdf5File::parse(encoded)
  let header_offset = address_to_offset(
    original.resolve("/values").address,
    original.superblock.base_address,
  )
  let chunk_size = encoded[header_offset + 6].to_int() |
    (encoded[header_offset + 7].to_int() << 8)
  let checksum_offset = header_offset + 8 + chunk_size
  guard position > header_offset &&
    position + pattern.length() < checksum_offset else {
    raise InvalidField(
      "filter pipeline is not in the first object-header chunk",
    )
  }
  let mutated = encoded.to_array()
  mutated[position + 2] = b'\x2a'
  mutated[position + 3] = b'\x00'
  let checksum = metadata_checksum(
    Bytes::from_array(mutated),
    header_offset,
    checksum_offset - header_offset,
  )
  for index in 0..<4 {
    mutated[checksum_offset + index] = ((checksum >> (index * 8)) & 0xFFU).to_byte()
  }
  Bytes::from_array(mutated)
}

///|
test "custom filters decode dataset reads selections and iterators" {
  let bytes = custom_filter_fixture()
  let file = Hdf5File::parse(bytes)
  let error = try {
    file.read_dataset("/values") |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(error, Some(UnsupportedFeature("filter 42 is not supported")))
  let custom : &Hdf5CustomFilter = { marker: 1, }
  let filtered = file.with_custom_filter(custom)
  assert_eq(integer_values(filtered.read_dataset("/values")), [
    1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L,
  ])
  let selected = filtered.read_dataset_selection(
    "/values",
    Hyperslab([1UL], [3UL], [2UL]),
  )
  match selected.values {
    Integers(values) => assert_eq(values, [2L, 4L, 6L])
    _ => fail("expected selected integer values")
  }
  let chunks = filtered.chunk_iterator("/values", [2UL])
  let first = chunks.next().unwrap()
  match first.values {
    Integers(values) => assert_eq(values, [1L, 2L])
    _ => fail("expected chunk integer values")
  }
  let duplicate = try {
    filtered.with_custom_filter(custom) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    duplicate,
    Some(InvalidArgument("custom filter is already registered: 42")),
  )
}

///|
test "resolver registers custom filters and rejects built-in IDs" {
  let bytes = custom_filter_fixture()
  let resolver = Hdf5Resolver::new(bytes)
  let custom : &Hdf5CustomFilter = { marker: 1, }
  resolver.add_custom_filter(custom)
  assert_eq(integer_values(resolver.read_dataset("/values")), [
    1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L,
  ])
  let builtin : &Hdf5CustomFilter = BuiltinTestFilter::{ }
  let error = try {
    resolver.add_custom_filter(builtin) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    error,
    Some(InvalidArgument("custom filters cannot replace built-in filter IDs")),
  )
}

///|
test "custom filters cannot return oversized intermediate output" {
  let decoder : &Hdf5CustomFilter = OversizedTestFilter::{ }
  let filter : Filter = { id: 43, parameters: [], decoder: Some(decoder), }
  let error = try {
    decode_filters(b"\x00", [filter], 0U, 1, 1) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(error, Some(InvalidField("oversized custom filter chunk")))
}
