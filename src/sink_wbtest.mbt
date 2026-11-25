///|
priv struct RecordingSink {
  inner : Hdf5MemorySink
  mut writes : Int32
  mut set_lengths : Int32
  mut largest_write : Int32
}

///|
impl Hdf5Sink for RecordingSink with fn length(self : RecordingSink) -> UInt64 {
  Hdf5Sink::length(self.inner)
}

///|
impl Hdf5Sink for RecordingSink with fn write_at(
  self : RecordingSink,
  offset : UInt64,
  bytes : Bytes,
) -> Unit raise Hdf5Error {
  self.writes = self.writes + 1
  if bytes.length() > self.largest_write {
    self.largest_write = bytes.length()
  }
  Hdf5Sink::write_at(self.inner, offset, bytes)
}

///|
impl Hdf5Sink for RecordingSink with fn set_length(
  self : RecordingSink,
  length : UInt64,
) -> Unit raise Hdf5Error {
  self.set_lengths = self.set_lengths + 1
  Hdf5Sink::set_length(self.inner, length)
}

///|
priv struct RejectingSink {
  mut writes : Int32
}

///|
impl Hdf5Sink for RejectingSink with fn length(self : RejectingSink) -> UInt64 {
  self.writes.to_uint64()
}

///|
impl Hdf5Sink for RejectingSink with fn write_at(
  self : RejectingSink,
  offset : UInt64,
  bytes : Bytes,
) -> Unit raise Hdf5Error {
  self.writes = self.writes + 1
  ignore(bytes)
  raise InvalidArgument("sink rejected write at " + offset.to_string())
}

///|
impl Hdf5Sink for RejectingSink with fn set_length(
  self : RejectingSink,
  length : UInt64,
) -> Unit raise Hdf5Error {
  ignore(self)
  ignore(length)
}

///|
#derive(@debug.Debug)
priv struct CopyRecordingSource {
  bytes : Bytes
  mut reads : Int32
  mut largest_read : Int32
}

///|
impl Hdf5Source for CopyRecordingSource with fn length(
  self : CopyRecordingSource,
) -> UInt64 {
  self.bytes.length().to_uint64()
}

///|
impl Hdf5Source for CopyRecordingSource with fn read_at(
  self : CopyRecordingSource,
  offset : UInt64,
  length : Int32,
) -> Bytes raise Hdf5Error {
  self.reads = self.reads + 1
  if length > self.largest_read {
    self.largest_read = length
  }
  Hdf5Source::read_at(self.bytes, offset, length)
}

///|
fn sink_operation_error(operation : () -> Unit raise Hdf5Error) -> Hdf5Error? {
  try {
    operation()
    None
  } catch {
    error => Some(error)
  }
}

///|
test "memory sink supports sparse writes and length changes" {
  let sink = Hdf5MemorySink::new()
  Hdf5Sink::write_at(sink, 4UL, b"bc")
  assert_eq(Hdf5Sink::length(sink), 6UL)
  assert_eq(sink.to_bytes(), b"\x00\x00\x00\x00bc")
  Hdf5Sink::write_at(sink, 0UL, b"AB")
  assert_eq(sink.to_bytes(), b"AB\x00\x00bc")
  Hdf5Sink::write_at(sink, 10UL, b"z")
  assert_eq(Hdf5Sink::length(sink), 11UL)
  assert_eq(
    sink.to_bytes().get_view(start=6, end=10).unwrap(),
    b"\x00\x00\x00\x00",
  )
  Hdf5Sink::set_length(sink, 2UL)
  assert_eq(sink.to_bytes(), b"AB")
  Hdf5Sink::set_length(sink, 5UL)
  assert_eq(sink.to_bytes(), b"AB\x00\x00\x00")
}

///|
test "writer sink output resets the destination and parses correctly" {
  let writer = Hdf5Writer::new()
  writer.set_attribute_sharing(true)
  writer.set_filter_sharing(true)
  writer.set_user_block_content(b"moon-hdf5-prefix")
  writer.add_group("/experiment")
  writer.add_dataset("/experiment/ids", [3UL], Int64Values([10L, 20L, 30L]))
  writer.add_chunked_dataset(
    "/experiment/codes",
    [4UL, 3UL],
    [2U, 2U],
    Int32Values([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    filters=[Shuffle, Deflate(4), Fletcher32],
  )
  writer.add_attribute(
    "/experiment/ids",
    "unit",
    [],
    FixedStringValues(["items"], 8),
  )
  let sink = Hdf5MemorySink::new()
  Hdf5Sink::set_length(sink, 4096UL)
  Hdf5Sink::write_at(sink, 0UL, Bytes::from_array(Array::make(4096, b'x')))
  writer.write_to(sink)
  let bytes = sink.to_bytes()
  assert_eq(Hdf5Sink::length(sink), bytes.length().to_uint64())
  assert_eq(bytes.get_view(start=0, end=16).unwrap(), b"moon-hdf5-prefix")
  let file = Hdf5File::parse(bytes)
  assert_eq(file.superblock_version(), 3)
  let ids = file.read_dataset("/experiment/ids")
  match ids.values {
    Integers(values) => assert_eq(values, [10L, 20L, 30L])
    _ => fail("expected integer ids")
  }
  let codes = file.read_dataset("/experiment/codes")
  match codes.values {
    Integers(values) =>
      assert_eq(values, [1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, 10L, 11L, 12L])
    _ => fail("expected integer codes")
  }
}

///|
test "writer emits independent payload writes through the sink" {
  let writer = Hdf5Writer::new()
  writer.add_group("/data")
  writer.add_dataset(
    "/data/values",
    [8UL],
    Int64Values([1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L]),
  )
  writer.add_attribute("/data", "note", [], FixedStringValues(["sink"], 8))
  let sink = RecordingSink::{
    inner: Hdf5MemorySink::new(),
    writes: 0,
    set_lengths: 0,
    largest_write: 0,
  }
  writer.write_to(sink)
  assert_true(sink.writes > 2)
  assert_true(sink.set_lengths >= 2)
  assert_true(sink.largest_write < Hdf5Sink::length(sink).to_int())
  assert_eq(sink.inner.to_bytes(), writer.encode())
  Hdf5File::parse(sink.inner.to_bytes()) |> ignore
}

///|
test "writer and copy propagate sink failures" {
  let writer = Hdf5Writer::new()
  writer.add_dataset("/values", [2UL], Int64Values([1L, 2L]))
  let sink = RejectingSink::{ writes: 0, }
  let write_error = sink_operation_error(() => writer.write_to(sink))
  assert_true(write_error is Some(InvalidArgument(_)))

  let bytes = writer.encode()
  let file = Hdf5File::parse(bytes)
  let copy_error = sink_operation_error(() => file.copy_to(sink))
  assert_true(copy_error is Some(InvalidArgument(_)))
}

///|
test "file copy preserves bytes and bounds source reads" {
  let values = Array::make(1100000, 7U)
  values[0] = 1U
  values[1099999] = 9U
  let writer = Hdf5Writer::new()
  writer.add_dataset("/payload", [1100000UL], UInt8Values(values))
  let bytes = writer.encode()
  let source = CopyRecordingSource::{ bytes, reads: 0, largest_read: 0, }
  let file = Hdf5File::parse_source(source)
  let sink = Hdf5MemorySink::new()
  file.copy_to(sink)
  assert_eq(sink.to_bytes(), bytes)
  assert_true(source.reads > 1)
  assert_true(source.largest_read <= 1048576)
  let copied = Hdf5File::parse(sink.to_bytes())
  match copied.read_dataset("/payload").values {
    Integers(_) | UnsignedIntegers(_) => ()
    _ => fail("expected copied numeric payload")
  }
  let bytes_file = Hdf5File::parse(bytes)
  let bytes_sink = Hdf5MemorySink::new()
  Hdf5Sink::set_length(bytes_sink, 64UL)
  bytes_file.copy_to(bytes_sink)
  assert_eq(bytes_sink.to_bytes(), bytes)
}
