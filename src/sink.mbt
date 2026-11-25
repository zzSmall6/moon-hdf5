///|
/// A random-access destination for HDF5 bytes.
///
/// Writes may arrive in any order. A write beyond the current length must
/// zero-fill the gap. `set_length` grows with zero bytes or truncates existing
/// content to the requested logical length.
pub(open) trait Hdf5Sink {
  fn length(Self) -> UInt64
  fn write_at(Self, offset : UInt64, bytes : Bytes) -> Unit raise Hdf5Error
  fn set_length(Self, length : UInt64) -> Unit raise Hdf5Error
}

///|
/// A growable, random-access in-memory sink.
pub struct Hdf5MemorySink {
  priv data : Array[Byte]
}

///|
pub fn Hdf5MemorySink::new() -> Hdf5MemorySink {
  { data: [], }
}

///|
/// Returns a snapshot of the bytes written so far.
pub fn Hdf5MemorySink::to_bytes(self : Hdf5MemorySink) -> Bytes {
  Bytes::from_array(self.data)
}

///|
fn Hdf5MemorySink::ensure_length(self : Hdf5MemorySink, length : Int32) -> Unit {
  if self.data.length() < length {
    self.data.resize(length, b'\x00')
  }
}

///|
pub impl Hdf5Sink for Hdf5MemorySink with fn length(self : Hdf5MemorySink) -> UInt64 {
  self.data.length().to_uint64()
}

///|
pub impl Hdf5Sink for Hdf5MemorySink with fn write_at(
  self : Hdf5MemorySink,
  offset : UInt64,
  bytes : Bytes,
) -> Unit raise Hdf5Error {
  guard offset <= 0x7FFFFFFFUL &&
    bytes.length().to_int64() <= 0x7FFFFFFFL - offset.reinterpret_as_int64() else {
    raise UnsupportedFeature("sink offset is too large")
  }
  let start = offset.to_int()
  let end = start + bytes.length()
  self.ensure_length(end)
  for index in 0..<bytes.length() {
    self.data[start + index] = bytes[index]
  }
}

///|
pub impl Hdf5Sink for Hdf5MemorySink with fn set_length(
  self : Hdf5MemorySink,
  length : UInt64,
) -> Unit raise Hdf5Error {
  guard length <= 0x7FFFFFFFUL else {
    raise UnsupportedFeature("sink length is too large")
  }
  let target = length.to_int()
  if target < self.data.length() {
    self.data.truncate(target)
  } else {
    self.ensure_length(target)
  }
}

///|
pub extend Hdf5MemorySink with Hdf5Sink::{length, write_at, set_length}

///|
fn writer_sink_write_at(
  sink : &Hdf5Sink,
  base_address : Int32,
  address : UInt64,
  payload : Bytes,
) -> Unit raise Hdf5Error {
  guard address <= 0x7FFFFFFFUL &&
    base_address.to_int64() + address.reinterpret_as_int64() <= 0x7FFFFFFFL &&
    payload.length().to_int64() <=
    0x7FFFFFFFL - base_address.to_int64() - address.reinterpret_as_int64() else {
    raise UnsupportedFeature("file offset is too large")
  }
  sink.write_at(base_address.to_uint64() + address, payload)
}

///|
/// Copies this file image into a random-access sink in bounded chunks.
///
/// A source-backed file is read through `Hdf5Source`; a bytes-backed file is
/// sliced without allocating a second complete image.
pub fn Hdf5File::copy_to(
  self : Hdf5File,
  sink : &Hdf5Sink,
) -> Unit raise Hdf5Error {
  let length = match self.source {
    Some(source) => source.length()
    None => self.bytes.length().to_uint64()
  }
  let chunk_size = 1048576
  sink.set_length(0UL)
  let mut offset = 0UL
  while offset < length {
    let remaining = length - offset
    let count = if remaining < chunk_size.to_uint64() {
      remaining.to_int()
    } else {
      chunk_size
    }
    let bytes = match self.source {
      Some(source) => source.read_at(offset, count)
      None =>
        self.bytes
        .get_view(start=offset.to_int(), end=offset.to_int() + count)
        .unwrap()
        .to_owned()
    }
    sink.write_at(offset, bytes)
    offset = offset + count.to_uint64()
  }
  sink.set_length(length)
}
