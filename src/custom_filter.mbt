///|
/// Decodes one third-party HDF5 filter during dataset chunk reads.
///
/// HDF5 applies filters in reverse pipeline order. `data` is the payload at
/// this stage, `element_size` is the dataset element width, and `expected_size`
/// is the decoded byte count. Implementations must return exactly
/// `expected_size` bytes. The file layer verifies that invariant after the
/// callback returns.
pub(open) trait Hdf5CustomFilter {
  fn id(Self) -> Int32
  fn decode(
    Self,
    data : Bytes,
    parameters : Array[UInt32],
    element_size : Int32,
    expected_size : Int32,
  ) -> Bytes raise Hdf5Error
}

///|
fn validate_custom_filter(filter : &Hdf5CustomFilter) -> Unit raise Hdf5Error {
  let id = filter.id()
  guard id >= 0 && id <= 65535 else {
    raise InvalidArgument("custom filter ID must fit in an HDF5 uint16")
  }
  guard id != 1 && id != 2 && id != 3 && id != 6 else {
    raise InvalidArgument("custom filters cannot replace built-in filter IDs")
  }
}

///|
fn register_custom_filter(
  filters : Array[&Hdf5CustomFilter],
  filter : &Hdf5CustomFilter,
) -> Unit raise Hdf5Error {
  validate_custom_filter(filter)
  let id = filter.id()
  guard filters.search_by(existing => existing.id() == id) is None else {
    raise InvalidArgument(
      "custom filter is already registered: " + id.to_string(),
    )
  }
  filters.push(filter)
}
