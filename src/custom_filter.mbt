///|
/// Encodes and decodes one third-party HDF5 filter for dataset chunks.
///
/// HDF5 applies filters in reverse pipeline order. `data` is the payload at
/// this stage, `element_size` is the dataset element width, and
/// `expected_size` is the decoded chunk byte count. A decoder must return
/// exactly `expected_size` bytes. Implementations that only read files may
/// omit `encode`; trying to write through such a filter raises
/// `UnsupportedFeature`.
pub(open) trait Hdf5CustomFilter {
  fn id(Self) -> Int32
  fn name(Self) -> String = _
  fn decode(
    Self,
    data : Bytes,
    parameters : Array[UInt32],
    element_size : Int32,
    expected_size : Int32,
  ) -> Bytes raise Hdf5Error
  fn encode(
    Self,
    data : Bytes,
    parameters : Array[UInt32],
    element_size : Int32,
    expected_size : Int32,
  ) -> Bytes raise Hdf5Error = _
}

///|
impl Hdf5CustomFilter with fn name(self) {
  "custom-filter-" + self.id().to_string()
}

///|
impl Hdf5CustomFilter with fn encode(
  self,
  data,
  parameters,
  element_size,
  expected_size,
) {
  self |> ignore
  data |> ignore
  parameters |> ignore
  element_size |> ignore
  expected_size |> ignore
  raise UnsupportedFeature(
    "custom filter " + self.id().to_string() + " does not provide an encoder",
  )
}

///|
fn validate_custom_filter_id(id : Int32) -> Unit raise Hdf5Error {
  guard id >= 0 && id <= 65535 else {
    raise InvalidArgument("custom filter ID must fit in an HDF5 uint16")
  }
  guard id != 1 && id != 2 && id != 3 && id != 6 else {
    raise InvalidArgument("custom filters cannot replace built-in filter IDs")
  }
}

///|
fn validate_custom_filter(filter : &Hdf5CustomFilter) -> Unit raise Hdf5Error {
  let id = filter.id()
  validate_custom_filter_id(id)
  if id >= 256 {
    guard @utf8.encode(filter.name()).length() <= 65535 else {
      raise InvalidArgument("custom filter name is too long")
    }
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

///|
fn find_custom_filter(
  filters : Array[&Hdf5CustomFilter],
  id : Int32,
) -> &Hdf5CustomFilter raise Hdf5Error {
  match filters.search_by(filter => filter.id() == id) {
    Some(index) => filters[index]
    None =>
      raise InvalidArgument(
        "custom filter is not registered: " + id.to_string(),
      )
  }
}
