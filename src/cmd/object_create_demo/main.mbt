///|
fn created_record_type(
  reference_size : Int32,
) -> @lib.Datatype raise @lib.Hdf5Error {
  let text = @lib.Datatype::write_vlen_string_with_address_size(reference_size)
  let object_reference = @lib.Datatype::write_object_reference_with_address_size(
    reference_size,
  )
  @lib.Datatype::write_compound([
    ("name", text),
    (
      "counts",
      @lib.Datatype::write_vlen_sequence_with_address_size(
        @lib.Datatype::write_integer(true, 4),
        reference_size,
      ),
    ),
    ("labels", @lib.Datatype::write_array(text, [2UL])),
    ("target", object_reference),
    ("pair", @lib.Datatype::write_array(object_reference, [2UL])),
    (
      "region",
      @lib.Datatype::write_region_reference_with_address_size(reference_size),
    ),
  ])
}

///|
fn created_record(
  name : String,
  counts : Array[@lib.DatasetValue],
  labels : Array[String],
  target : String?,
  pair : Array[String?],
  region_path : String?,
) -> @lib.DatasetValue {
  @lib.Compound([
    { name: "name", value: @lib.Text(name), },
    { name: "counts", value: @lib.Sequence(counts), },
    {
      name: "labels",
      value: @lib.Elements(labels.map(label => @lib.Text(label))),
    },
    { name: "target", value: @lib.ObjectReference(target), },
    {
      name: "pair",
      value: @lib.Elements(pair.map(path => @lib.ObjectReference(path))),
    },
    {
      name: "region",
      value: @lib.RegionReference(region_path, @lib.RegionPoints([[0UL, 0UL]])),
    },
  ])
}

///|
fn run(args : Array[String]) -> Int32 {
  if args.length() != 3 {
    eprintln(
      "usage: moon run src/cmd/object_create_demo -- <input.h5> <output.h5>",
    )
    return 2
  }
  try {
    let input = @fs.read_file_to_bytes(args[1])
    let file = @lib.Hdf5File::parse(input)
    let mut edited = file.add_group("/", "created_root")
    let root_file = @lib.Hdf5File::parse(edited)
    let inner_parent = try {
      ignore(root_file.resolve("/group"))
      "/group"
    } catch {
      _ => "/"
    }
    let reference_target = try {
      ignore(root_file.resolve("/group/values"))
      "/group/values"
    } catch {
      _ => "/group/child"
    }
    edited = root_file.add_group(inner_parent, "created_leaf")
    edited = @lib.Hdf5File::parse(edited).add_dataset(
      "/created_root",
      "numbers",
      [2UL, 3UL],
      @lib.Int64Values([1L, 2L, 3L, 4L, 5L, 6L]),
      fill_value=@lib.Int64Values([-1L]),
    )
    edited = @lib.Hdf5File::parse(edited).add_dataset(
      inner_parent + "/created_leaf",
      "fixed",
      [2UL],
      @lib.UInt32Values([100U, 200U]),
    )
    edited = @lib.Hdf5File::parse(edited).add_attribute(
      "/created_root",
      "note",
      [],
      @lib.UInt64Values([11UL]),
    )
    edited = @lib.Hdf5File::parse(edited).add_attribute(
      "/created_root/numbers",
      "unit",
      [],
      @lib.FixedStringValues(["count"], 8),
    )
    edited = @lib.Hdf5File::parse(edited).add_dataset(
      "/created_root",
      "labels",
      [4UL],
      @lib.VariableStringValues(["moon", "数据", "", "moon"]),
    )
    edited = @lib.Hdf5File::parse(edited).add_dataset(
      "/created_root",
      "references",
      [3UL],
      @lib.ObjectReferenceValues([Some("/matrix"), None, Some(reference_target)]),
    )
    let record_type = created_record_type(file.address_size())
    let first_record = created_record(
      "moon",
      [@lib.SignedInteger(1L), @lib.SignedInteger(-2L)],
      ["一", ""],
      Some("/matrix"),
      [Some("/matrix"), None],
      Some("/matrix"),
    )
    let empty_record = created_record(
      "",
      [],
      ["", "二"],
      None,
      [None, Some("/matrix")],
      None,
    )
    edited = @lib.Hdf5File::parse(edited).add_dataset(
      "/created_root",
      "nested_records",
      [2UL],
      @lib.StructuredValues(record_type, [first_record, empty_record]),
    )
    if file.address_size() == 8 {
      let fill = @lib.StructuredValues(record_type, [empty_record])
      edited = @lib.Hdf5File::parse(edited).add_chunked_dataset(
        "/created_root",
        "nested_chunked",
        [3UL],
        [2U],
        @lib.StructuredValues(record_type, [
          first_record, empty_record, first_record,
        ]),
        fill_value=fill,
        filters=[Shuffle, Deflate(3)],
      )
      edited = @lib.Hdf5File::parse(edited).add_chunked_dataset(
        "/created_root",
        "nested_resizable",
        [1UL],
        [1U],
        @lib.StructuredValues(record_type, [first_record]),
        maximum_dimensions=[None],
        fill_value=fill,
        filters=[Deflate(3)],
      )
    } else {
      edited = @lib.Hdf5File::parse(edited).add_chunked_dataset(
        "/created_root",
        "nested_chunked",
        [3UL],
        [2U],
        @lib.StructuredValues(record_type, [
          first_record, empty_record, first_record,
        ]),
        filters=[Shuffle, Deflate(3)],
      )
      edited = @lib.Hdf5File::parse(edited).add_chunked_dataset(
        "/created_root",
        "nested_resizable",
        [1UL],
        [1U],
        @lib.StructuredValues(record_type, [first_record]),
        maximum_dimensions=[None],
        filters=[Deflate(3)],
      )
    }
    edited = @lib.Hdf5File::parse(edited).add_chunked_dataset(
      "/created_root",
      "chunked",
      [4UL, 3UL],
      [2U, 3U],
      @lib.UInt32Values([0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U, 10U, 11U]),
      fill_value=@lib.UInt32Values([99U]),
      filters=[Shuffle, Deflate(6), Fletcher32],
    )
    edited = @lib.Hdf5File::parse(edited).add_chunked_dataset(
      "/created_root",
      "chunked_labels",
      [4UL],
      [2U],
      @lib.VariableStringValues(["moon", "数据", "", "moon"]),
      filters=[Deflate(3)],
    )
    edited = @lib.Hdf5File::parse(edited).add_chunked_dataset(
      "/created_root",
      "resizable",
      [4UL, 3UL],
      [2U, 3U],
      @lib.Int32Values([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
      maximum_dimensions=[None, Some(3UL)],
      fill_value=@lib.Int32Values([-1]),
      filters=[Shuffle, Deflate(4)],
    )
    edited = @lib.Hdf5File::parse(edited).add_chunked_dataset(
      "/created_root",
      "growable",
      [3UL, 4UL],
      [2U, 2U],
      @lib.UInt16Values([1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U, 10U, 11U, 12U]),
      maximum_dimensions=[None, None],
      fill_value=@lib.UInt16Values([99U]),
      filters=[Deflate(3)],
    )
    let external_path = args[2] + ".external.bin"
    @fs.write_bytes_to_file(
      external_path,
      Bytes::from_array([
        b'\x00', b'\x00', b'\x00', b'\x00', b'\x0a', b'\x00', b'\x00', b'\x00', b'\x14',
        b'\x00', b'\x00', b'\x00', b'\x1e', b'\x00', b'\x00', b'\x00', b'\x28', b'\x00',
        b'\x00', b'\x00', b'\x32', b'\x00', b'\x00', b'\x00', b'\x3c', b'\x00', b'\x00',
        b'\x00',
      ]),
    )
    edited = @lib.Hdf5File::parse(edited).add_external_dataset(
      "/created_root",
      "external",
      [6UL],
      @lib.Int32Values([10, 20, 30, 40, 50, 60]),
      [{ file_name: external_path, file_offset: 4UL, data_size: 24UL, }],
      fill_value=@lib.Int32Values([-1]),
    )
    @fs.write_bytes_to_file(args[2], edited)
    println("wrote " + args[2])
    0
  } catch {
    error => {
      eprintln("object creation error: " + error.to_string())
      1
    }
  }
}

///|
fn main {
  @sys.exit(run(@env.args()))
}
