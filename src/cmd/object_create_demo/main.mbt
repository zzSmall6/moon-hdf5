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
