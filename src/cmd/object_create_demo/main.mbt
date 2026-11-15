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
    edited = root_file.add_group(inner_parent, "created_leaf")
    edited = @lib.Hdf5File::parse(edited).add_dataset(
      "/created_root",
      "numbers",
      [2UL, 3UL],
      @lib.Int64Values([1L, 2L, 3L, 4L, 5L, 6L]),
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
