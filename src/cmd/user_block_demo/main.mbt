///|
fn run(args : Array[String]) -> Int32 {
  if args.length() != 5 {
    eprintln(
      "usage: moon run src/cmd/user_block_demo -- <input.h5> <new.h5> <edited.h5> <existing-edited.h5>",
    )
    return 2
  }
  try {
    let writer = @lib.Hdf5Writer::new()
    writer.set_user_block(512)
    writer.add_group("/group")
    writer.add_dataset("/group/values", [4UL], Int32Values([1, 2, 3, 4]))
    writer.add_compact_dataset(
      "/labels",
      [2UL],
      VariableStringValues(["one", "two"]),
    )
    writer.add_attribute(
      "/group/values",
      "unit",
      [],
      FixedStringValues(["count"], 8),
    )

    let created = writer.encode()
    @fs.write_bytes_to_file(args[2], created)
    let file = @lib.Hdf5File::parse(created)
    let labelled = file.patch_dataset_values(
      "/labels",
      VariableStringValues(["a longer value", "second"]),
    )
    let edited = @lib.Hdf5File::parse(labelled).patch_dataset_values(
      "/group/values",
      Int32Values([10, 20, 30, 40]),
    )
    @fs.write_bytes_to_file(args[3], edited)

    let existing = @lib.Hdf5File::parse(@fs.read_file_to_bytes(args[1]))
    let created_group = existing.add_group("/", "created")
    let existing_edited = @lib.Hdf5File::parse(created_group).patch_dataset_values(
      "/x",
      Int32Values([10, 20, 30, 40]),
    )
    @fs.write_bytes_to_file(args[4], existing_edited)
    println("wrote " + args[2] + ", " + args[3] + " and " + args[4])
    0
  } catch {
    error => {
      eprintln("user-block error: " + error.to_string())
      1
    }
  }
}

///|
fn main {
  @sys.exit(run(@env.args()))
}
