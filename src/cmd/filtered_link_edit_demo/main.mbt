///|
fn run(args : Array[String]) -> Int32 {
  if args.length() != 3 {
    eprintln(
      "usage: moon run src/cmd/filtered_link_edit_demo -- <input.h5> <output.h5>",
    )
    return 2
  }
  try {
    let file = @lib.Hdf5File::parse(@fs.read_file_to_bytes(args[1]))
    let edited = file.rename_link("/filtered", "item_0000", "renamed_item")
    @fs.write_bytes_to_file(args[2], edited)
    println("wrote " + args[2])
    0
  } catch {
    error => {
      eprintln("filtered link edit error: " + error.to_string())
      1
    }
  }
}

///|
fn main {
  @sys.exit(run(@env.args()))
}
