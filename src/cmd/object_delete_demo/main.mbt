///|
fn run(args : Array[String]) -> Int32 {
  if args.length() != 3 {
    eprintln(
      "usage: moon run src/cmd/object_delete_demo -- <input.h5> <output.h5>",
    )
    return 2
  }
  try {
    let file = @lib.Hdf5File::parse(@fs.read_file_to_bytes(args[1]))
    let aliased = file.add_hard_link("/", "matrix_alias", "/matrix")
    let deleted = @lib.Hdf5File::parse(aliased).delete_object("/matrix")
    @fs.write_bytes_to_file(args[2], deleted)
    println("wrote " + args[2])
    0
  } catch {
    error => {
      eprintln("object deletion error: " + error.to_string())
      1
    }
  }
}

///|
fn main {
  @sys.exit(run(@env.args()))
}
