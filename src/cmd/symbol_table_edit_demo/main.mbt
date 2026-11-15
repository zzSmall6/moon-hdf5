///|
fn run(args : Array[String]) -> Int32 {
  if args.length() != 3 && args.length() != 4 {
    eprintln(
      "usage: moon run src/cmd/symbol_table_edit_demo -- <input.h5> <output.h5> [compact|dense]",
    )
    return 2
  }
  let dense = args.length() == 3 || args[3] == "dense"
  try {
    let input = @fs.read_file_to_bytes(args[1])
    let file = @lib.Hdf5File::parse(input)
    let mut edited = file.add_hard_link("/", "matrix_alias", "/matrix")
    edited = @lib.Hdf5File::parse(edited).add_soft_link(
      "/", "group_soft", "group",
    )
    edited = @lib.Hdf5File::parse(edited).add_external_link(
      "/", "external", "external-target.h5", "/matrix",
    )
    if dense {
      for index in 0..<4 {
        edited = @lib.Hdf5File::parse(edited).add_soft_link(
          "/",
          "extra_" + index.to_string(),
          "group",
        )
      }
    }
    edited = @lib.Hdf5File::parse(edited).rename_link(
      "/", "alias", "alias_renamed",
    )
    edited = @lib.Hdf5File::parse(edited).rename_link(
      "/", "matrix_alias", "matrix_renamed",
    )
    edited = @lib.Hdf5File::parse(edited).delete_link("/", "matrix_renamed")
    edited = @lib.Hdf5File::parse(edited).add_hard_link(
      "/group", "values_alias", "/group/values",
    )
    edited = @lib.Hdf5File::parse(edited).add_soft_link("/group", "self", ".")
    edited = @lib.Hdf5File::parse(edited).add_attribute(
      "/group",
      "note",
      [],
      @lib.UInt64Values([7UL]),
    )
    @fs.write_bytes_to_file(args[2], edited)
    println("wrote " + args[2])
    0
  } catch {
    error => {
      eprintln("symbol-table edit error: " + error.to_string())
      1
    }
  }
}

///|
fn main {
  @sys.exit(run(@env.args()))
}
