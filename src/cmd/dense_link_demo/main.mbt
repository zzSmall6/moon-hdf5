///|
fn dense_link_demo_writer() -> @lib.Hdf5Writer raise @lib.Hdf5Error {
  let writer = @lib.Hdf5Writer::new()
  writer.add_dataset("/target", [1UL], UInt64Values([42UL]))
  writer.add_group("/links")
  for index in 0..<65536 {
    writer.add_soft_link("/links", "link_" + index.to_string(), "/target")
  }
  writer
}

///|
fn run(args : Array[String]) -> Int32 {
  if args.length() != 3 {
    eprintln(
      "usage: moon run src/cmd/dense_link_demo -- <created.h5> <edited.h5>",
    )
    return 2
  }
  try {
    let created = dense_link_demo_writer().encode()
    @fs.write_bytes_to_file(args[1], created)
    let edited = @lib.Hdf5File::parse(created).add_soft_link(
      "/links", "link_65536", "/target",
    )
    @fs.write_bytes_to_file(args[2], edited)
    println("wrote " + args[1] + " and " + args[2])
    0
  } catch {
    error => {
      eprintln("dense-link error: " + error.to_string())
      1
    }
  }
}

///|
fn main {
  @sys.exit(run(@env.args()))
}
