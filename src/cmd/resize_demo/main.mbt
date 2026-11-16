///|
fn resize_parse_uint(raw : String) -> UInt64? {
  if raw.length() == 0 {
    return None
  }
  let mut value = 0UL
  for index in 0..<raw.length() {
    let byte = raw[index].to_int()
    if byte < 48 || byte > 57 {
      return None
    }
    let digit = (byte - 48).to_uint64()
    if value > (0xFFFFFFFFFFFFFFFFUL - digit) / 10UL {
      return None
    }
    value = value * 10UL + digit
  }
  Some(value)
}

///|
fn resize_parse_dimensions(raw : String) -> Array[UInt64]? {
  let dimensions : Array[UInt64] = []
  for item in raw.split(",") {
    guard resize_parse_uint(item.to_owned()) is Some(value) else { return None }
    dimensions.push(value)
  }
  Some(dimensions)
}

///|
fn run(args : Array[String]) -> Int32 {
  if args.length() != 5 {
    eprintln(
      "usage: moon run src/cmd/resize_demo -- <input.h5> <output.h5> <path> <dim0,dim1,...>",
    )
    return 2
  }
  let dimensions = match resize_parse_dimensions(args[4]) {
    Some(dimensions) => dimensions
    None => {
      eprintln("invalid dimensions: " + args[4])
      return 2
    }
  }
  try {
    let input = @fs.read_file_to_bytes(args[1])
    let resized = @lib.Hdf5File::parse(input).resize_dataset(
      args[3],
      dimensions,
    )
    @fs.write_bytes_to_file(args[2], resized)
    println("resized " + args[3] + " in " + args[2])
    0
  } catch {
    error => {
      eprintln("resize error: " + error.to_string())
      1
    }
  }
}

///|
fn main {
  @sys.exit(run(@env.args()))
}
