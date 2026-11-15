///|
fn extended_float_values() -> Array[@lib.DatasetValue] {
  [
    @lib.FloatingPoint(1.5),
    @lib.FloatingPoint(-2.25),
    @lib.FloatingPoint(1UL.reinterpret_as_double()),
    @lib.FloatingPoint(0x7FF0000000000000UL.reinterpret_as_double()),
    @lib.FloatingPoint(0xFFF0000000000000UL.reinterpret_as_double()),
    @lib.FloatingPoint(0x7FF8000000000000UL.reinterpret_as_double()),
    @lib.FloatingPoint(-0.0),
  ]
}

///|
fn run(args : Array[String]) -> Int32 {
  if args.length() != 2 {
    eprintln("usage: moon run src/cmd/extended_float_demo -- <output.h5>")
    return 2
  }
  try {
    let writer = @lib.Hdf5Writer::new()
    for size in [10, 16] {
      for little_endian in [true, false] {
        let name = "/x87_" +
          size.to_string() +
          (if little_endian { "_le" } else { "_be" })
        writer.add_dataset(
          name,
          [7UL],
          StructuredValues(
            @lib.Datatype::write_float_with_order(size, little_endian),
            extended_float_values(),
          ),
        )
      }
    }
    @fs.write_bytes_to_file(args[1], writer.encode())
    println("wrote " + args[1])
    0
  } catch {
    error => {
      eprintln("extended floating-point error: " + error.to_string())
      1
    }
  }
}

///|
fn main {
  @sys.exit(run(@env.args()))
}
