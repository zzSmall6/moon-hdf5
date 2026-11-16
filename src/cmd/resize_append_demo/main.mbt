///|
fn append_demo_names(count : Int32) -> Array[String] {
  let values : Array[String] = []
  for index in 0..<count {
    values.push("value-" + index.to_string())
  }
  values
}

///|
fn append_demo_filtered_values(count : Int32) -> Array[Int32] {
  let values : Array[Int32] = Array::make(count, 9)
  values[0] = -11
  values
}

///|
fn run(args : Array[String]) -> Int32 {
  if args.length() != 2 {
    eprintln("usage: moon run src/cmd/resize_append_demo -- <output.h5>")
    return 2
  }
  try {
    let writer = @lib.Hdf5Writer::new()
    writer.add_extensible_dataset(
      "/numbers",
      [2UL],
      [None],
      [2U],
      @lib.Int64Values([1L, 2L]),
      fill_value=@lib.Int64Values([-1L]),
    )
    writer.add_extensible_dataset(
      "/filtered",
      [4UL],
      [None],
      [4U],
      @lib.Int32Values([1, 2, 3, 4]),
      fill_value=@lib.Int32Values([-7]),
      filters=[@lib.Shuffle, @lib.Deflate(6)],
    )
    writer.add_extensible_dataset(
      "/matrix",
      [2UL, 2UL],
      [None, None],
      [1U, 1U],
      @lib.Int32Values([1, 2, 3, 4]),
      fill_value=@lib.Int32Values([-1]),
    )
    writer.add_extensible_dataset(
      "/names",
      [1UL],
      [None],
      [1U],
      @lib.VariableStringValues(["moon"]),
      fill_value=@lib.VariableStringValues(["missing"]),
    )
    writer.add_extensible_dataset(
      "/filtered_names",
      [1UL],
      [None],
      [2U],
      @lib.VariableStringValues(["moon"]),
      fill_value=@lib.VariableStringValues(["missing"]),
      filters=[@lib.Shuffle, @lib.Deflate(6)],
    )
    writer.add_extensible_dataset(
      "/resized",
      [2UL],
      [None],
      [2U],
      @lib.Int64Values([10L, 20L]),
      fill_value=@lib.Int64Values([-1L]),
    )
    writer.add_extensible_dataset(
      "/hidden",
      [4UL],
      [None],
      [2U],
      @lib.Int64Values([1L, 2L, 3L, 4L]),
      fill_value=@lib.Int64Values([-1L]),
    )
    let mut bytes = writer.encode()
    bytes = @lib.Hdf5File::parse(bytes).append_dataset(
      "/numbers",
      [2UL],
      @lib.Int64Values([3L, 4L]),
    )
    bytes = @lib.Hdf5File::parse(bytes).append_dataset(
      "/filtered",
      [513UL],
      @lib.Int32Values(append_demo_filtered_values(513)),
    )
    bytes = @lib.Hdf5File::parse(bytes).append_dataset(
      "/matrix",
      [1UL, 1UL],
      @lib.Int32Values([5]),
    )
    bytes = @lib.Hdf5File::parse(bytes).append_dataset(
      "/names",
      [200UL],
      @lib.VariableStringValues(append_demo_names(200)),
    )
    bytes = @lib.Hdf5File::parse(bytes).append_dataset(
      "/filtered_names",
      [3UL],
      @lib.VariableStringValues(["数据", "", "tail"]),
    )
    bytes = @lib.Hdf5File::parse(bytes).resize_dataset("/resized", [5UL])
    let shrunk = @lib.Hdf5File::parse(bytes).resize_dataset("/hidden", [2UL])
    bytes = @lib.Hdf5File::parse(shrunk).resize_dataset("/hidden", [4UL])
    @fs.write_bytes_to_file(args[1], bytes)
    println("wrote " + args[1])
    0
  } catch {
    error => {
      eprintln("resize/append error: " + error.to_string())
      1
    }
  }
}

///|
fn main {
  @sys.exit(run(@env.args()))
}
