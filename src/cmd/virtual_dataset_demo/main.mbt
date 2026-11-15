///|
fn run(args : Array[String]) -> Int32 {
  if args.length() != 3 {
    eprintln(
      "usage: moon run src/cmd/virtual_dataset_demo -- <new.h5> <edited.h5>",
    )
    return 2
  }
  try {
    let writer = @lib.Hdf5Writer::new()
    writer.add_dataset(
      "/source",
      [3UL, 4UL],
      Int32Values([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    )
    writer.add_dataset("/vector", [6UL], Int32Values([0, 1, 2, 3, 4, 5]))
    let datatype = @lib.Datatype::write_integer(true, 4)
    writer.add_virtual_dataset("/same", [3UL, 4UL], datatype, [
      {
        source_file: ".",
        source_dataset: "/source",
        source_selection: @lib.RegionAll,
        virtual_selection: @lib.RegionAll,
      },
    ])
    writer.add_virtual_dataset(
      "/mapped",
      [2UL, 3UL],
      datatype,
      [
        {
          source_file: ".",
          source_dataset: "/source",
          source_selection: @lib.RegionHyperslab(
            [1UL, 1UL],
            [1UL, 1UL],
            [2UL, 2UL],
            [1UL, 1UL],
          ),
          virtual_selection: @lib.RegionHyperslab(
            [0UL, 0UL],
            [1UL, 1UL],
            [2UL, 2UL],
            [1UL, 1UL],
          ),
        },
      ],
      fill_value=@lib.SignedInteger(-1L),
    )
    writer.add_virtual_dataset("/strided", [2UL, 2UL], datatype, [
      {
        source_file: ".",
        source_dataset: "/source",
        source_selection: @lib.RegionHyperslab(
          [0UL, 0UL],
          [2UL, 2UL],
          [2UL, 2UL],
          [1UL, 1UL],
        ),
        virtual_selection: @lib.RegionAll,
      },
    ])
    writer.add_virtual_dataset(
      "/empty",
      [2UL, 2UL],
      datatype,
      [
        {
          source_file: ".",
          source_dataset: "/source",
          source_selection: @lib.RegionEmpty,
          virtual_selection: @lib.RegionEmpty,
        },
      ],
      fill_value=@lib.SignedInteger(-5L),
    )
    writer.add_virtual_dataset(
      "/multiple_mappings",
      [6UL],
      datatype,
      [
        {
          source_file: ".",
          source_dataset: "/vector",
          source_selection: @lib.RegionHyperslab([0UL], [1UL], [3UL], [1UL]),
          virtual_selection: @lib.RegionHyperslab([0UL], [1UL], [3UL], [1UL]),
        },
        {
          source_file: ".",
          source_dataset: "/vector",
          source_selection: @lib.RegionHyperslab([3UL], [1UL], [3UL], [1UL]),
          virtual_selection: @lib.RegionHyperslab([3UL], [1UL], [3UL], [1UL]),
        },
      ],
      fill_value=@lib.SignedInteger(-1L),
    )
    @fs.write_bytes_to_file(args[1], writer.encode())

    let source_file = @fs.read_file_to_bytes("testdata/symbol_table_32.h5")
    let source = @lib.Hdf5File::parse(source_file)
    let created_source = source.add_dataset(
      "/group",
      "virtual_source",
      [2UL],
      @lib.Int32Values([7, 8]),
    )
    let created = @lib.Hdf5File::parse(created_source).add_virtual_dataset(
      "/group",
      "virtual_mapped",
      [2UL],
      datatype,
      [
        {
          source_file: ".",
          source_dataset: "/group/virtual_source",
          source_selection: @lib.RegionAll,
          virtual_selection: @lib.RegionAll,
        },
      ],
      fill_value=@lib.SignedInteger(-9L),
    )
    @fs.write_bytes_to_file(args[2], created)
    println("wrote " + args[1] + " and " + args[2])
    0
  } catch {
    error => {
      eprintln("virtual dataset write error: " + error.to_string())
      1
    }
  }
}

///|
fn main {
  @sys.exit(run(@env.args()))
}
