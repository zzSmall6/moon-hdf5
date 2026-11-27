///|
fn main {
  let args = @env.args()
  try {
    if args.length() == 4 && args[1] == "create-chunks" {
      let writer = @lib.Hdf5Writer::new()
      writer.add_group("/edits")
      writer.set_file_space_reserve(1048576)
      if args[2] == "userblock" {
        writer.set_user_block_content(b"chunk-reserve")
      }
      let values = Array::make(1200, 0)
      for index in 0..<values.length() {
        values[index] = index - 600
      }
      for filtered in [false, true] {
        let suffix = if filtered { "filtered" } else { "raw" }
        let filters : Array[@lib.Hdf5WriteFilter] = if filtered {
          [Shuffle, Deflate(4), Fletcher32]
        } else {
          []
        }
        writer.add_chunked_dataset(
          "/single_" + suffix,
          [30UL, 40UL],
          [30U, 40U],
          Int32Values(values),
          filters~,
        )
        writer.add_chunked_dataset(
          "/fixed_" + suffix,
          [30UL, 40UL],
          [3U, 4U],
          Int32Values(values),
          filters~,
        )
        writer.add_extensible_dataset(
          "/array_" + suffix,
          [30UL, 40UL],
          [None, Some(40UL)],
          [3U, 4U],
          Int32Values(values),
          filters~,
        )
        writer.add_extensible_dataset(
          "/tree_" + suffix,
          [30UL, 40UL],
          [None, None],
          [3U, 4U],
          Int32Values(values),
          filters~,
        )
        if filtered {
          writer.set_chunk_filter_mask("/fixed_" + suffix, 0, 7U)
        }
      }
      writer.add_chunked_dataset(
        "/names",
        [5UL],
        [2U],
        VariableStringValues(["alpha", "月亮", "", "beta", "alpha"]),
        filters=[Deflate(4)],
      )
      @fs.write_bytes_to_file(args[3], writer.encode())
      return
    }
    if args.length() == 4 && args[1] == "create" {
      let writer = @lib.Hdf5Writer::new()
      writer.set_file_space_reserve(8192)
      if args[2] == "userblock" {
        writer.set_user_block_content(b"reserved-space")
      }
      writer.add_dataset(
        "/target",
        [512UL],
        Int32Values(Array::make(512, 42)),
        fill_value=Int32Values([-1]),
      )
      writer.add_dataset(
        "/second",
        [512UL],
        Int32Values(Array::make(512, 7)),
        fill_value=Int32Values([-1]),
      )
      writer.add_chunked_dataset(
        "/chunks",
        [4UL],
        [2U],
        Int32Values([1, 2, 3, 4]),
        filters=[Shuffle, Deflate(4)],
      )
      writer.add_attribute(
        "/target",
        "unit",
        [],
        FixedStringValues(["items"], 8),
      )
      writer.add_attribute(
        "/second",
        "unit",
        [],
        FixedStringValues(["items"], 8),
      )
      writer.add_attribute(
        "/chunks",
        "unit",
        [],
        FixedStringValues(["items"], 8),
      )
      @fs.write_bytes_to_file(args[3], writer.encode())
      return
    }
    if args.length() == 5 && args[1] == "delete" {
      let file = @lib.Hdf5File::parse(@fs.read_file_to_bytes(args[2]))
      @fs.write_bytes_to_file(args[4], file.delete_object(args[3]))
      return
    }
    if args.length() == 5 &&
      (args[1] == "relocate" || args[1] == "patch" || args[1] == "patch-noise") {
      let file = @lib.Hdf5File::parse(@fs.read_file_to_bytes(args[2]))
      let output = if args[1] == "relocate" {
        file.relocate_dataset_storage(args[3])
      } else {
        let count = file.read_dataset_metadata(args[3]).dataspace.element_count()
        let mut state = 123456789U
        let values = Array::makei(count.to_int(), _index => {
          if args[1] == "patch-noise" {
            state = state ^ (state << 13)
            state = state ^ (state >> 17)
            state = state ^ (state << 5)
            state.reinterpret_as_int()
          } else {
            42
          }
        })
        file.patch_dataset_values(args[3], Int32Values(values))
      }
      @fs.write_bytes_to_file(args[4], output)
      return
    }
    for index in 1..<args.length() {
      let file = @lib.Hdf5File::parse(@fs.read_file_to_bytes(args[index]))
      let managers = file.free_space_managers()
      let mut total = 0UL
      for manager in managers {
        total = total + manager.total_space
      }
      println("file " + args[index] + " " + total.to_string())
      for manager in managers {
        for section in manager.sections {
          println(section.address.to_string() + " " + section.size.to_string())
        }
      }
    }
  } catch {
    error => {
      eprintln(error.to_string())
      @sys.exit(1)
    }
  }
}
