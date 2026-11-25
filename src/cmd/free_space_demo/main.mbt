///|
fn main {
  let args = @env.args()
  try {
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
