///|
fn size_matrix_integers(dataset : Dataset) -> Array[Int64] raise Hdf5Error {
  match dataset.values {
    Integers(values) => values
    _ => raise InvalidField("expected size-matrix integer values")
  }
}

///|
test "address and length width matrix reads contiguous and chunked data" {
  let fixtures = [
    ("testdata/size_22.h5", 2, 2),
    ("testdata/size_24.h5", 2, 4),
    ("testdata/size_28.h5", 2, 8),
    ("testdata/size_42.h5", 4, 2),
    ("testdata/size_44.h5", 4, 4),
    ("testdata/size_48.h5", 4, 8),
    ("testdata/size_82.h5", 8, 2),
    ("testdata/size_84.h5", 8, 4),
    ("testdata/size_88.h5", 8, 8),
  ]
  let expected = [0L, 1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, 10L, 11L]
  for (path, offset_size, length_size) in fixtures {
    let file = Hdf5File::parse(@fs.read_file_to_bytes(path))
    assert_eq(file.superblock.offset_size, offset_size)
    assert_eq(file.superblock.length_size, length_size)
    assert_eq(size_matrix_integers(file.read_dataset("/g/data")), expected)
    assert_eq(size_matrix_integers(file.read_dataset("/g/chunked")), expected)
    let attributes = file.attributes("/g/data")
    assert_eq(attributes.length(), 1)
    assert_eq(attributes[0].name, "answer")
    match attributes[0].values {
      Some(Integers(values)) => assert_eq(values, [42L])
      _ => raise InvalidField("expected size-matrix answer attribute")
    }
    let group_attributes = file.attributes("/g")
    assert_eq(group_attributes.length(), 1)
    assert_eq(group_attributes[0].name, "label")
    match group_attributes[0].values {
      Some(Strings(values)) => assert_eq(values, ["matrix"])
      _ => raise InvalidField("expected size-matrix label attribute")
    }
  }
}
