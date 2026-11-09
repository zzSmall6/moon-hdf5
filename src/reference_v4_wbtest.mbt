///|
fn version4_attribute_integer_values(
  attribute : Attribute,
) -> Array[Int64] raise Error {
  match attribute.values {
    Some(Integers(values)) => values
    _ => fail("expected integer attribute-reference values")
  }
}

///|
test "version-4 object references resolve across layouts and attributes" {
  let file = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/references_v4.h5"),
  )
  let references = reference_value_bytes(file.read_dataset("/object_refs"))
  assert_eq(references.length(), 7)
  assert_eq(references[0].length(), 18)

  let expected = [
    ("/target", Dataset),
    ("/group", Group),
    ("/group/child", Dataset),
    ("/", Group),
    ("/named_i4", Datatype),
    ("/matrix", Dataset),
  ]
  for index, (path, kind) in expected {
    let node = file.resolve(path)
    assert_eq(
      file.resolve_object_reference(references[index]),
      Some({ address: node.address, kind, path: Some(path), }),
    )
  }
  assert_eq(file.resolve_object_reference(references[6]), None)

  let attributes = file.attributes("/target")
  let child_ref = match
    attributes.search_by(attribute => attribute.name == "child_ref") {
    Some(index) => attributes[index]
    None => raise InvalidField("target.child_ref attribute is missing")
  }
  let child = file.resolve("/group/child")
  match child_ref.values {
    Some(Structured([Reference(bytes)])) =>
      assert_eq(
        file.resolve_object_reference(bytes),
        Some({
          address: child.address,
          kind: Dataset,
          path: Some("/group/child"),
        }),
      )
    _ => raise InvalidField("target.child_ref has an unexpected value")
  }

  let region_as_object = try {
    let region = reference_value_bytes(file.read_dataset("/region_refs"))[0]
    file.resolve_object_reference(region) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    region_as_object,
    Some(InvalidArgument("region references require resolve_region_reference")),
  )
}

///|
test "version-4 region references decode selections and values" {
  let file = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/references_v4.h5"),
  )
  let references = reference_value_bytes(file.read_dataset("/region_refs"))
  assert_eq(references.length(), 6)
  assert_eq(references[0].length(), 18)

  let slice = resolve_region_or_fail(file, references[0], "version-4 slice")
  assert_eq(slice.path, Some("/target"))
  assert_region_hyperslab(slice.selection, [1UL], [2UL], [2UL], [1UL])
  check_region_values(file, references[0], [2UL], [1L, 3L])

  let points = resolve_region_or_fail(file, references[1], "version-4 points")
  match points.selection {
    RegionPoints(actual) => assert_eq(actual, [[4UL], [0UL], [3UL]])
    _ => fail("expected version-4 point selection")
  }
  check_region_values(file, references[1], [3UL], [4L, 0L, 3L])

  let all = resolve_region_or_fail(file, references[2], "version-4 all")
  assert_eq(all.selection, RegionAll)
  check_region_values(file, references[2], [6UL], [0L, 1L, 2L, 3L, 4L, 5L])

  let empty = resolve_region_or_fail(file, references[3], "version-4 none")
  assert_eq(empty.selection, RegionEmpty)
  check_region_values(file, references[3], [0UL], [])

  let matrix = resolve_region_or_fail(file, references[4], "version-4 matrix")
  assert_eq(matrix.path, Some("/matrix"))
  assert_region_hyperslab(
    matrix.selection,
    [0UL, 0UL],
    [1UL, 3UL],
    [1UL, 2UL],
    [4UL, 1UL],
  )
  check_region_values(file, references[4], [4UL, 2UL], [
    0L, 3L, 5L, 8L, 10L, 13L, 15L, 18L,
  ])
  assert_true(file.resolve_region_reference(references[5]) is None)

  let object_as_region = try {
    let object = reference_value_bytes(file.read_dataset("/object_refs"))[0]
    file.resolve_region_reference(object) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    object_as_region,
    Some(InvalidArgument("object references require resolve_object_reference")),
  )
}

///|
test "version-4 attribute references resolve names and values" {
  let file = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/references_v4.h5"),
  )
  let references = reference_value_bytes(file.read_dataset("/attribute_refs"))
  assert_eq(references.length(), 3)

  let target = file.resolve("/target")
  assert_eq(
    file.resolve_attribute_reference(references[0]),
    Some({
      address: target.address,
      kind: Dataset,
      path: Some("/target"),
      attribute_name: "unit",
    }),
  )
  let attribute = match file.read_attribute_reference(references[0]) {
    Some(attribute) => attribute
    None => fail("expected a non-null attribute reference")
  }
  assert_eq(attribute.name, "unit")
  assert_eq(attribute.dataspace.kind, Scalar)
  assert_eq(version4_attribute_integer_values(attribute), [7L])

  let matrix = file.resolve("/matrix")
  let matrix_attribute = match file.read_attribute_reference(references[1]) {
    Some(attribute) => attribute
    None => fail("expected a non-null matrix attribute reference")
  }
  assert_eq(matrix_attribute.name, "unit")
  assert_eq(version4_attribute_integer_values(matrix_attribute), [11L])
  assert_eq(references[1].length(), 18)
  assert_eq(
    file.resolve_attribute_reference(references[1]),
    Some({
      address: matrix.address,
      kind: Dataset,
      path: Some("/matrix"),
      attribute_name: "unit",
    }),
  )

  assert_true(file.resolve_attribute_reference(references[2]) is None)
  assert_true(file.read_attribute_reference(references[2]) is None)

  let object_as_attribute = try {
    let object = reference_value_bytes(file.read_dataset("/object_refs"))[0]
    file.resolve_attribute_reference(object) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    object_as_attribute,
    Some(InvalidArgument("object references require resolve_object_reference")),
  )
}

///|
fn replace_reference_byte(
  reference : Bytes,
  index : Int32,
  value : Byte,
) -> Bytes {
  let bytes = reference.to_array()
  bytes[index] = value
  Bytes::from_array(bytes)
}

///|
fn version4_error_of(operation : () -> Unit raise Hdf5Error) -> Hdf5Error? {
  try {
    operation()
    None
  } catch {
    error => Some(error)
  }
}

///|
test "version-4 references reject malformed slots and flags" {
  let file = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/references_v4.h5"),
  )
  let objects = reference_value_bytes(file.read_dataset("/object_refs"))
  let regions = reference_value_bytes(file.read_dataset("/region_refs"))
  let attributes = reference_value_bytes(file.read_dataset("/attribute_refs"))

  let short : Array[Byte] = []
  for index in 0..<17 {
    short.push(objects[0][index])
  }
  let short_error = version4_error_of(() => {
    file.resolve_object_reference(Bytes::from_array(short)) |> ignore
  })
  assert_true(short_error is Some(InvalidField(_)))

  let long = objects[0].to_array()
  long.push(b'\x00')
  let long_error = version4_error_of(() => {
    file.resolve_object_reference(Bytes::from_array(long)) |> ignore
  })
  assert_true(long_error is Some(InvalidField(_)))

  let bad_kind = replace_reference_byte(objects[0], 0, b'\x05')
  assert_eq(
    version4_error_of(() => file.resolve_object_reference(bad_kind) |> ignore),
    Some(InvalidField("invalid version-4 reference type")),
  )

  let reserved_flags = replace_reference_byte(objects[0], 1, b'\x02')
  assert_eq(
    version4_error_of(() => {
      file.resolve_object_reference(reserved_flags) |> ignore
    }),
    Some(InvalidField("version-4 reference has unsupported flags")),
  )

  let invalid_null = replace_reference_byte(objects[6], 2, b'\x01')
  assert_eq(
    version4_error_of(() => {
      file.resolve_object_reference(invalid_null) |> ignore
    }),
    Some(InvalidField("invalid null version-4 reference")),
  )

  let external_object = replace_reference_byte(objects[0], 1, b'\x01')
  let external_region = replace_reference_byte(regions[0], 1, b'\x01')
  let external_attribute = replace_reference_byte(attributes[0], 1, b'\x01')
  let external_error = UnsupportedFeature(
    "version-4 references to external files are not supported",
  )
  assert_eq(
    version4_error_of(() => {
      file.resolve_object_reference(external_object) |> ignore
    }),
    Some(external_error),
  )
  assert_eq(
    version4_error_of(() => {
      file.resolve_region_reference(external_region) |> ignore
    }),
    Some(external_error),
  )
  assert_eq(
    version4_error_of(() => {
      file.resolve_attribute_reference(external_attribute) |> ignore
    }),
    Some(external_error),
  )
}

///|
test "version-4 references reject bad tokens and dangling targets" {
  let file = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/references_v4.h5"),
  )
  let objects = reference_value_bytes(file.read_dataset("/object_refs"))
  let regions = reference_value_bytes(file.read_dataset("/region_refs"))
  let attributes = reference_value_bytes(file.read_dataset("/attribute_refs"))

  let object_token = replace_reference_byte(objects[0], 2, b'\x04')
  assert_eq(
    version4_error_of(() => {
      file.resolve_object_reference(object_token) |> ignore
    }),
    Some(
      UnsupportedFeature(
        "version-4 reference token size 4 does not match the file address size",
      ),
    ),
  )

  let region_size = replace_reference_byte(regions[0], 2, b'\x38')
  assert_eq(
    version4_error_of(() => file.resolve_region_reference(region_size) |> ignore),
    Some(InvalidField("version-4 reference blob size does not match the heap")),
  )

  let attribute_size = replace_reference_byte(attributes[0], 2, b'\x0e')
  assert_eq(
    version4_error_of(() => {
      file.resolve_attribute_reference(attribute_size) |> ignore
    }),
    Some(InvalidField("version-4 reference blob size does not match the heap")),
  )

  let invalid_heap = regions[0].to_array()
  for index in 14..<18 {
    invalid_heap[index] = b'\x00'
  }
  assert_eq(
    version4_error_of(() => {
      file.resolve_region_reference(Bytes::from_array(invalid_heap)) |> ignore
    }),
    Some(InvalidField("invalid version-4 reference heap identifier")),
  )

  let dangling = objects[0].to_array()
  for index in 3..<11 {
    dangling[index] = b'\xff'
  }
  assert_true(
    version4_error_of(() => {
      file.resolve_object_reference(Bytes::from_array(dangling)) |> ignore
    })
    is Some(InvalidField(_)),
  )
}

///|
test "version-4 reference type dispatch is explicit" {
  let file = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/references_v4.h5"),
  )
  let objects = reference_value_bytes(file.read_dataset("/object_refs"))
  let regions = reference_value_bytes(file.read_dataset("/region_refs"))
  let attributes = reference_value_bytes(file.read_dataset("/attribute_refs"))

  assert_eq(
    version4_error_of(() => {
      file.resolve_attribute_reference(objects[0]) |> ignore
    }),
    Some(InvalidArgument("object references require resolve_object_reference")),
  )
  assert_eq(
    version4_error_of(() => {
      file.resolve_attribute_reference(regions[0]) |> ignore
    }),
    Some(InvalidArgument("region references require resolve_region_reference")),
  )
  assert_eq(
    version4_error_of(() => {
      file.resolve_object_reference(attributes[0]) |> ignore
    }),
    Some(
      InvalidArgument(
        "attribute references require resolve_attribute_reference",
      ),
    ),
  )
  assert_eq(
    version4_error_of(() => {
      file.resolve_region_reference(attributes[0]) |> ignore
    }),
    Some(
      InvalidArgument(
        "attribute references require resolve_attribute_reference",
      ),
    ),
  )
}
