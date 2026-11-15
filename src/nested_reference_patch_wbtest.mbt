///|
fn nested_reference_record_type() -> Datatype raise Hdf5Error {
  Datatype::write_compound([
    ("label", Datatype::write_fixed_string(8)),
    ("obj", Datatype::write_object_reference()),
    (
      "obj_pair",
      Datatype::write_array(Datatype::write_object_reference(), [2UL]),
    ),
    ("region", Datatype::write_region_reference()),
    ("vobj", Datatype::write_version4_object_reference()),
    (
      "vpair",
      Datatype::write_array(Datatype::write_version4_object_reference(), [2UL]),
    ),
    ("vregion", Datatype::write_version4_region_reference()),
    ("vattr", Datatype::write_version4_attribute_reference()),
  ])
}

///|
fn nested_reference_record(
  label : String,
  obj : String?,
  obj_pair : Array[String?],
  region : Hdf5RegionSelection,
  region_path : String?,
  vobj : String?,
  vpair : Array[String?],
  vregion_path : String?,
  vregion : Hdf5RegionSelection,
  attribute_path : String?,
  attribute_name : String,
) -> DatasetValue {
  Compound([
    { name: "label", value: Text(label), },
    { name: "obj", value: ObjectReference(obj), },
    {
      name: "obj_pair",
      value: Elements(obj_pair.map(path => ObjectReference(path))),
    },
    { name: "region", value: RegionReference(region_path, region), },
    { name: "vobj", value: ObjectReference(vobj), },
    {
      name: "vpair",
      value: Elements(vpair.map(path => ObjectReference(path))),
    },
    { name: "vregion", value: RegionReference(vregion_path, vregion), },
    {
      name: "vattr",
      value: AttributeReference(attribute_path, attribute_name),
    },
  ])
}

///|
fn nested_reference_fields(
  file : Hdf5File,
  path : String,
) -> Array[Array[CompoundFieldValue]] raise Hdf5Error {
  let values = match file.read_dataset(path).values {
    Structured(values) => values
    _ => raise InvalidField("expected structured nested reference values")
  }
  let records : Array[Array[CompoundFieldValue]] = []
  for value in values {
    match value {
      Compound(fields) => records.push(fields)
      _ => raise InvalidField("expected a nested reference compound")
    }
  }
  records
}

///|
fn nested_patch_error_of(operation : () -> Unit raise Hdf5Error) -> Hdf5Error? {
  try {
    operation()
    None
  } catch {
    error => Some(error)
  }
}

///|
test "nested references patch datasets and attributes" {
  let record_type = nested_reference_record_type()
  let writer = Hdf5Writer::new()
  writer.add_dataset("/target", [4UL], Int64Values([0L, 1L, 2L, 3L]))
  writer.add_dataset("/target2", [4UL], Int64Values([10L, 11L, 12L, 13L]))
  writer.add_group("/group")
  writer.add_attribute("/target", "unit", [], Int64Values([7L]))
  writer.add_attribute("/target2", "unit", [], Int64Values([11L]))

  let original = nested_reference_record(
    "original",
    Some("/target"),
    [Some("/group"), None],
    RegionPoints([[1UL]]),
    Some("/target"),
    Some("/group"),
    [Some("/target"), None],
    Some("/target"),
    RegionAll,
    Some("/target"),
    "unit",
  )
  let empty = nested_reference_record(
    "empty",
    None,
    [None, None],
    RegionEmpty,
    None,
    None,
    [None, None],
    None,
    RegionEmpty,
    None,
    "",
  )
  writer.add_dataset(
    "/records",
    [2UL],
    StructuredValues(record_type, [original, empty]),
  )
  writer.add_attribute(
    "/records",
    "prototype",
    [],
    StructuredValues(record_type, [original]),
  )

  let file = Hdf5File::parse(writer.encode())
  let patched_record = nested_reference_record(
    "patched",
    Some("/target2"),
    [Some("/group"), None],
    RegionPoints([[3UL]]),
    Some("/target2"),
    Some("/target"),
    [Some("/target2"), None],
    Some("/target2"),
    RegionAll,
    Some("/target2"),
    "unit",
  )
  let patched = Hdf5File::parse(
    file.patch_dataset_values(
      "/records",
      StructuredValues(record_type, [patched_record, empty]),
    ),
  )
  let records = nested_reference_fields(patched, "/records")
  assert_eq(records.length(), 2)
  assert_eq(records[0][0].value, Text("patched"))
  match records[0][1].value {
    Reference(bytes) =>
      assert_eq(
        patched.resolve_object_reference(bytes),
        Some({
          address: patched.resolve("/target2").address,
          kind: Dataset,
          path: Some("/target2"),
        }),
      )
    _ => abort("expected patched nested object reference")
  }
  match records[0][2].value {
    Elements([Reference(group), Reference(null_ref)]) => {
      match patched.resolve_object_reference(group) {
        Some(target) => assert_eq(target.kind, Group)
        None => abort("expected nested group reference")
      }
      assert_true(patched.resolve_object_reference(null_ref) is None)
    }
    _ => abort("expected patched nested object-reference array")
  }
  match records[0][3].value {
    Reference(bytes) =>
      match patched.resolve_region_reference(bytes) {
        Some(region) => {
          assert_eq(region.path, Some("/target2"))
          assert_eq(region.selection, RegionPoints([[3UL]]))
        }
        None => abort("expected patched nested region reference")
      }
    _ => abort("expected nested region reference")
  }
  match records[0][4].value {
    Reference(bytes) =>
      match patched.resolve_object_reference(bytes) {
        Some(target) => assert_eq(target.path, Some("/target"))
        None => abort("expected nested version-4 object reference")
      }
    _ => abort("expected nested version-4 object reference")
  }
  match records[0][5].value {
    Elements([Reference(first), Reference(null_ref)]) => {
      match patched.resolve_object_reference(first) {
        Some(target) => assert_eq(target.path, Some("/target2"))
        None => abort("expected nested version-4 object-reference array")
      }
      assert_true(patched.resolve_object_reference(null_ref) is None)
    }
    _ => abort("expected nested version-4 object-reference array")
  }
  match records[0][6].value {
    Reference(bytes) =>
      match patched.resolve_region_reference(bytes) {
        Some(region) => {
          assert_eq(region.path, Some("/target2"))
          assert_eq(region.selection, RegionAll)
        }
        None => abort("expected patched nested version-4 region reference")
      }
    _ => abort("expected nested version-4 region reference")
  }
  match records[0][7].value {
    Reference(bytes) =>
      match patched.resolve_attribute_reference(bytes) {
        Some(target) => assert_eq(target.attribute_name, "unit")
        None => abort("expected nested version-4 attribute reference")
      }
    _ => abort("expected nested version-4 attribute reference")
  }
  let empty_record = records[1]
  match empty_record[1].value {
    Reference(bytes) =>
      assert_true(patched.resolve_object_reference(bytes) is None)
    _ => abort("expected null nested object reference")
  }
  match empty_record[2].value {
    Elements([Reference(first), Reference(second)]) => {
      assert_true(patched.resolve_object_reference(first) is None)
      assert_true(patched.resolve_object_reference(second) is None)
    }
    _ => abort("expected null nested object-reference array")
  }
  match empty_record[3].value {
    Reference(bytes) =>
      assert_true(patched.resolve_region_reference(bytes) is None)
    _ => abort("expected null nested region reference")
  }
  match empty_record[4].value {
    Reference(bytes) =>
      assert_true(patched.resolve_object_reference(bytes) is None)
    _ => abort("expected null nested version-4 object reference")
  }
  match empty_record[5].value {
    Elements([Reference(first), Reference(second)]) => {
      assert_true(patched.resolve_object_reference(first) is None)
      assert_true(patched.resolve_object_reference(second) is None)
    }
    _ => abort("expected null nested version-4 object-reference array")
  }
  match empty_record[6].value {
    Reference(bytes) =>
      assert_true(patched.resolve_region_reference(bytes) is None)
    _ => abort("expected null nested version-4 region reference")
  }
  match empty_record[7].value {
    Reference(bytes) =>
      assert_true(patched.resolve_attribute_reference(bytes) is None)
    _ => abort("expected null nested version-4 attribute reference")
  }

  let attribute_record = nested_reference_record(
    "attr",
    Some("/group"),
    [Some("/target2"), Some("/group")],
    RegionPoints([[2UL]]),
    Some("/target2"),
    Some("/target2"),
    [Some("/target2"), Some("/group")],
    Some("/target2"),
    RegionHyperslab([0UL], [1UL], [2UL], [1UL]),
    Some("/target"),
    "unit",
  )
  let attribute_bytes = file.patch_attribute_values(
    "/records",
    "prototype",
    StructuredValues(record_type, [attribute_record]),
  ) catch {
    error => abort("attribute patch failed: " + error.message())
  }
  let attribute_patched = Hdf5File::parse(attribute_bytes) catch {
    error => abort("attribute parse failed: " + error.message())
  }
  let attributes = attribute_patched.attributes("/records")
  let prototype = match
    attributes.search_by(attribute => attribute.name == "prototype") {
    Some(index) => attributes[index]
    None => abort("missing patched prototype attribute")
  }
  match prototype.values {
    Some(Structured([Compound(fields)])) => {
      assert_eq(fields[0].value, Text("attr"))
      match fields[1].value {
        Reference(bytes) =>
          match attribute_patched.resolve_object_reference(bytes) {
            Some(target) => assert_eq(target.kind, Group)
            None => abort("expected patched attribute object reference")
          }
        _ => abort("expected patched attribute object reference")
      }
      match fields[3].value {
        Reference(bytes) =>
          match attribute_patched.resolve_region_reference(bytes) {
            Some(region) => assert_eq(region.selection, RegionPoints([[2UL]]))
            None => abort("expected patched attribute region reference")
          }
        _ => abort("expected patched attribute region reference")
      }
      match fields[7].value {
        Reference(bytes) =>
          match attribute_patched.resolve_attribute_reference(bytes) {
            Some(target) => assert_eq(target.attribute_name, "unit")
            None => abort("expected patched attribute reference")
          }
        _ => abort("expected patched attribute reference")
      }
    }
    _ => abort("expected structured patched attribute")
  }
}

///|
test "nested reference patching validates targets and supports reference sequences" {
  let record_type = nested_reference_record_type()
  let writer = Hdf5Writer::new()
  writer.add_dataset("/target", [4UL], Int64Values([0L, 1L, 2L, 3L]))
  writer.add_dataset(
    "/records",
    [1UL],
    StructuredValues(record_type, [
      nested_reference_record(
        "original",
        Some("/target"),
        [None, None],
        RegionAll,
        Some("/target"),
        None,
        [None, None],
        Some("/target"),
        RegionEmpty,
        None,
        "",
      ),
    ]),
  )
  let file = Hdf5File::parse(writer.encode())
  let missing = nested_reference_record(
    "missing",
    Some("/missing"),
    [None, None],
    RegionAll,
    Some("/target"),
    None,
    [None, None],
    Some("/target"),
    RegionEmpty,
    None,
    "",
  )
  let error = nested_patch_error_of(() => {
    file.patch_dataset_values(
      "/records",
      StructuredValues(record_type, [missing]),
    )
    |> ignore
  })
  assert_true(error is Some(InvalidField(_)))

  let sequence_type = Datatype::write_vlen_sequence(
    Datatype::write_object_reference(),
  )
  writer.add_dataset(
    "/sequence_records",
    [1UL],
    StructuredValues(sequence_type, [
      Sequence([ObjectReference(Some("/target"))]),
    ]),
  )
  let sequence_file = Hdf5File::parse(writer.encode())
  let sequence_patched = Hdf5File::parse(
    sequence_file.patch_dataset_values(
      "/sequence_records",
      StructuredValues(sequence_type, [
        Sequence([ObjectReference(Some("/target"))]),
      ]),
    ),
  )
  match sequence_patched.read_dataset("/sequence_records").values {
    Structured([Sequence([Reference(bytes)])]) =>
      assert_eq(
        sequence_patched
        .resolve_object_reference(bytes)
        .map(value => value.path),
        Some(Some("/target")),
      )
    _ => abort("expected a patched reference sequence")
  }
}

///|
test "four-byte nested legacy references can be patched" {
  let file = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/nested_reference_32.h5"),
  )
  assert_eq(file.superblock.offset_size, 4)
  let datatype = file.read_dataset("/records").datatype
  assert_eq(datatype.size, 20)
  assert_eq(datatype.compound_members.map(field => field.name), [
    "label", "obj", "region",
  ])
  assert_eq(datatype.compound_members[1].datatype.size, 4)
  assert_eq(datatype.compound_members[2].datatype.size, 8)

  let record = Compound([
    { name: "label", value: Text("patched"), },
    { name: "obj", value: ObjectReference(Some("/target2")), },
    {
      name: "region",
      value: RegionReference(Some("/target2"), RegionPoints([[2UL]])),
    },
  ])
  let empty = Compound([
    { name: "label", value: Text("empty"), },
    { name: "obj", value: ObjectReference(None), },
    { name: "region", value: RegionReference(None, RegionEmpty), },
  ])
  let patched_bytes = file.patch_dataset_values(
    "/records",
    StructuredValues(datatype, [record, empty]),
  ) catch {
    error => abort("four-byte dataset patch failed: " + error.message())
  }
  let patched = Hdf5File::parse(patched_bytes)
  let records = nested_reference_fields(patched, "/records")
  assert_eq(records[0][0].value, Text("patched"))
  match records[0][1].value {
    Reference(bytes) => {
      assert_eq(bytes.length(), 4)
      assert_eq(
        patched.resolve_object_reference(bytes).map(target => target.path),
        Some(Some("/target2")),
      )
    }
    _ => abort("expected a four-byte nested object reference")
  }
  match records[0][2].value {
    Reference(bytes) => {
      assert_eq(bytes.length(), 8)
      match patched.resolve_region_reference(bytes) {
        Some(region) => {
          assert_eq(region.path, Some("/target2"))
          assert_eq(region.selection, RegionPoints([[2UL]]))
        }
        None => abort("expected a four-byte nested region reference")
      }
    }
    _ => abort("expected a four-byte nested region reference")
  }
  match records[1][1].value {
    Reference(bytes) =>
      assert_true(patched.resolve_object_reference(bytes) is None)
    _ => abort("expected a null four-byte object reference")
  }
  match records[1][2].value {
    Reference(bytes) =>
      assert_true(patched.resolve_region_reference(bytes) is None)
    _ => abort("expected a null four-byte region reference")
  }

  let attribute_record = Compound([
    { name: "label", value: Text("attr"), },
    { name: "obj", value: ObjectReference(Some("/group")), },
    {
      name: "region",
      value: RegionReference(
        Some("/target2"),
        RegionHyperslab([0UL], [1UL], [2UL], [1UL]),
      ),
    },
  ])
  let attribute_patched = Hdf5File::parse(
    file.patch_attribute_values(
      "/records",
      "prototype",
      StructuredValues(datatype, [attribute_record]),
    ),
  )
  let attributes = attribute_patched.attributes("/records")
  let prototype = match
    attributes.search_by(attribute => attribute.name == "prototype") {
    Some(index) => attributes[index]
    None => abort("missing four-byte prototype attribute")
  }
  match prototype.values {
    Some(Structured([Compound(fields)])) => {
      assert_eq(fields[0].value, Text("attr"))
      match fields[1].value {
        Reference(bytes) => {
          assert_eq(bytes.length(), 4)
          assert_eq(
            attribute_patched
            .resolve_object_reference(bytes)
            .map(target => target.kind),
            Some(Group),
          )
        }
        _ => abort("expected a four-byte attribute object reference")
      }
      match fields[2].value {
        Reference(bytes) => {
          assert_eq(bytes.length(), 8)
          match attribute_patched.resolve_region_reference(bytes) {
            Some(region) =>
              assert_eq(
                region.selection,
                RegionHyperslab([0UL], [1UL], [2UL], [1UL]),
              )
            None => abort("expected a four-byte attribute region reference")
          }
        }
        _ => abort("expected a four-byte attribute region reference")
      }
    }
    _ => abort("expected a four-byte structured attribute")
  }
}

///|
test "nested reference datatype validation accepts four-byte slot widths" {
  validate_patch_nested_reference_datatype(writer_datatype(7, 4), 4)
  validate_patch_nested_reference_datatype(
    writer_datatype(7, 8, reference_kind=1),
    4,
  )
  for kind in [2, 3, 4] {
    validate_patch_nested_reference_datatype(
      writer_datatype(7, 14, version=4, reference_kind=kind),
      4,
    )
  }
  let wrong_size = nested_patch_error_of(() => {
    validate_patch_nested_reference_datatype(
      writer_datatype(7, 18, version=4, reference_kind=2),
      4,
    )
  })
  assert_true(wrong_size is Some(UnsupportedFeature(_)))
}

///|
test "references inside variable-length sequences can be patched" {
  let legacy_type = Datatype::write_vlen_sequence(
    Datatype::write_object_reference(),
  )
  let v4_type = Datatype::write_vlen_sequence(
    Datatype::write_version4_object_reference(),
  )
  let event_base = Datatype::write_compound([
    ("label", Datatype::write_fixed_string(6)),
    ("target", Datatype::write_object_reference()),
  ])
  let event_type = Datatype::write_vlen_sequence(event_base)
  let writer = Hdf5Writer::new()
  writer.add_dataset("/target", [4UL], Int64Values([0L, 1L, 2L, 3L]))
  writer.add_dataset("/target2", [4UL], Int64Values([10L, 11L, 12L, 13L]))
  writer.add_group("/group")
  writer.add_dataset(
    "/legacy_history",
    [3UL],
    StructuredValues(legacy_type, [
      Sequence([ObjectReference(Some("/target")), ObjectReference(None)]),
      Sequence([]),
      Sequence([ObjectReference(Some("/group"))]),
    ]),
  )
  writer.add_dataset(
    "/v4_history",
    [2UL],
    StructuredValues(v4_type, [
      Sequence([ObjectReference(Some("/target"))]),
      Sequence([]),
    ]),
  )
  writer.add_dataset(
    "/events",
    [2UL],
    StructuredValues(event_type, [
      Sequence([
        Compound([
          { name: "label", value: Text("old"), },
          { name: "target", value: ObjectReference(Some("/target")), },
        ]),
      ]),
      Sequence([]),
    ]),
  )
  writer.add_attribute(
    "/legacy_history",
    "template",
    [],
    StructuredValues(legacy_type, [Sequence([ObjectReference(Some("/target"))])]),
  )
  let file = Hdf5File::parse(writer.encode())

  let legacy_file = Hdf5File::parse(
    file.patch_dataset_values(
      "/legacy_history",
      StructuredValues(legacy_type, [
        Sequence([
          ObjectReference(Some("/target2")),
          ObjectReference(None),
          ObjectReference(Some("/group")),
        ]),
        Sequence([]),
        Sequence([ObjectReference(Some("/target"))]),
      ]),
    ),
  )
  match legacy_file.read_dataset("/legacy_history").values {
    Structured(
      [
        Sequence([Reference(first), Reference(null_ref), Reference(group)]),
        Sequence([]),
        Sequence([Reference(target)]),
      ]
    ) => {
      assert_eq(
        legacy_file.resolve_object_reference(first).map(value => value.path),
        Some(Some("/target2")),
      )
      assert_true(legacy_file.resolve_object_reference(null_ref) is None)
      assert_eq(
        legacy_file.resolve_object_reference(group).map(value => value.kind),
        Some(Group),
      )
      assert_eq(
        legacy_file.resolve_object_reference(target).map(value => value.path),
        Some(Some("/target")),
      )
    }
    _ => abort("expected patched legacy reference sequence")
  }

  let v4_file = Hdf5File::parse(
    legacy_file.patch_dataset_values(
      "/v4_history",
      StructuredValues(v4_type, [
        Sequence([ObjectReference(Some("/target2")), ObjectReference(None)]),
        Sequence([ObjectReference(Some("/group"))]),
      ]),
    ),
  )
  match v4_file.read_dataset("/v4_history").values {
    Structured(
      [
        Sequence([Reference(first), Reference(null_ref)]),
        Sequence([Reference(group)]),
      ]
    ) => {
      assert_eq(
        v4_file.resolve_object_reference(first).map(value => value.path),
        Some(Some("/target2")),
      )
      assert_true(v4_file.resolve_object_reference(null_ref) is None)
      assert_eq(
        v4_file.resolve_object_reference(group).map(value => value.kind),
        Some(Group),
      )
    }
    _ => abort("expected patched version-4 reference sequence")
  }

  let event_file = Hdf5File::parse(
    v4_file.patch_dataset_values(
      "/events",
      StructuredValues(event_type, [
        Sequence([
          Compound([
            { name: "label", value: Text("new"), },
            { name: "target", value: ObjectReference(Some("/target2")), },
          ]),
        ]),
        Sequence([]),
      ]),
    ),
  )
  match event_file.read_dataset("/events").values {
    Structured(
      [
        Sequence(
          [
            Compound(
              [
                { name: "label", value: Text("new"), },
                { name: "target", value: Reference(bytes), },
              ]
            ),
          ]
        ),
        Sequence([]),
      ]
    ) =>
      assert_eq(
        event_file.resolve_object_reference(bytes).map(value => value.path),
        Some(Some("/target2")),
      )
    _ => abort("expected patched compound reference sequence")
  }

  let attribute_file = Hdf5File::parse(
    event_file.patch_attribute_values(
      "/legacy_history",
      "template",
      StructuredValues(legacy_type, [
        Sequence([
          ObjectReference(Some("/group")),
          ObjectReference(Some("/target2")),
        ]),
      ]),
    ),
  )
  let attributes = attribute_file.attributes("/legacy_history")
  let template = match
    attributes.search_by(attribute => attribute.name == "template") {
    Some(index) => attributes[index]
    None => abort("missing reference sequence attribute")
  }
  match template.values {
    Some(Structured([Sequence([Reference(group), Reference(target)])])) => {
      assert_eq(
        attribute_file.resolve_object_reference(group).map(value => value.kind),
        Some(Group),
      )
      assert_eq(
        attribute_file.resolve_object_reference(target).map(value => value.path),
        Some(Some("/target2")),
      )
    }
    _ => abort("expected patched reference sequence attribute")
  }

  let all_empty = Hdf5File::parse(
    attribute_file.patch_dataset_values(
      "/legacy_history",
      StructuredValues(legacy_type, [Sequence([]), Sequence([]), Sequence([])]),
    ),
  )
  match all_empty.read_dataset("/legacy_history").values {
    Structured([Sequence([]), Sequence([]), Sequence([])]) => ()
    _ => abort("expected all-empty reference sequences")
  }
}
