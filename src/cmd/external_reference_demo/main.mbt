///|
fn external_reference_demo_target() -> Bytes raise Error {
  let writer = @lib.Hdf5Writer::new()
  writer.add_dataset(
    "/matrix",
    [2UL, 3UL],
    Int64Values([0L, 1L, 2L, 3L, 4L, 5L]),
  )
  writer.add_group("/group")
  writer.add_dataset("/group/child", [1UL], Int64Values([9L]))
  writer.add_attribute("/matrix", "unit", [], Int64Values([7L]))
  writer.add_attribute("/matrix", "label", [], FixedStringValues(["matrix"], 8))
  writer.encode()
}

///|
fn external_reference_demo_source(target : Bytes) -> Bytes raise Error {
  let writer = @lib.Hdf5Writer::new()
  writer.add_external_source("external_reference_target.h5", target)
  let object_type = @lib.Datatype::write_version4_object_reference()
  let region_type = @lib.Datatype::write_version4_region_reference()
  let attribute_type = @lib.Datatype::write_version4_attribute_reference()
  writer.add_dataset(
    "/object_refs",
    [3UL],
    StructuredValues(object_type, [
      @lib.ExternalObjectReference("external_reference_target.h5", "/matrix"),
      @lib.ExternalObjectReference("external_reference_target.h5", "/group"),
      @lib.ExternalObjectReference(
        "external_reference_target.h5", "/group/child",
      ),
    ]),
  )
  writer.add_dataset(
    "/region_refs",
    [2UL],
    StructuredValues(region_type, [
      @lib.ExternalRegionReference(
        "external_reference_target.h5",
        "/matrix",
        @lib.RegionPoints([[1UL, 2UL]]),
      ),
      @lib.ExternalRegionReference(
        "external_reference_target.h5",
        "/matrix",
        @lib.RegionAll,
      ),
    ]),
  )
  writer.add_dataset(
    "/attribute_refs",
    [1UL],
    StructuredValues(attribute_type, [
      @lib.ExternalAttributeReference(
        "external_reference_target.h5", "/matrix", "unit",
      ),
    ]),
  )
  writer.add_attribute(
    "/object_refs",
    "remote",
    [],
    StructuredValues(attribute_type, [
      @lib.ExternalAttributeReference(
        "external_reference_target.h5", "/matrix", "unit",
      ),
    ]),
  )
  writer.encode()
}

///|
fn external_reference_demo_patch(
  source : Bytes,
  target : Bytes,
) -> Bytes raise Error {
  let object_type = @lib.Datatype::write_version4_object_reference()
  let region_type = @lib.Datatype::write_version4_region_reference()
  let attribute_type = @lib.Datatype::write_version4_attribute_reference()
  let file = @lib.Hdf5File::parse(source).with_external_source(
    "external_reference_target.h5", target,
  )
  let patched_objects = file.patch_dataset_values(
    "/object_refs",
    StructuredValues(object_type, [
      @lib.ExternalObjectReference(
        "external_reference_target.h5", "/group/child",
      ),
      @lib.ExternalObjectReference("external_reference_target.h5", "/matrix"),
      @lib.ExternalObjectReference("external_reference_target.h5", "/group"),
    ]),
  )
  let patched_regions = @lib.Hdf5File::parse(patched_objects)
    .with_external_source("external_reference_target.h5", target)
    .patch_dataset_values(
      "/region_refs",
      StructuredValues(region_type, [
        @lib.ExternalRegionReference(
          "external_reference_target.h5",
          "/matrix",
          @lib.RegionPoints([[0UL, 0UL], [1UL, 2UL]]),
        ),
        @lib.ExternalRegionReference(
          "external_reference_target.h5",
          "/matrix",
          @lib.RegionAll,
        ),
      ]),
    )
  @lib.Hdf5File::parse(patched_regions)
  .with_external_source("external_reference_target.h5", target)
  .patch_attribute_values(
    "/object_refs",
    "remote",
    StructuredValues(attribute_type, [
      @lib.ExternalAttributeReference(
        "external_reference_target.h5", "/matrix", "label",
      ),
    ]),
  )
}

///|
fn run(args : Array[String]) -> Int32 {
  if args.length() != 4 {
    eprintln(
      "usage: moon run src/cmd/external_reference_demo -- <target.h5> <source.h5> <patched.h5>",
    )
    return 2
  }
  try {
    let target = external_reference_demo_target()
    let source = external_reference_demo_source(target)
    let patched = external_reference_demo_patch(source, target)
    @fs.write_bytes_to_file(args[1], target)
    @fs.write_bytes_to_file(args[2], source)
    @fs.write_bytes_to_file(args[3], patched)
    println("wrote external reference demo files")
    0
  } catch {
    error => {
      eprintln("external reference demo error: " + error.to_string())
      1
    }
  }
}

///|
fn main {
  @sys.exit(run(@env.args()))
}
