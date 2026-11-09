# moon-hdf5

Pure MoonBit HDF5 reader and new-file writer for the HDF5 1.14.6 common-file
profile. It has no native HDF5 dependency and runs on wasm and native targets.

moon-hdf5 implements a documented subset of HDF5. The
[compatibility matrix](docs/compatibility.md) is the v1.0 delivery boundary;
[design notes](docs/design.md) describe the implementation and
[audit notes](docs/audit.md) record verification. Public APIs are listed in
[`src/pkg.generated.mbti`](src/pkg.generated.mbti).

## Status

- `Hdf5File` reads existing files without modification.
- `Hdf5Writer` creates complete new files in memory.
- `Hdf5Resolver` follows registered external links and reads resolver-backed
  external raw data.
- `patch_contiguous_dataset` replaces the values of an allocated contiguous
  dataset and returns a new byte image; metadata editing and other layouts
  remain unsupported.

## Installation

Add the released package to a MoonBit module:

```sh
moon add Zz04321/moon-hdf5@0.1.0
moon update
```

The package targets `wasm` by default. To build this repository from source
instead, use the commands in [Quick Start](#quick-start).

## Capabilities

- Superblocks v0–v3, 4- and 8-byte addresses, compact and dense groups and
  attributes, hard/soft/external links, and every named hard-link path.
- Compact, contiguous and chunked datasets; v1 B-tree, single-chunk, implicit,
  fixed-array, extensible-array and v2 B-tree indexes.
- Deflate, shuffle, Fletcher32, integer scale-offset and f32/f64 decimal
  scale-offset, including per-chunk filter masks.
- Caller-registered decoders for third-party dataset chunk filters.
- Signed and unsigned 1–8-byte integers in either byte order, IEEE
  float16/32/64, normalized binary floats with implied MSB, strings, compound,
  enum, array, time, bitfield, opaque, sequence, reference and complex values.
- Hyperslab and ordered-point reads, metadata-only reads, bounded-memory chunk
  iteration, legacy and version-4 object/region references, and version-4
  attribute-reference reads.
- SOHM reading and optional writing: shared attributes, filter pipelines,
  datatype/dataspace fields, fills and named committed datatypes.
- New-file output for groups, datasets, attributes, links, committed datatypes,
  filters, fills, references, contiguous external raw-data slots and one or more
  unlimited dataset dimensions.
- Value replacement for fixed-size, unfiltered contiguous datasets in a
  bytes-backed file.

## Quick Start

Install the [MoonBit toolchain](https://www.moonbitlang.com/download); version
0.10.19 or newer is required. The native target also needs a C compiler.

```sh
git clone https://github.com/zzSmall6/moon-hdf5.git
cd moon-hdf5
moon update
moon run src/cmd/main --target native -- testdata/basic_v0.h5
```

## Reading

Imports for the examples below are in
[`src/cmd/main/moon.pkg`](src/cmd/main/moon.pkg).

```moonbit nocheck
///|
let bytes = @fs.read_file_to_bytes("data.h5")
let file = @lib.Hdf5File::parse(bytes)

let metadata = file.read_dataset_metadata("/group/values")
let values = file.read_dataset("/group/values")
let slice = file.read_dataset_selection(
  "/matrix",
  @lib.Hyperslab([0UL, 0UL], [2UL, 2UL], [1UL, 2UL]),
)
let attributes = file.attributes("/group/values")
let nodes = file.walk()
```

`read_dataset_metadata` does not load values. Selections support compact,
contiguous and chunked storage, and only intersecting chunks are decoded.
`chunk_iterator` returns edge-clipped tiles with memory bounded by one tile plus
the intersecting storage chunk. For a large hyperslab or point selection, use
`selection_iterator` to process the result in bounded pages.

```moonbit nocheck
///|
let chunks = file.chunk_iterator("/matrix", [1UL, 2UL])
while chunks.next() is Some(chunk) {
  chunk.offset |> ignore
  chunk.values |> ignore
}

///|
let pages = file.selection_iterator(
  "/matrix",
  @lib.Hyperslab([0UL, 0UL], [1024UL, 1024UL], [1UL, 1UL]),
  4096,
)
while pages.next() is Some(page) {
  page.offset |> ignore
  page.values |> ignore
}

///|
let resolver = @lib.Hdf5Resolver::new(bytes)
resolver.add_source("other-file.h5", @fs.read_file_to_bytes("other-file.h5"))
let remote = resolver.read_dataset("/experiment/remote_counts")
```

Simple values use `Integers`, `UnsignedIntegers`, `Floats`, `Strings` or
`Enums`; other datatypes use recursive `Structured` values. Legacy and
version-4 references are returned as encoded bytes and can be resolved with
`resolve_object_reference`, `resolve_region_reference`,
`read_region_reference`, `resolve_attribute_reference` or
`read_attribute_reference`. Region and attribute targets must be in the same
file.

`Hdf5Resolver` follows hard, soft and external links across registered
`Hdf5Source` values, rejects cycles and reports unregistered targets. It also
reads contiguous, unfiltered external raw-data slots. `Hdf5File` intentionally
leaves external targets and external raw data unresolved.

Third-party dataset chunk filters use `Hdf5CustomFilter` with
`Hdf5File::with_custom_filter` or `Hdf5Resolver::add_custom_filter`. Decoders
receive the stored bytes, filter parameters, element width and expected decoded
size. Built-in filter IDs cannot be replaced, duplicate IDs are rejected, and a
decoder must return exactly the expected number of bytes.

## Writing

```moonbit nocheck
///|
let writer = @lib.Hdf5Writer::new()
writer.set_attribute_sharing(true)
writer.set_fill_sharing(true)
writer.set_filter_sharing(true)

let record = @lib.Datatype::write_compound([
  ("id", @lib.Datatype::write_integer(true, 4)),
  ("value", @lib.Datatype::write_float(8)),
])
writer.add_committed_datatype("/types/record", record)
writer.add_dataset(
  "/experiment/records",
  [2UL],
  @lib.StructuredValues(record, [
    @lib.Compound([
      { name: "id", value: @lib.SignedInteger(1L) },
      { name: "value", value: @lib.FloatingPoint(1.25) },
    ]),
    @lib.Compound([
      { name: "id", value: @lib.SignedInteger(2L) },
      { name: "value", value: @lib.FloatingPoint(-3.5) },
    ]),
  ]),
)
writer.add_attribute(
  "/experiment/records",
  "unit",
  [],
  @lib.FixedStringValues(["items"], 8),
)
writer.add_chunked_dataset(
  "/experiment/codes",
  [6UL],
  [3U],
  @lib.Int64Values([10L, 11L, 12L, 20L, 21L, 22L]),
  fill_value=@lib.Int64Values([-1L]),
  filters=[@lib.Shuffle, @lib.Deflate(6), @lib.Fletcher32],
)
writer.add_extensible_dataset(
  "/experiment/series",
  [0UL, 2UL],
  [None, Some(2UL)],
  [2U, 2U],
  @lib.Int64Values([]),
  fill_value=@lib.Int64Values([-1L]),
)
writer.append_dataset(
  "/experiment/series",
  [2UL, 2UL],
  @lib.Int64Values([1L, 2L, 3L, 4L]),
)
writer.add_soft_link("/experiment", "latest", "/experiment/series")

let output = writer.encode()
```

- `add_dataset` writes contiguous storage. `add_chunked_dataset`,
  `add_extensible_dataset` and `resize_dataset` cover finite and multi-unlimited
  chunked storage. One unlimited dimension uses an extensible-array index;
  multiple unlimited dimensions use a v2 B-tree. `append_dataset` appends a
  block at the trailing corner of every unlimited axis, growing each such axis
  by the block extent.
- `write_dataset_slab` writes a rectangular block at arbitrary coordinates.
  Finite datasets require the slab to fit their current shape; extensible
  datasets grow unlimited axes to the slab's trailing edge and fill gaps with
  the dataset fill value.
- `add_external_dataset` stores raw bytes in caller-managed external files. Slot
  sizes must match the dataset byte count; the writer does not create those
  files.
- Numeric output covers signed and unsigned 1–8-byte integers plus IEEE
  float16/32/64 in either byte order. `Float16Values` and
  `Datatype::write_float(2)` write half-precision values.
- Dense link and attribute collections support 9–65,535 records with encoded
  heap objects up to 65,535 bytes. Larger sets remain compact and use OCHK
  continuation chunks.
- Whole attributes, fills and encoded filter pipelines up to 65,535 bytes use one
  SOHM table and heap. Equal fills, attributes and pipelines reuse one heap ID.
  Committed datatypes use wrapper version 3 storage type 2 and are referenced by
  datasets and attributes with matching encoded datatypes.
- Variable-length strings and sequences share a global heap. Sequence bases must
  be fixed-size and cannot contain region references.

Version-4 references are opt-in per datatype; existing code keeps writing the
legacy 8-byte object and 12-byte region forms:

```moonbit nocheck
///|
let object_type = @lib.Datatype::write_version4_object_reference()
let region_type = @lib.Datatype::write_version4_region_reference()
let attribute_type = @lib.Datatype::write_version4_attribute_reference()
writer.add_dataset(
  "/experiment/object_refs",
  [2UL],
  @lib.StructuredValues(object_type, [
    @lib.ObjectReference(Some("/experiment/counts")),
    @lib.ObjectReference(None),
  ]),
)
writer.add_dataset(
  "/experiment/attribute_refs",
  [1UL],
  @lib.StructuredValues(attribute_type, [
    @lib.AttributeReference(Some("/experiment/records"), "unit"),
  ]),
)
```

The writer emits 18-byte reference slots in its 8-byte-address output. Region
and attribute payloads are stored in a dedicated global heap. References to
external files are rejected with `UnsupportedFeature`.

### Patching contiguous values

```moonbit nocheck
///|
let original = @fs.read_file_to_bytes("data.h5")
let file = @lib.Hdf5File::parse(original)
let patched = file.patch_contiguous_dataset(
  "/group/values",
  @lib.Int64Values([10L, 20L, 30L, 40L]),
)
@fs.write_bytes_to_file("patched.h5", patched)
```

`patch_contiguous_dataset` requires an allocated contiguous dataset in a
bytes-backed `Hdf5File`. The replacement must have exactly the declared shape
and a compatible fixed-size datatype; both little- and big-endian numeric
datasets are accepted. Object headers, attributes, links, checksums and all
bytes outside the dataset value region are preserved. Chunked, compact,
external, filtered and unallocated storage is rejected, as are variable-length
and reference datatypes. The operation returns a complete new byte image; it
does not modify the input or resize/relayout the dataset.

Run the larger example and verify it independently with h5py:

```sh
moon run src/cmd/write_demo --target native -- output.h5
python scripts/verify_writer_output.py output.h5
```

## Command Line

```sh
moon run src/cmd/main -- inspect testdata/basic_v0.h5
moon run src/cmd/main -- metadata testdata/basic_v0.h5 /matrix
moon run src/cmd/main -- stats testdata/basic_v0.h5 /matrix
moon run src/cmd/main -- read testdata/basic_v0.h5 /matrix 0,0 2,2 1,2
moon run src/cmd/main -- read testdata/basic_v0.h5 /matrix --points "1,2;0,1"
```

Coordinates are zero-based. Hyperslabs use comma-separated `start`, `count` and
`stride`; point lists use semicolons. Exit status is 0 for success, 1 for data
errors and 2 for invalid arguments.

## Limitations

moon-hdf5 is a documented subset, not a drop-in replacement for libhdf5. The
[compatibility matrix](docs/compatibility.md) is the authoritative boundary;
the items below summarize the main user-visible gaps.

- **Memory and streaming:** `Hdf5File::parse` requires the complete file bytes.
  `parse_source` provides bounded random access, but `read_dataset` and a single
  `read_dataset_selection` result are materialized in memory. Use
  `chunk_iterator` for large datasets or `selection_iterator` for large
  selections; both read in bounded pages and never allocate the complete
  result.
- **Writer lifecycle:** `Hdf5Writer` is an in-memory builder and `encode` returns
  the complete file. `patch_contiguous_dataset` can replace values in an
  allocated contiguous dataset and returns another complete byte image, but
  streaming output and general in-place metadata editing are unsupported.
  Multi-unlimited resizing and v2 B-tree output are rebuilt for the current
  in-memory shape.
- **Read coverage:** virtual datasets, SWMR and parallel HDF5 are outside the
  profile. Built-in chunk filters are decoded directly; other dataset chunk
  filter IDs can be supplied through `Hdf5CustomFilter`. Custom filters on
  metadata heaps are not supported, and built-in filter IDs cannot be
  overridden. Unsupported layouts, indexes, datatypes and unregistered filters
  raise `UnsupportedFeature`; detection of unsupported input is not exhaustive.
  Unsupported attribute datatypes fail the read instead of returning empty
  values.
- **Writer coverage:** output uses superblock v3 and v2 object headers; older
  output formats and virtual storage are not selected. Fixed-size values accept
  shuffle, deflate, Fletcher32 and scale-offset; scale-offset must lead the
  pipeline, and floating-point scale-offset cannot be combined with Fletcher32.
  Variable-length values accept shuffle and deflate; Fletcher32 and
  scale-offset are rejected for them.
- **Dense and shared metadata:** dense output is limited to 9–65,535 records
  with encoded heap objects up to 65,535 bytes, one direct FHDB block and one
  leaf per index. Indirect/huge heap blocks and a free-space manager are not
  emitted. One SOHM table and heap are written; multiple tables, multiple heaps
  and heap extension across blocks are not.
- **External raw data:** only contiguous, unfiltered, fixed-size storage is
  supported. It requires `Hdf5Resolver` and caller-registered, caller-managed
  files; `Hdf5File` alone cannot dereference it. Remote URLs, filtered/chunked
  external layouts and automatic external-file creation are unsupported, and
  short sources are zero-filled for the declared remainder.
- **Numeric and string constraints:** numeric storage wider than 8 bytes and
  binary-float normalizations other than implied MSB are unsupported.
- **Variable-length structures:** sequence bases must be fixed-size and cannot
  contain region references. Custom variable-length string and sequence fills
  are stored through the global heap.
- **References:** support covers legacy object and dataset-region forms plus
  version-4 object, dataset-region and attribute forms. Version-4 references to
  external files are unsupported, and region and attribute targets must be in
  the same file. An object target with no named hard link resolves with a null
  path. The writer keeps legacy references unless a version-4 datatype is
  requested explicitly.
- **Unlimited dimensions:** `append_dataset` writes only at the trailing edge of
  every unlimited axis. Use `write_dataset_slab` for interior writes and
  arbitrary rectangular placement; unlimited axes grow to the slab's trailing
  edge.
- **High-level semantics:** netCDF-4 and MATLAB metadata are read as HDF5
  structures, not as complete netCDF-4 or MATLAB semantics.

Unsupported structures raise `UnsupportedFeature`; malformed input raises
`InvalidField` or `UnexpectedEof`; invalid caller input raises
`InvalidArgument`.

## Development

```sh
moon info
moon fmt
moon check
moon test --target wasm
moon test --target native
H5PY_PYTHON=/path/to/python ./scripts/release_check.sh
```

`scripts/release_check.sh` runs formatting, documentation and package checks,
both test targets, the writer demo and independent h5py verification. The
baseline is h5py 3.13.0 with HDF5 1.14.6. Set `SKIP_H5PY=1` to omit Python
interoperability checks. Fixture regeneration is documented in
[testdata/README.md](testdata/README.md).

## License

Apache-2.0. Zlib decompression uses
[MoonZip](https://github.com/moonbitstack/moonzip), also Apache-2.0.
