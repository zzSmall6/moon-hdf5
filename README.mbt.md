# moon-hdf5

Pure MoonBit reader and writer for the HDF5 1.14.6 common-file profile. It has
no native HDF5 dependency and runs on wasm and native targets.

This is a documented subset of HDF5, not a complete libhdf5 replacement. The
[compatibility matrix](docs/compatibility.md) defines the supported boundary;
[design](docs/design.md) and [audit](docs/audit.md) contain implementation and
verification notes. Public APIs are listed in
[`src/pkg.generated.mbti`](src/pkg.generated.mbti).

## Status

- `Hdf5File` reads files without modification. `Hdf5Resolver` follows
  registered external links, external raw data and version-4 references.
- `Hdf5Writer` creates complete files, including groups, datasets, attributes,
  links, committed datatypes, filters, fills, variable-length values,
  references, virtual datasets and dense/shared metadata.
- Existing files support value and attribute patching, resize and append,
  attribute/link add, rename and delete, and object creation/deletion. Every
  edit returns a new byte image.
- Writers can emit directly into a caller-provided random-access `Hdf5Sink`;
  `encode()` is the in-memory convenience path. `Hdf5File::copy_to` streams
  bytes-backed or source-backed files in bounded chunks.
- Chunked and extensible datasets can read values from an `Hdf5ChunkSource`;
  chunks are converted, filtered and written one at a time instead of holding
  the complete dataset.
- The release gate passes 491 tests on each target, plus independent h5py
  3.13.0/HDF5 1.14.6 interoperability checks.

## Installation

```sh
moon add Zz04321/moon-hdf5@0.4.2
moon update
```

The package targets `wasm` by default.

## Quick Start

MoonBit 0.10.19 or newer is required. The native target also needs a C compiler.

```sh
git clone https://github.com/zzSmall6/moon-hdf5.git
cd moon-hdf5
moon update
moon run src/cmd/main --target native -- testdata/basic_v0.h5
```

## Reading

```moonbit nocheck
///|
let file = @lib.Hdf5File::parse(@fs.read_file_to_bytes("data.h5"))
let values = file.read_dataset("/matrix")
let slice = file.read_dataset_selection(
  "/matrix",
  @lib.Hyperslab([0UL, 0UL], [2UL, 2UL], [1UL, 2UL]),
)
let attributes = file.attributes("/matrix")
```

`read_dataset_metadata` avoids loading values. Use `chunk_iterator` and
`selection_iterator` for bounded-memory access. `Hdf5Resolver` resolves
version-4 references and links into external files registered with
`add_source`; reader-side custom chunk and metadata filters are registered
through `with_custom_filter` or `add_custom_filter`. `copy_to` copies an
unmodified file image into an `Hdf5Sink` without allocating a second complete
file buffer.

## Datatype Conversion

`read_dataset_as`, `read_dataset_selection_as`, `read_attribute_as` and
`Dataset::convert_to` convert decoded values to an explicit memory datatype.
The default `Strict` mode rejects narrowing, fractional integer results,
precision loss and out-of-range values. `Lossy` applies HDF5-style wrapping or
truncation; `NoConversion` requires identical datatypes. Writer and editor
`*_typed` methods convert memory values for any supported storage layout, and
`chunk_iterator_as` / `selection_iterator_as` convert bounded pages.

```moonbit nocheck
///|
let floats = file.read_dataset_as("/matrix", @lib.Datatype::write_float(8))
let compact = file.read_dataset_as(
  "/matrix",
  @lib.Datatype::write_float(4),
  options=@lib.Hdf5ConversionOptions::lossy(),
)
```

The typed APIs cover every supported storage layout. The release gate checks
binary16 boundaries, integer width/order/offset layouts, extended floats,
opaque resizing, complex-component precision, bounded chunk/selection
conversion, and numeric fixture round-trips with h5py against HDF5 1.14.6.

## Writing

```moonbit nocheck
///|
let writer = @lib.Hdf5Writer::new()
writer.add_group("/experiment")
writer.add_dataset(
  "/experiment/ids",
  [3UL],
  @lib.Int64Values([10L, 20L, 30L]),
)
writer.add_chunked_dataset(
  "/experiment/codes",
  [6UL],
  [3U],
  @lib.Int64Values([10L, 11L, 12L, 20L, 21L, 22L]),
  fill_value=@lib.Int64Values([-1L]),
  filters=[@lib.Shuffle, @lib.Deflate(6), @lib.Fletcher32],
)
writer.add_attribute(
  "/experiment/ids",
  "unit",
  [],
  @lib.FixedStringValues(["items"], 8),
)
writer.add_soft_link("/experiment", "latest", "/experiment/ids")

let sink = @lib.Hdf5MemorySink::new()
writer.write_to(sink)
let bytes = sink.to_bytes()
```

The writer also provides compact, contiguous, resizable/extensible, external
and virtual dataset creation; committed datatypes; user blocks; structured,
variable-length and reference values; built-in filters and caller-registered
custom filters. See the [compatibility matrix](docs/compatibility.md) for exact
variants and constraints. `encode()` is shorthand for writing to an
`Hdf5MemorySink` and taking its bytes.

For bounded-memory chunk output, implement `Hdf5ChunkSource` and pass a
zero-element datatype prototype:

```moonbit nocheck
///|
writer.add_chunked_dataset_from_source(
  "/experiment/streamed",
  [rows, columns],
  [64U, 64U],
  @lib.Float32Values([]),
  source,
  filters=[@lib.Shuffle, @lib.Deflate(6)],
)
```

Sources must return repeatable edge-clipped chunks. `add_extensible_dataset_from_source`
has the same contract. Source-backed datasets cannot be resized or patched in
the builder.

## Editing Existing Files

```moonbit nocheck
///|
let file = @lib.Hdf5File::parse(@fs.read_file_to_bytes("data.h5"))
let patched = file.patch_dataset_values(
  "/group/values",
  @lib.Int64Values([10L, 20L, 30L, 40L]),
)
let with_attribute = @lib.Hdf5File::parse(patched).add_attribute(
  "/group/values",
  "units",
  [],
  @lib.FixedStringValues(["items"], 8),
)
let edited = @lib.Hdf5File::parse(with_attribute).add_soft_link(
  "/group",
  "latest",
  "/group/values",
)
@fs.write_bytes_to_file("edited.h5", edited)
```

Value patching preserves shapes and compatible datatypes. `resize_dataset` and
`append_dataset` update chunked storage while preserving chunk masks and
coordinates. Attribute, link and object edits rebuild compact/dense or
symbol-table metadata as needed. Editing is append-only; superseded bytes
remain unreferenced.

## Command Line

```sh
moon run src/cmd/main -- inspect testdata/basic_v0.h5
moon run src/cmd/main -- metadata testdata/basic_v0.h5 /matrix
moon run src/cmd/main -- stats testdata/basic_v0.h5 /matrix
moon run src/cmd/main -- read testdata/basic_v0.h5 /matrix 0,0 2,2 1,2
moon run src/cmd/main -- read testdata/basic_v0.h5 /matrix --points "1,2;0,1"
```

Coordinates are zero-based. Hyperslabs use `start,count,stride`; point lists use
semicolons. Exit status is 0 for success, 1 for data errors and 2 for invalid
arguments.

## Limitations

- Scope is the documented HDF5 1.14.6 common-file subset. Non-seekable
  streaming output, SWMR, parallel HDF5 and in-file storage reclamation are
  outside the profile.
- In-memory writer inputs and single dataset/selection results are memory
  resident; chunk-source output and read iterators are bounded. Edits return a
  complete byte image. Output sinks must support random access. Edits are
  append-only, and external sources require caller registration and
  caller-managed files.
- Mapping, storage and conversion constraints are listed in the
  [compatibility matrix](docs/compatibility.md).
- Writer limits include 65,524-byte compact payloads, dense attributes up to
  65,535 records and user blocks up to 1 GiB. Four-byte files reject structured
  variable-length or reference custom fills for HDF5 1.14.6 interoperability.
- Unsupported structures raise `UnsupportedFeature`; malformed input raises
  `InvalidField` or `UnexpectedEof`. Detection is not exhaustive.

## Development

```sh
moon info
moon fmt
moon check
moon test --target wasm
moon test --target native
H5PY_PYTHON=/path/to/python ./scripts/release_check.sh
```

`release_check.sh` runs formatting, documentation, package and both test-target
checks, plus independent h5py verification. Set `SKIP_H5PY=1` to omit Python
checks. CI pins MoonBit `0.10.19+bee578715` so formatting and compiler behavior
stay reproducible; newer toolchains are expected to be validated before
advancing the pin. Fixture generation is documented in
[testdata/README.md](testdata/README.md).

## License

Apache-2.0. Zlib decompression uses
[MoonZip](https://github.com/moonbitstack/moonzip), also Apache-2.0.
