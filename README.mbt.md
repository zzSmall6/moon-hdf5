# moon-hdf5

Pure MoonBit HDF5 reader and writer for the HDF5 1.14.6 common-file profile.
It has no native HDF5 dependency and runs on wasm and native targets.

This project implements a documented subset of HDF5. The
[compatibility matrix](docs/compatibility.md) is the delivery boundary,
[design notes](docs/design.md) cover implementation details, and
[audit notes](docs/audit.md) record verification. Public APIs are listed in
[`src/pkg.generated.mbti`](src/pkg.generated.mbti).

## Status

- `Hdf5File` reads existing files without modification. `Hdf5Writer` creates
  complete new files in memory. `Hdf5Resolver` follows registered external
  links and resolver-backed raw-data sources.
- Reading covers superblocks v0–v3, compact/contiguous/chunked storage, built-in
  and registered custom filters, dense and shared metadata, structured and
  variable-length values, references, virtual datasets, and bounded
  chunk/selection iterators for directly addressable mappings.
- Writing covers groups, datasets, attributes, links, committed datatypes,
  filters, fills, dense/SOHM metadata, nested variable-length and reference
  values, compact/contiguous/chunked/virtual/external storage, optional user
  blocks, and resizable datasets.
- Existing-file editing covers value and attribute patching, dataset resize and
  trailing-corner append, attribute and link add/rename/delete, object
  creation/deletion, and append-only metadata rebuilds. Virtual datasets can be
  added to new or existing files. Every editing API returns a new byte image.

See [Compatibility](docs/compatibility.md) for the exact supported and rejected
cases.

## Installation

Add the released package to a MoonBit module:

```sh
moon add Zz04321/moon-hdf5@0.3.1
moon update
```

The package targets `wasm` by default.

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

Imports for the examples are shown in
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

`read_dataset_metadata` does not load values. Use `chunk_iterator` for
bounded-memory chunk processing and `selection_iterator` for large hyperslab or
point selections. Simple values use `Integers`, `UnsignedIntegers`, `Floats`,
`Strings` or `Enums`; other datatypes use recursive `Structured` values.

References are returned as encoded bytes. `Hdf5File` resolves same-file targets;
`Hdf5Resolver` can also follow version-4 references and external links into
registered sources.

```moonbit nocheck
///|
let resolver = @lib.Hdf5Resolver::new(bytes)
resolver.add_source("other-file.h5", @fs.read_file_to_bytes("other-file.h5"))
let remote = resolver.read_dataset("/experiment/remote_counts")

let object = resolver.resolve_object_reference(reference_bytes)
let region = resolver.read_region_reference(region_bytes)
let attribute = resolver.read_attribute_reference(attribute_bytes)
```

Third-party chunk filters use `Hdf5CustomFilter` with
`Hdf5File::with_custom_filter` or `Hdf5Resolver::add_custom_filter`. Built-in
filter IDs cannot be overridden, duplicate IDs are rejected, and decoders must
return exactly the expected byte count.

## Writing

```moonbit nocheck
///|
let writer = @lib.Hdf5Writer::new()
writer.set_user_block(512)

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
writer.add_chunked_dataset(
  "/experiment/codes",
  [6UL],
  [3U],
  @lib.Int64Values([10L, 11L, 12L, 20L, 21L, 22L]),
  fill_value=@lib.Int64Values([-1L]),
  filters=[@lib.Shuffle, @lib.Deflate(6), @lib.Fletcher32],
)
writer.add_attribute(
  "/experiment/records",
  "unit",
  [],
  @lib.FixedStringValues(["items"], 8),
)
writer.add_soft_link("/experiment", "latest", "/experiment/records")

let output = writer.encode()
```

Dataset creation options:

- `add_dataset`: contiguous storage with an optional fill value.
- `add_compact_dataset`: inline object-header storage with an optional fill
  value, up to 65,524 bytes.
- `add_chunked_dataset`: finite fixed-array chunks or resizable chunks via
  `maximum_dimensions`.
- `add_extensible_dataset`, `resize_dataset`, `append_dataset` and
  `write_dataset_slab`: in-memory resizing, append and rectangular writes.
- `add_external_dataset`: unfiltered external raw-data slots for fixed-width
  values, including legacy object references nested in compounds and
  fixed-arrays; the caller creates the external files.
- `set_user_block`: reserve a zero-filled 512 B, 1 KiB, ... or 1 GiB prefix and
  place the superblock after it. Existing user-block files remain editable.
- `set_user_block_content`: store caller-provided prefix bytes, rounded up to
  the next permitted user-block size through 1 GiB and zero-padded.
- `add_virtual_dataset`: virtual storage from one or more source mappings. Use
  `.` for a same-file source; external source names are stored without opening
  the file. `maximum_dimensions` with a `None` entry declares one unlimited
  axis. In a single unlimited virtual hyperslab, `%b` in a source file or
  dataset name expands to the block number and `%%` is a literal percent sign.
  Numeric, enum, complex, array, compound, same-charset fixed-string and
  fixed-string-to-vlen-string source values are converted to the virtual
  datatype.

Fixed-size values support Shuffle, N-bit, deflate, Fletcher32 and scale-offset.
N-bit must precede lossless filters, with Shuffle only before it; combining
Shuffle with N-bit requires full-precision fields.
Variable-length values support Shuffle/deflate and must use fixed-size sequence
bases. Nested variable-length and reference members are encoded recursively.
Register custom encoders with `Hdf5Writer::add_custom_filter` before adding a
dataset that uses `Custom(id, parameters)`. Existing-file
`add_chunked_dataset` accepts the same filters through `custom_filters=[filter]`.

Version-4 references are opt-in per datatype; legacy references remain the
writer default. Use the `Datatype::write_version4_*_reference` constructors for
version-4 object, region or attribute references. Same-file targets use
`ObjectReference`, `RegionReference` or `AttributeReference`. External targets
use `ExternalObjectReference`, `ExternalRegionReference` or
`ExternalAttributeReference` with a file name registered through
`Hdf5Writer::add_external_source`. The writer validates the target object,
region dataspace and attribute, supports 4- and 8-byte target files, and stores
the file name in the reference. The caller owns and must keep the target file
available under that name.

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
  "/group", "latest", "/group/values",
)
@fs.write_bytes_to_file("edited.h5", edited)
```

- `patch_dataset_values` handles compact, contiguous and chunked datasets,
  including variable-length values, same-file references and external
  version-4 references registered through `Hdf5File::with_external_source`.
  The replacement must keep the dataset's shape and a compatible datatype.
- `resize_dataset` changes simple chunked datasets with extensible-array, v1
  B-tree or v2 B-tree indexes. Existing chunk coordinates and filter masks are
  retained; shrinking hides chunks and regrowing restores them.
- `append_dataset` writes a block at the trailing corner, grows unlimited axes,
  and updates filtered, variable-length and newly created chunk metadata.
- `patch_contiguous_dataset` is the strict allocated-contiguous-only form.
- `patch_attribute_values` replaces existing attribute values without changing
  names, creation order or storage form when the record can be updated in place.
- `add_attribute`, `rename_attribute`, `delete_attribute`, `add_*_link`,
  `rename_link` and `delete_link` rebuild compact/dense collections as needed.
- `add_group`, dataset creation APIs, `add_compact_dataset`,
  `add_committed_datatype` and `delete_object` append objects and attach or
  detach links through the same rebuild path. Compact payloads are capped at
  65,524 bytes; large object headers use checksummed continuation chunks.

Editing is append-only: new objects, values and rebuilt metadata are appended,
and superseded bytes remain unreferenced.

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

moon-hdf5 is not a drop-in replacement for libhdf5.

- Outside the common-file profile: streaming output, free-space reclamation,
  SWMR and parallel HDF5.
- Virtual datasets support all, empty and hyperslab mappings, same-file or
  registered external sources, one unlimited virtual hyperslab axis with
  `%b`/`%%` source names, numeric, enum, complex, array, compound,
  same-charset fixed-string and fixed-string-to-vlen-string conversion.
  Compound members map by name; missing target members are zero-filled and
  extra source members are ignored. Enum conversion uses the underlying
  integer. Point/block mappings and multiple unlimited axes are rejected by
  HDF5 1.14.6; unlimited source selections, variable-length source conversions
  and reference conversions have no supported HDF5 1.14.6 VDS conversion path.
  Float conversion is limited to 64-bit integer magnitudes and standard binary
  floats. Writers do not open sources; readers require source registration.
- The API is memory-resident, editing is append-only, and references or patches
  target existing objects. External raw data is contiguous and unfiltered;
  reading requires the resolver, and creation is limited to fixed-width values
  and legacy object references. Dense attributes hold 9–65,535 records; dense
  links use the format's 64-bit creation-order field and are not capped at
  65,535. Four-byte structured values with variable-length or reference custom
  fills are rejected for HDF5 1.14.6 interoperability.
- The in-memory writer caps user blocks at 1 GiB; HDF5 permits larger
  power-of-two blocks.
- Detection of unsupported input is not exhaustive. Unsupported structures
  raise `UnsupportedFeature`; malformed input raises `InvalidField` or
  `UnexpectedEof`.

The [compatibility matrix](docs/compatibility.md) is the authoritative list of
supported variants and boundaries; [design](docs/design.md) and
[audit](docs/audit.md) notes cover implementation and verification details.

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
both test targets, writer demos and independent h5py verification. The baseline
is h5py 3.13.0 with HDF5 1.14.6. Set `SKIP_H5PY=1` to omit Python checks.
Fixture regeneration is documented in [testdata/README.md](testdata/README.md).

## License

Apache-2.0. Zlib decompression uses
[MoonZip](https://github.com/moonbitstack/moonzip), also Apache-2.0.
