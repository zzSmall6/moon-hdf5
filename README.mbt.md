# moon-hdf5

Pure MoonBit reader and writer for a documented subset of HDF5 1.14.6.
Runs on wasm and native without a native HDF5 dependency.

## Install

```sh
moon add Zz04321/moon-hdf5@0.4.3
moon update
```

Requires MoonBit 0.10.19 or newer; native builds also need a C compiler.
The default target is `wasm`.

## Read and Write

```moonbit nocheck
///|
let file = @lib.Hdf5File::parse(@fs.read_file_to_bytes("data.h5"))
let dataset = file.read_dataset("/matrix")
let slice = file.read_dataset_selection(
  "/matrix",
  @lib.Hyperslab([0UL, 0UL], [2UL, 2UL], [1UL, 2UL]),
)
let floats = file.read_dataset_as("/matrix", @lib.Datatype::write_float(8))

let writer = @lib.Hdf5Writer::new()
writer.add_dataset("/ids", [3UL], @lib.Int64Values([10L, 20L, 30L]))
writer.add_attribute("/ids", "unit", [], @lib.FixedStringValues(["items"], 8))
@fs.write_bytes_to_file("output.h5", writer.encode())
```

- **Read:** compact, contiguous, chunked, external and virtual datasets;
  attributes, links, committed types, variable-length values and references.
  `Hdf5Resolver` accesses caller-registered external files.
- **Write:** groups, datasets, links and attributes; resizable chunked storage,
  built-in/custom filters, fills, references and dense/shared metadata.
- **Convert:** `*_as`, `convert_to` and `*_typed` APIs use strict conversion by
  default. Lossy and no-conversion policies are explicit options.
- **Stream:** `parse_source`, chunk/selection iterators, `Hdf5Sink`, `copy_to`,
  `Hdf5ChunkSource` and `Hdf5ValueSource` avoid complete dataset/output buffers.
  Source callbacks must be repeatable; typed source APIs convert each block.
- **Edit:** patch values, resize/append datasets, add/rename/delete attributes
  and links, and create/delete objects. Each edit returns a new byte image.
- **Inspect/reuse space:** `file_space_info` and `free_space_managers` inspect
  persisted sections. Contiguous allocation and `relocate_dataset_storage`
  reuse persistent aggregate raw-data space; explicit relocation supports
  contiguous data and chunks with explicit indexes, preserving stored bytes.
  `set_file_space_reserve(bytes)` creates a new persistent raw manager with a
  zero-filled reserve; shared-message output must be disabled.

See the [API](src/pkg.generated.mbti), [compatibility matrix](docs/compatibility.md),
[design](docs/design.md) and [audit](docs/audit.md) for contracts and evidence.

## Boundaries

This is a common-file subset, not a complete libhdf5 replacement. SWMR,
parallel HDF5, non-seekable output and general storage reclamation are outside
the current implementation. Space reuse does not reclaim old heaps or shrink
files. Other editing paths append storage.

Single dataset/selection results and ordinary writer inputs remain resident;
iterators and source callbacks bound value buffers. Variable-length streaming
still plans global heaps. External files and custom filters require caller
registration. Exact format and conversion limits are in the compatibility matrix.
Malformed-input detection is not exhaustive.

## CLI and Development

```sh
moon run src/cmd/main -- inspect testdata/basic_v0.h5
moon run src/cmd/main -- metadata testdata/basic_v0.h5 /matrix
moon run src/cmd/main -- read testdata/basic_v0.h5 /matrix --points "1,2;0,1"
moon info && moon fmt
H5PY_PYTHON=/path/to/python ./scripts/release_check.sh
```

The release gate checks interfaces, formatting, documentation, package contents,
528 tests on each target and h5py 3.13.0/HDF5 1.14.6 interoperability. CI pins
MoonBit `0.10.19+bee578715`. [Fixture regeneration](testdata/README.md).

## License

Apache-2.0. Zlib decompression uses
[MoonZip](https://github.com/moonbitstack/moonzip), also Apache-2.0.
