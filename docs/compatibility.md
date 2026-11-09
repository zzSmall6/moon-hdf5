# Compatibility

moon-hdf5 targets the HDF5 1.14.6 common-file profile. “Verified” means the
repository contains a fixture, test or h5py check for that direction. This is a
documented subset, not complete HDF5 support.

## v1.0 profile

The v1.0 profile is the verified rows in the reader and writer tables below,
including their boundaries. The external raw-data table is a separate
post-v1.0 extension. HDF5 2.0 datatype fixtures are forward-compatibility
evidence, not part of the 1.14.6 baseline.

Outside the profile: virtual datasets, SWMR, parallel HDF5, caller-registered
filters on metadata heaps, general in-place metadata editing, netCDF-4/MATLAB
high-level semantics, version-4 references to external files, numeric storage
wider than 8 bytes and non-implied-MSB binary floats.

The release gate is `scripts/release_check.sh`: formatting, documentation and
package checks; wasm/native tests; writer demo; and independent h5py
verification with h5py 3.13.0/HDF5 1.14.6.

## Reader

| Area | Verified | Boundary |
| --- | --- | --- |
| Framing | Superblocks v0–v3, user blocks, 4- and 8-byte addresses | Other address/length combinations are not fully tested |
| Groups | Symbol-table, compact and dense groups, creation-order indexes, soft links with relative and `.`/`..` targets, all named hard-link paths | Traversal stops at an address already in the current ancestor chain; dot-segment normalization is lexical |
| Cross-file links | `Hdf5Resolver` follows chained external links and relative targets | `Hdf5File` alone does not open external targets |
| Attributes | Inline v1–v3, dense, SOHM wrappers, shared datatype/dataspace fields | Unsupported attribute datatypes fail the read instead of returning empty values |
| Layouts and indexes | Compact, contiguous, chunked; v1 B-tree, single-chunk, implicit, fixed-array, extensible-array and v2 B-tree indexes | Unsupported index variants are rejected |
| Filters | Deflate, shuffle, Fletcher32, integer and f32/f64 decimal scale-offset, masks and fill markers; caller-registered dataset chunk decoders through `Hdf5CustomFilter` | Other scale modes are rejected; custom metadata-heap filters and overrides of built-in filter IDs are unsupported; unregistered required filters raise `UnsupportedFeature` |
| Numeric values | 1–8-byte fixed-point with byte order, bit offset/precision and padding; IEEE float16/32/64; implied-MSB binary floats | Wider storage and other normalizations are rejected |
| Structured values | Compound, enum, array, vlen sequence, bitfield, opaque, time, reference, complex; datatype messages v1–v5 | Reference values decode as bytes; resolution is separate |
| References | Legacy object and dataset-region references under 4- and 8-byte addresses; version-4 object, dataset-region and attribute references; compact, contiguous and filtered targets; point, hyperslab, block, all and empty selections | Version-4 external-file references are rejected; region and attribute targets must be in the same file; unresolved object targets keep a null path |
| Fill values | Default, fixed-size, structured, custom vlen-string and vlen-sequence fills; undefined and `FILL_TIME_NEVER` policies; disabled or absent fills zero unallocated storage | Malformed fill messages and unresolvable global-heap targets are rejected |
| Selections | Hyperslabs, ordered points, bounded-memory tile iteration and bounded-memory selection pages | A single `read_dataset_selection` result is memory resident; use `selection_iterator` for large selections |
| Shared messages | SOHM table/indexes/heaps, reserved indexes without heaps, wrappers v1–v3, committed targets, fills, filter pipelines, field sharing and dense shared records | Other shared message types are not fixture-verified |
| Dense metadata | Managed, tiny, direct-huge and indirect-huge heap objects; metadata v2 B-trees | Broader non-matrix interoperability remains untested |
| Producer files | netCDF4 1.7.5/libnetcdf 4.10.1/HDF5 2.2.0 structures; hdf5storage 0.2.2 MATLAB metadata | Verifies encoded HDF5 structures, not complete netCDF-4 or MATLAB semantics |

Fixtures cover superblock v0/v2/v3 matrices, HDF5 1.14.6 default/`latest` and
HDF5 2.0 datatypes, scale-offset, hard links, dense/fractal heaps, references,
shared messages and producer files.

## External raw data

This post-v1.0 reader extension is resolver-only.

| Area | Verified | Boundary |
| --- | --- | --- |
| External raw data | Registered 4-/8-byte-address files, multiple slots, offsets, element-unaligned boundaries, full/selection reads and chunk iteration | `Hdf5File` cannot dereference it; filters, chunked external layouts and remote URLs are unsupported; short sources are zero-filled |

`Hdf5Writer::add_external_dataset` writes the same contiguous, unfiltered subset
for fixed-size values. Callers create the external files and exact slot sizes.

## Writer

`src/cmd/write_demo` and `scripts/verify_writer_output.py` check the output with
h5py 3.13.0/HDF5 1.14.6. Version-4 reference output is checked by MoonBit
round-trip tests because h5py 3.13.0 does not expose version-4 reference values
through its NumPy dtype bridge.

| Area | Verified | Boundary |
| --- | --- | --- |
| Framing | Superblock v3 and v2 object headers with OCHK continuations | Older output formats are not selected |
| Groups and attributes | Compact/dense collections, creation order, continued headers, inline/dense attributes | Dense limits below |
| Dense metadata | 9–65,535 records with encoded objects up to 65,535 bytes; FRHP/FHDB, BTHD/BTLF and type 6/9 indexes; optional one SOHM table/heap | One direct block and leaf per index; no huge/indirect blocks or free-space manager; larger compact sets use OCHK |
| Datasets | Contiguous, external raw data, paged fixed-array, extensible-array and v2 B-tree chunks | No virtual storage; external storage is unfiltered fixed-size only |
| Numeric values | int/uint 1–8 bytes and IEEE float32/64 in either byte order | Nonstandard floating-point output is absent |
| Strings and structures | Fixed/vlen UTF-8; compound, enum, array, time, bitfield, opaque, references, complex and sequences | Complex output is little-endian canonical `r`/`i`; sequence bases are fixed-size and cannot contain region references |
| References | Legacy 8-byte objects and 12-byte regions plus opt-in version-4 object, region and attribute references in datasets, attributes, compounds and arrays | Same-file targets only; regions inside sequences are absent; legacy remains the default |
| Filters | Shuffle, deflate, Fletcher32, integer and f32/f64 decimal scale-offset, per-chunk masks | Fixed-size values support the full set; variable-length values support shuffle/deflate, not Fletcher32 or scale-offset; scale-offset must lead; floating scale-offset cannot use Fletcher32 |
| Fills | Default, fixed-size/structured, vlen-string and vlen-sequence fills; optional SOHM | Variable-length fills use the global heap |
| Shared messages | One SOHM table/heap for whole attributes, fills and filter pipelines; equal fills, attributes and pipelines reuse one message; committed datatypes use wrapper v3 type 2 | Multiple SOHM tables/heaps are not written |
| Extension | One or more unlimited axes with in-memory resize, append and arbitrary rectangular slab writes; chunk filter masks stay attached to chunk coordinates across shape changes | Multiple unlimited axes use a v2 B-tree; `append_dataset` writes at the trailing corner, `write_dataset_slab` writes at arbitrary coordinates and grows unlimited axes to the slab edge |
| Value patching | `patch_contiguous_dataset` replaces fixed-size values in an allocated, unfiltered contiguous dataset and returns a complete byte image | Bytes-backed `Hdf5File` only; shape and datatype must match; chunked, compact, external, filtered, unallocated, variable-length and reference storage is rejected; metadata is not edited and the dataset is not resized or relaid out |

## Reproduce

```sh
moon test --target wasm
moon test --target native
moon run src/cmd/write_demo --target native -- output.h5
python scripts/verify_writer_output.py output.h5
```

Ordinary MoonBit tests need no Python or native HDF5. Fixture generation is in
[testdata/README.md](../testdata/README.md).
