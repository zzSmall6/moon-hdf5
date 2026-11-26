# Compatibility

moon-hdf5 implements a verified subset of the HDF5 1.14.6 common-file profile.
“Supported” means the repository contains a fixture, test or h5py check for
that form. This document is the delivery boundary; detailed evidence is in
[audit.md](audit.md).

## Profile

- Baseline: h5py 3.13.0 with HDF5 1.14.6 on wasm and native targets.
- Outside the profile: non-seekable streaming output, SWMR, parallel HDF5,
  general storage reclamation, metadata edits outside the documented APIs,
  and high-level netCDF-4 or MATLAB semantics.
- The release gate is `scripts/release_check.sh`: formatting, documentation and
  package checks; both test targets; writer/editor demos; independent h5py
  verification.

## Reader

| Area | Supported | Boundary |
| --- | --- | --- |
| Framing and navigation | Superblocks v0–v3, user blocks, 2/4/8-byte address and length widths, compact/dense groups, hard and soft links, source-backed reads and bounded `copy_to` output | Values must fit the declared widths; path normalization is lexical; traversal stops on an address repeated in the current ancestor chain |
| File-space inspection | Allocation policies, legacy/v1 file-space messages, checksummed file-client managers and simple/small/large serialized sections; bytes- and source-backed reads | Reports persisted sections only; excludes transient aggregators and untracked holes; no allocation or reclamation; fractal-heap client managers are separate |
| Cross-file data | `Hdf5Resolver` follows chained external links and references; contiguous external raw data spans registered files and element boundaries | `Hdf5File` alone does not open external files; short sources are zero-filled |
| Storage and indexes | Compact, contiguous, virtual and chunked datasets; v1 B-tree, single-chunk, implicit, fixed-array, extensible-array and v2 B-tree indexes; default/custom/undefined fills | Unsupported index variants are rejected; missing chunks use the effective fill value |
| Filters | Deflate, shuffle, Fletcher32, N-bit and integer/f32/f64 scale-offset; registered dataset and metadata-heap decoders | N-bit must precede lossless filters and follow only shuffle; unsupported scale modes and unregistered required filters are rejected |
| Datatypes and values | Fixed-point (including `BigInt`), IEEE/extended floats, fixed/vlen strings, compound, enum, array, time, bitfield, opaque, sequence, reference and complex values | Values beyond `Double` precision are rounded; reference bytes are decoded but resolved separately |
| Datatype conversion | Strict, lossy and no-conversion policies; numeric, string, enum, opaque, compound, array, sequence, time, bitfield, reference and complex values; typed read/write/edit APIs and bounded conversion iterators | Target enums must contain the converted value; references require matching kind and width; strict compound conversion requires every target member unless missing members are explicitly allowed; strict complex bridges compare both component types; no-conversion requires identical datatypes even for empty inputs. Binary16 uses IEEE round-to-nearest-even; HDF5 1.14.6's generic converter differs at some subnormal ties and overflow values. HDF5 1.14.6 interoperates through compound or two-element array complex forms; class-11 complex is a post-profile extension |
| Attributes and shared metadata | Inline v1–v3, dense, SOHM wrappers, shared datatype/dataspace fields, fills and filter pipelines | Unsupported attribute datatypes fail the read; SOHM must be referenced by the superblock extension and use one master table |
| Virtual datasets | All, empty and hyperslab mappings; same-file and registered external sources; fill gaps; one unlimited axis with `%b`/`%%`; numeric, enum, complex, array, compound, fixed-string and fixed-to-vlen-string conversion; bounded page/tile reads | Point/block mappings and multiple unlimited axes are rejected by HDF5 1.14.6; unlimited source selections and vlen/reference conversions are unsupported; a single selection result is memory resident |
| References | Legacy object/region references and version-4 object/region/attribute references at 4- or 8-byte widths; same-file and registered external targets | Unresolved external targets require `Hdf5Resolver`; object targets may remain unresolved with a null path |

## Writer and editing

| Area | Supported | Boundary |
| --- | --- | --- |
| Framing and output | Superblock v3, v2 object headers, OCHK continuations, user blocks; direct `Hdf5Writer::write_to` emission, bounded `Hdf5File::copy_to`, one-chunk-at-a-time `Hdf5ChunkSource` and one-block-at-a-time `Hdf5ValueSource` output | New output uses v3; user blocks are 0 or powers of two from 512 B through 1 GiB; sinks must support random access and truncation; sources must be repeatable |
| Object creation | Groups, contiguous/compact/chunked/external/virtual datasets, committed datatypes, attributes and links; fills; finite, extensible-array and v2 B-tree chunks; in-memory, streamed chunk or contiguous block values | Compact payloads are limited to 65,524 bytes; v2 B-tree output is bulk-built; external creation does not open or validate sources; source-backed datasets cannot be resized or patched in the builder |
| Dense and shared metadata | Compact/dense links, dense attributes, SOHM fills/filter pipelines/attributes, committed datatypes, huge objects and indirect heap roots | Dense attributes cap at 65,535 records; dense links use the 64-bit creation-order field; one SOHM master table; shared-message output is verified for reads and documented editor operations, not arbitrary libhdf5 metadata creation |
| Filters and fills | Built-in filters and caller-registered encoders/decoders; named v2 entries; per-chunk masks; custom fills | Built-in IDs cannot be overridden; vlen chunk data supports no filter, shuffle or deflate, not Fletcher32/scale-offset; four-byte structured vlen/ref custom fills are rejected |
| Value editing | Compact, allocated/unallocated contiguous, filtered/unfiltered chunked, resize, trailing-corner append, vlen and reference values | Patching extends existing writable storage or reconstructs missing chunk-index records; unrelated metadata is not created |
| Raw-space reuse | `set_file_space_reserve` creates a persistent aggregate raw manager and zero-filled reserve in new files; unallocated contiguous patches reuse sections; `relocate_dataset_storage` moves an allocated contiguous payload and frees its old region; aligned best-fit allocation and manager writeback | New reserves require shared-message output disabled; existing reuse needs suitable simple sections without ghosts; no file shrink, paged allocation, old-heap collection or automatic delete/resize reclamation |
| Attribute editing | Add, rename and delete on v1/v2 headers; compact/dense conversion; vlen and reference values; shared attributes rebuilt as local messages | Same-file targets must exist; external version-4 targets require registered sources; metadata-heap filters need available encoders |
| Link and object editing | Hard/soft/external link add, rename and delete; object deletion; symbol-table conversion; compact/dense rebuilds | Hard targets must already exist; `delete_object` removes every hard link and rejects the root; external targets are stored without resolution |
| External raw data | Contiguous unfiltered slots for fixed-width values and legacy object references, including nested compound/fixed-array members | Variable-length and heap-backed region/version-4 references are rejected; HDF5 external storage cannot be chunked or filtered |

## Reproduce

```sh
moon test --target wasm
moon test --target native
moon run src/cmd/write_demo --target native -- output.h5
python scripts/verify_writer_output.py output.h5
moon run src/cmd/resize_append_demo --target native -- resize-append.h5
python scripts/verify_resize_append_output.py resize-append.h5
moon run src/cmd/conversion_demo --target native -- conversion.h5 \
  testdata/datatypes_v114.h5 testdata/numeric_layouts.h5 \
  testdata/wide_numeric.h5
python scripts/verify_conversion_interop.py conversion.h5 \
  testdata/datatypes_v114.h5 testdata/numeric_layouts.h5 \
  testdata/wide_numeric.h5
```

`scripts/release_check.sh` runs the complete gate. Fixture generation is in
[testdata/README.md](../testdata/README.md).
