# Release audit — 0.4.3

Scope: HDF5 1.14.6 common-file support and recent streaming/space-reuse APIs.
The [compatibility matrix](compatibility.md) defines the supported boundary;
this audit does not establish complete libhdf5 equivalence.

## Findings

- Fixed a cross-manager validation gap: a serialized free section could overlap
  another manager's header or section allocation. Inspection now checks all
  manager metadata regions before space reuse. A regression test exercises
  both bytes- and source-backed reads with valid checksums.
- Reviewed contiguous allocation/relocation, aligned splitting/coalescing,
  FSHD/FSSE writeback and typed source conversion. Existing tests cover empty
  managers, repeated relocation, preserved references and continued HDF5 writes.
- Updated release metadata and condensed README. The registry version checked
  before this release was 0.4.2.

## Verification

Toolchain: MoonBit 0.10.19, `moonbitlang/x@0.5.7`,
`moonbitstack/moonzip@0.4.0`; h5py 3.13.0/HDF5 1.14.6.

`scripts/release_check.sh` checks interfaces, formatting, compilation,
documentation, package contents and 518 tests on each of wasm/native.
Independent h5py checks cover writer/editor layouts, filters, datatype
conversion, user blocks, external/virtual data, references and shared/dense
metadata. Space checks compare every persisted section with
`H5Fget_free_sections`, then allocate and reopen edited files with HDF5.

## Remaining Boundaries

- Free-space reuse requires an existing persistent aggregate raw manager.
  Creating managers, paged allocation, automatic deletion/shrink reclamation,
  old-heap collection and whole-file compaction remain unimplemented.
- Sources must be repeatable. Value buffers can be bounded, but variable-length
  writing still plans global heaps; `encode()` and edits return complete images.
- Unsupported/malformed-input detection is not exhaustive. Recorded free
  sections are not a complete liveness proof for every object in a corrupt file.
- SWMR, parallel HDF5 and non-seekable output remain outside the implementation.
  Exact datatype, mapping and format limits are in the compatibility matrix.
