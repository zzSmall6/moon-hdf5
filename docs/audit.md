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
  Paged allocation, automatic deletion/shrink reclamation,
  old-heap collection and whole-file compaction remain unimplemented.
- Sources must be repeatable. Value buffers can be bounded, but variable-length
  writing still plans global heaps; `encode()` and edits return complete images.
- Unsupported/malformed-input detection is not exhaustive. Recorded free
  sections are not a complete liveness proof for every object in a corrupt file.
- SWMR, parallel HDF5 and non-seekable output remain outside the implementation.
  Exact datatype, mapping and format limits are in the compatibility matrix.

## Post-release: new-file reserves

The GitHub worktree adds `set_file_space_reserve`: persistent aggregate raw
manager creation with a zero-filled reserve, user blocks, sink and source output.
Five additional tests bring the total to 523 per target. h5py verifies manager
settings, every reserved section, explicit relocation and allocation into the
reserve, followed by reopen/data checks.

Shared-message output and reserves are rejected together before sink mutation.
Continued HDF5 metadata creation exposed an existing shared-index limitation;
read interoperability alone does not establish write interoperability for SOHM.
No package release accompanies this work.

## Post-release: chunk relocation

Explicit `relocate_dataset_storage` now supports allocated chunks in single,
fixed-array, extensible-array and v1/v2 B-tree indexes. Five tests bring the
total to 528 per target, covering sparse fills, vlen heaps, skipped filters,
multi-level uncompressed v2 checksums and insufficient-space failure.
`verify_chunk_relocation.py` compares stored bytes and free regions with HDF5,
then rewrites chunks, allocates another dataset and reopens the output; both
address widths and user blocks are covered. Single-chunk filter masks are
checked by MoonBit because HDF5 1.14.6's chunk-query APIs expose unstable mask
values for that index; stored payload bytes are checked independently.

Further interoperability testing also encountered a pre-existing limitation
when HDF5 creates links in writer-produced dense groups. Continued-allocation
checks use a compact child group; they do not establish arbitrary dense-group
editing compatibility. Automatic chunk replacement/deletion/resize reclamation
and paged allocation remain unimplemented. No MoonCakes release is made.

## Post-release: automatic chunk patch reuse

Chunk patching now shares transaction-local allocation state: new/growing
chunks consume persisted aggregate raw sections before appending; replaced
payloads and unused shrink tails are returned after all chunk writes finish.
Five tests bring the total to 533 per target, including empty slots, insufficient
reserves, repeated growth/shrink, vlen heaps and live/free overlap rejection.
`verify_chunk_space_patch.py` independently checks exact free-byte accounting,
repeated edits across modern indexes and v1 B-trees at 4-/8-byte address widths,
user blocks, HDF5 rewrites and subsequent allocations. Automatic object deletion,
dataset resize and abandoned index/heap reclamation remain unimplemented.

## Post-release: conservative dataset deletion

`delete_object` now returns eligible fixed-width internal dataset payloads to an
existing aggregate raw manager, after all hard aliases are removed. Six tests
bring the total to 539 per target. References in reachable datasets/attributes,
vlen storage, groups and conservatively unsupported layouts prevent reclamation.
Overlap checks reject ambiguous live/free or shared raw storage before deletion.
`verify_deletion_space.py` checks exact free-byte increases, aliases, preserved
object references, 4-/8-byte addresses, user blocks and continued HDF5 allocation.
Headers, indexes, heaps, deleted group descendants and hidden chunks after resize
remain allocated. This work does not publish a MoonCakes package.
