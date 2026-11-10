# Code audit — 2026-11-10

Scope: current worktree, reader and writer, with emphasis on the HDF5 1.14.6
v1.0 delivery profile and post-v1.0 extensions.

## Verdict

The release gate passes on the current worktree:

- wasm: 281 tests passed;
- native: 281 tests passed;
- interface, formatting, `moon check`, `moon doc` and package checks passed;
- writer demo verified by h5py 3.13.0/HDF5 1.14.6;
- compact and sparse chunked patch output was re-read by h5py 3.13.0/HDF5 1.14.6;
- caller-defined filter encoding, named pipelines, chunk-mask skips, patch
  re-encoding and growth relocation are covered by MoonBit round-trip tests;
- numeric-layout, scale-offset, vlen-fill, NetCDF-4, external-storage,
  shared-message, hard-link, MATLAB-compatible and reference fixtures verified;
- the version-4 reference fixture and writer output are exercised by MoonBit
  round-trip tests.

The implementation remains a documented HDF5 subset. Known unsupported
structures raise errors; detection is not exhaustive.

## Verification coverage

| Area | Evidence |
| --- | --- |
| Framing and navigation | Superblock v0–v3, user blocks, compact/dense groups, hard/soft links, `.`/`..` path normalization, source-backed reads |
| Dense metadata | FRHP direct/indirect blocks, huge objects, v2 B-tree leaf/internal nodes and SOHM heap IDs |
| Datasets | Compact, contiguous, external raw data and every supported chunk index |
| Fill policies | Default, custom, undefined and `FILL_TIME_NEVER`; malformed messages and external-storage boundaries remain rejected |
| Filters | Shuffle, deflate, Fletcher32, integer/f32/f64 scale-offset, masks and fill markers; registered caller-defined encoders/decoders with named v2 pipeline entries and per-chunk mask skips |
| Datatypes | Fixed-point, IEEE/normalized floats, strings, compound, enum, array, time, bitfield, opaque, sequence, references, complex |
| References | 4-/8-byte object references, legacy region references and version-4 object/region/attribute references against compact, contiguous and filtered targets |
| Shared messages | Table/indexes/heaps, wrappers v1–v3, field sharing, fills, filter pipelines, whole attributes and committed targets |
| Writer | Superblock v3, OHDR/OCHK, dense metadata, chunk indexes, SOHM, references and external slots |
| In-place patching | Compact, contiguous and filtered or unfiltered chunked datasets, including sparse fill-only chunks and registered custom filters with encoders; filtered chunks are re-encoded in place when they fit or appended at an aligned EOF when they grow, with address, stored size, affected checksums and superblock EOF updated; big-endian scale-offset is preserved through byte-order normalization, and chunk edge padding is preserved |
| Producers | h5py, hdf5storage and netCDF4 fixtures; HDF5 1.14.6 and 2.0 datatype messages |

## Reader findings

Address conversion remains relative to the superblock base. Object headers,
continuation chunks, dense metadata, heap blocks, B-tree nodes and chunk indexes
are bounds checked before use. Metadata checksums are verified. Group traversal
stops recursive hard-link aliases at the current ancestor chain.

`Hdf5File` canonicalizes empty, `.` and `..` path components before traversal,
so relative soft-link targets that walk to a parent group resolve consistently
with `Hdf5Resolver`. Normalization is lexical and bounded by the existing
soft-link indirection limit.

Dataset metadata and payload reads are separate. Selection reads validate rank,
bounds and stride and decode only intersecting allocated chunks. Missing chunks
use fill values. `chunk_iterator` bounds memory by one tile plus an intersecting
storage chunk and supports early termination.

Undefined fill messages and fill-time `NEVER` produce no stored fill payload, so
unallocated contiguous regions and chunks read as zero, matching HDF5 1.14.6.
Custom payloads that are present but disabled are still parsed and size-checked.
Malformed fill messages remain rejected, and external raw data still requires
`Hdf5Resolver` independently of the fill policy.

Variable-length values read global-heap objects on demand. Dataset-region
references resolve a selection against a same-file dataset and reuse the normal
selection and filter path. Null references return `None`; dangling addresses are
rejected. Region results remain memory resident.

Version-4 reference slots are recognized from the datatype width and validated
by type and flag bytes. Region and attribute payloads are fetched from their
global heap with exact blob-size checks. Attribute-reference names are decoded
as UTF-8 and resolved through the native attribute collection. External-file
flags are rejected instead of being misread as local heap identifiers.

`Hdf5Resolver` is the multi-source boundary. It follows chained external links,
rejects cycles, reports unregistered sources and reads contiguous external
raw-data slots across file and element boundaries. `Hdf5File` intentionally does
not dereference external targets or external raw data.

Numeric storage wider than 8 bytes and binary-float normalizations other than
implied MSB are rejected. Attribute value decoding propagates unsupported
datatype errors instead of reporting an empty value. HDF5 2.0 datatype-message
v5 is accepted as forward-compatibility evidence and is outside the 1.14.6
baseline.

## Writer findings

The builder validates absolute paths, duplicates, datatype nesting, shapes,
value sizes, filters, fill values and encoded link/message limits before
encoding. Caller-owned arrays are copied. All object, continuation,
dense-metadata, global-heap, external-slot and raw-data addresses are assigned
before checksummed output is written.

The writer emits a v3 superblock and v2 object headers. Large compact headers use
checksummed OCHK continuations. Dense collections use one FHDB direct block,
one BTHD/BTLF leaf per index and type 6/9 creation-order indexes. They are limited
to 9–65,535 records with encoded heap objects up to 65,535 bytes; indirect/huge
blocks and a free-space manager are not emitted.

Compact datasets embed their payload in the data-layout message and enforce the
65,535-byte limit before encoding. In-place patching locates compact payloads
from parsed message offsets; OHDR and OCHK checksums are recomputed when those
chunks cover the modified bytes. Contiguous patching continues to overwrite only
the assigned payload address.

Chunked patching walks the active index while recording each allocated chunk's
address, stored size, filter mask and writable index location. It covers v1/v2
B-trees, single and implicit chunks, and fixed/extensible arrays. Unfiltered
chunks are overwritten at native size. Filtered chunks are decoded, rewritten
only at in-bounds elements so edge padding survives, and re-encoded with their
existing mask and built-in pipeline. Encodings that fit overwrite the original
payload; larger encodings are collected in one tail buffer, appended at aligned
file offsets, and written after all index updates. Each relocation updates the
index address and size fields, the covering page, node, index or object-header
checksum, and the superblock EOF and checksum. Missing chunks stay sparse only
when patched values equal the effective fill; non-fill values in missing chunks
and caller-defined filter re-encoding remain rejected. Relocation is append-only
and does not write a free-space manager or reclaim the old payload.

Finite chunked datasets use fixed-array indexes; one unlimited dimension uses
extensible-array indexes, including inline entries, direct/super blocks and paged
data blocks. Multiple unlimited dimensions use bulk-built v2 B-tree chunk indexes
with client type 10/11 records, including internal nodes. `append_dataset`
requires fixed axes to match the current shape, grows every unlimited axis by
the appended block extent and places the block at the old shape's trailing
corner. Fill-only chunks keep undefined addresses, except that variable-length
datasets allocate in-shape chunks so empty strings do not depend on undefined
vlen fill storage.

Integer and floating-point scale-offset, filter masks and filter pipelines are
encoded in the index. Variable-length values accept shuffle and deflate on their
fixed-width heap descriptors; Fletcher32 and scale-offset are rejected for them,
matching HDF5 1.14.6. Custom variable-length string and sequence fills use the
same global heap as dataset values. Sequence bases must be fixed-size; bases
containing region references are rejected because HDF5 1.14.6 cannot safely
consume them.

Chunk filter masks are keyed by chunk-grid coordinates rather than transient
linear indices. Resizing or appending a dataset with multiple unlimited axes can
therefore change chunk strides without moving a mask to a different logical
chunk.

Structured values use the reader's recursive datatype model. References resolve
after final layout, allowing forward targets. Region references use a dedicated
global heap because mixing them with string or sequence payloads can corrupt HDF5
bookkeeping.

## SOHM review

The reader resolves the superblock-extension SOHM table, SMTB indexes and fractal
heaps. Wrapper versions 1–3 cover heap IDs and committed object-header targets.
Dense attribute records referencing SOHM read the payload from the shared heap.
Whole fill and filter-pipeline messages use the same heap-ID path.
Reserved indexes with no entries may omit their fractal heap; an index that
claims entries must provide one. Attribute datatype/dataspace flags are resolved
separately. Cycles, malformed indexes and checksum failures are rejected.

The writer optionally emits one SOHM table and one managed-object heap for whole
attributes, custom fills and encoded filter pipelines up to 65,535 bytes. Equal
encoded fills, attributes and pipelines reuse one heap ID. Named committed
datatypes are written as datatype objects and referenced through wrapper version
3 storage type 2; datasets and attributes with matching encoded datatypes use the
shared flag and target address. This path also works without a
superblock-extension SOHM table.

The remaining SOHM boundary is one table and one heap. Multiple tables, multiple
heaps and heap extension across multiple blocks are not emitted.

## Residual risks

- Detection of unsupported input is not exhaustive.
- Whole-dataset and single-selection results are memory resident.
- Custom filter callbacks are limited to dataset chunk pipelines. Writing and
  patching require a registered encoder; metadata-heap filters and overrides of
  built-in filter IDs are not accepted.
- External raw data requires caller registration and caller-managed files.
- Dense and SOHM output use compact single-block writer forms; larger producer
  structures are reader-side features unless explicitly listed as writer output.
- V2 B-tree output is bulk-built for a completed in-memory shape; arbitrary
  incremental insertion at interior coordinates is not exposed.
- Dot-segment path normalization is lexical; combinations where `..` follows a
  soft-link component are not fixture-verified against every HDF5 path rule.
- Version-4 references to external files and high-level netCDF-4/MATLAB
  semantics remain outside the profile.

## Milestones

- 2026-10-19: initial parser, framing, groups, layouts, filters and CLI audit.
- 2026-10-20: selection reads, datatype decoding, writer MVP and examples.
- 2026-10-21: release gate, source reads, soft/external links.
- 2026-10-23: HDF5 1.14.6 datatypes, dense writer metadata, resolver/streaming,
  NetCDF-4 producer interoperability and v1.0 profile.
- 2026-10-27: numeric layouts, scale-offset, SOHM reader, external raw data,
  references, big-endian and structured writer extensions.
- 2026-10-28: hard-link traversal, MATLAB-compatible fixture and vlen string fills.
- 2026-10-29: structured references, region references, chunk masks, external
  slots, creation order, complex values, large dense objects, SOHM output,
  fill/attribute deduplication and committed datatypes.
- 2026-10-30: multi-unlimited writer storage and append with v2 B-tree chunk
  indexes, filtered records, internal-node coverage, undefined/disabled fill
  policies, fixture-verified shared filter pipelines and writer-side pipeline
  sharing, plus filtered variable-length strings and sequences and vlen fills.
- 2026-11-04: audit hardening for coordinate-stable chunk filter masks across
  multi-axis resize/append and parent-segment normalization in paths and soft
  links, with native/wasm and HDF5 1.14.6 release verification.
- 2026-11-09: bounded-memory selection iterators, arbitrary-coordinate writer
  slabs, contiguous allocated-value patching and caller-registered dataset
  chunk filters, with native/wasm release verification.
- 2026-11-09: version-4 object, region and attribute reference reading, plus
  opt-in version-4 reference writing with global-heap payloads and
  fixture-backed round-trip tests.
- 2026-11-09: IEEE half-precision writer values and contiguous patching, with
  h5py-verified demo output.
- 2026-11-09: big-endian numeric value patching, including half-precision data.
- 2026-11-09: compact dataset writing and checksum-safe in-place compact patching,
  with h5py-verified demo output and patched compact values re-read by
  h5py 3.13.0/HDF5 1.14.6.
- 2026-11-09: in-place patching for unfiltered chunked datasets across v1/v2
  B-tree, single, implicit, fixed-array and extensible-array indexes, preserving
  sparse fill-only chunks when patched values stay at fill; patched sparse
  output was re-read by h5py 3.13.0/HDF5 1.14.6.
- 2026-11-10: in-place patching for built-in filtered chunks across v1/v2
  B-tree, single, fixed-array, extensible-array and paged indexes, with
  in-place or aligned-EOF re-encoding, address and stored-size updates, EOF and
  checksum repair;
  sparse fill-only chunks remain supported. Scale-offset re-encoding validates
  fill markers and byte-order metadata and covers big-endian integer/f32
  fixtures plus a scale-offset plus deflate pipeline.
- 2026-11-10: caller-defined chunk-filter encoding for new files and filtered
  patching, including optional encoder/name trait methods, named version-2
  entries for IDs above 255, built-in pipeline combinations, chunk-mask skips,
  shared filter pipelines and growth relocation. Decoder-only implementations
  remain source compatible and report a clear unsupported-output error.
