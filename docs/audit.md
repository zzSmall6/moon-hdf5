# Code audit — 2026-11-15

Scope: current worktree, reader and writer, with emphasis on the HDF5 1.14.6
v1.0 delivery profile and post-v1.0 extensions.

## Verdict

The MoonBit release checks pass on the current worktree with the stable
MoonBit 0.10.19 toolchain (`moon 0.1.20261102`), `moonbitlang/x@0.5.7` and
`moonbitstack/moonzip@0.4.0`:

- wasm: 395 tests passed;
- native: 395 tests passed;
- interface, formatting, `moon check`, `moon doc` and package checks passed;
- the writer demo and earlier compact and sparse chunked patch output remain
  covered by prior h5py 3.13.0/HDF5 1.14.6 verification, and the new v1 B-tree
  record creation paths were independently read back by the same h5py/HDF5
  versions;
- missing v1 B-tree chunk records are covered for sparse and filtered fixtures,
  including a non-default K=4 tree with 256 chunks, and the complete rebuilt
  tree is re-read correctly by h5py;
- inline, dense, creation-order-tracked, big-endian, variable-length string and
  fixed-base sequence attribute patching is covered by MoonBit round-trip tests;
  compact, dense, ordered, 7000-element and vlen-string output, including empty
  strings, was independently re-read by h5py/HDF5;
- attribute addition, renaming and deletion are covered by MoonBit round-trip
  tests for compact, dense, ordered-dense and v1/v2 object headers, including
  compact-to-dense and dense-to-compact conversion, variable-length strings,
  variable-length sequences, committed datatypes, shared compact/dense
  attributes, 4-byte-address dense rebuilding, duplicate/missing/invalid names
  and source-backed rejection; edited v1, 4-byte dense and shared outputs were
  independently re-read by h5py 3.13.0/HDF5 1.14.6;
- link addition, renaming and deletion are covered by MoonBit round-trip tests
  for compact and dense link-message groups, ordered creation, high creation
  orders, compact/dense conversion, 4-byte addresses, multi-block soft links and
  huge external-link objects; release-demo edits and a separate 4-byte edit
  were independently re-read by h5py 3.13.0/HDF5 1.14.6;
- group and dataset creation are covered by MoonBit round-trip tests for nested
  objects, 4-byte symbol-table parents, subsequent attribute and link edits,
  variable-length UTF-8 strings, fixed-base sequences and object/region
  references, including values nested in compound, fixed-array and sequence
  datatypes in 4- and 8-byte-address files. Contiguous, filtered chunked and
  resizable nested outputs are covered, including the four-byte fill rejection
  boundary; the mixed 4/8 release demo is independently read, filtered and
  resized by h5py 3.13.0/HDF5 1.14.6 through low-level reference-slot reads;
- object deletion removes every hard link to the target while preserving other
  objects and dangling soft links; MoonBit tests cover dataset/group aliases and
  root rejection, and a symbol-table release demo is independently re-read by
  h5py 3.13.0/HDF5 1.14.6;
- finite chunked/filtered and external-storage datasets can be created in
  existing bytes-backed files, with fill values and file-width address/length
  metadata. MoonBit tests cover fixed-array chunks with Shuffle, deflate,
  Fletcher32 and scale-offset, chunked variable-length strings and object
  references, caller-registered custom filters across fixed-array,
  extensible-array and v2 B-tree indexes, external raw-data slots, 4- and
  8-byte files, resolver reads and point selections; mixed-width release-demo
  outputs are independently re-read by h5py 3.13.0/HDF5 1.14.6;
- resizable chunked datasets can be created with `maximum_dimensions`. One
  unlimited axis emits an extensible-array index, multiple unlimited axes emit a
  v2 B-tree, and MoonBit tests cover 4- and 8-byte files with filtered chunks;
  the release demo's datasets are independently resized and written by
  h5py/HDF5 after read-back;
- variable-length UTF-8 string and fixed-base sequence dataset patching is
  covered for compact, contiguous, unallocated contiguous, chunked, Shuffle and
  deflate storage, including empty and repeated values and back-to-back patches;
  h5py-generated contiguous, chunked, filtered and unallocated string inputs and
  compact/contiguous/chunked outputs were independently re-read by h5py/HDF5;
- legacy object- and region-reference dataset and attribute patching is covered
  for compact, contiguous and filtered chunked storage, repeated updates, null
  slots, groups/datasets/committed datatypes, point/hyperslab/block/all/empty
  selections and 4- and 8-byte-address fixtures; non-dataset region targets are
  rejected;
- same-file version-4 object, region and attribute reference patching is covered
  for compact, contiguous and filtered chunked datasets and inline attributes,
  including null slots, repeated indirect heap payloads, object/region/attribute
  target resolution and target-validation errors;
- caller-defined filter encoding, named pipelines, chunk-mask skips, patch
  re-encoding and growth relocation are covered by MoonBit round-trip tests;
- unallocated contiguous payloads and undefined slots in existing single,
  fixed-array and extensible-array indexes are covered by MoonBit round-trip
  tests, including filtered and registered custom-filter chunks;
- unallocated fixed-array pages in an existing data block are covered for both
  unfiltered and filtered indexes, including bitmap, page-checksum and FADB
  checksum refresh;
- missing fixed-array data blocks are covered for unfiltered, filtered, paged
  and non-paged indexes, including FAHD/FADB linkage, page initialization and
  subsequent patches against the new index;
- missing extensible-array data blocks are covered for EAIB direct pointers and
  EASB block pointers, unfiltered and filtered entries, and paged or non-paged
  data blocks, including pointer/checksum updates and subsequent patches;
- missing extensible-array index and super blocks are covered for inline,
  direct, super-block and paged storage, including EAHD maximum-set recovery,
  block counts/sizes, realized elements, pointer checksums and subsequent
  patches;
- unallocated extensible-array pages in an existing data block are covered for
  both unfiltered and filtered indexes, including page checksums and the EASB
  bitmap checksum;
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
| References | 4-/8-byte object references, legacy region references and version-4 object/region/attribute references against compact, contiguous and filtered targets; legacy and same-file version-4 object-, region- and attribute-reference datasets and attributes, including references nested in compound, fixed-array and fixed-base sequence values, can be patched in place |
| Shared messages | Table/indexes/heaps, wrappers v1–v3, field sharing, fills, filter pipelines, whole attributes and committed targets |
| Writer | Superblock v3, OHDR/OCHK, dense metadata, chunk indexes, SOHM, references and external slots |
| In-place patching | Compact, allocated or unallocated contiguous and filtered or unfiltered chunked datasets, including fixed-size, variable-length UTF-8 string, sequence, legacy object-reference, legacy region-reference and same-file version-4 object/region/attribute-reference values, plus references and variable-length values nested in compound, fixed-array and sequence members, sparse fill-only chunks, undefined slots in existing indexes, missing fixed-array data blocks, missing extensible-array data/index/super blocks, unallocated fixed- and extensible-array pages, missing v1/v2 B-tree chunk records and registered custom filters with encoders; unallocated payloads and growing filtered chunks are appended at an aligned EOF, with layout, slot, page-bitmap, FADB/FAHD, EAHD/EAIB/EASB/EADB and B-tree fields, affected checksums and superblock EOF updated; v1 tree rebuilds honor the superblock K, sibling links and datatype-sized termination keys; big-endian scale-offset is preserved through byte-order normalization, and chunk edge padding is preserved; variable-length values use one appended global heap collection and null references for empty values; object references rewrite address-width slots from existing target paths; region references append deduplicated target-address plus selection payloads to a global heap and rewrite heap-address/index slots; version-4 object references write direct address tokens, while version-4 region and attribute references append indirect heap payloads and rewrite kind, length, heap address and index fields |
| Attribute patching | Existing fixed-size inline, dense and creation-order-tracked attributes on objects, including compact attributes, big-endian numeric values, large dense values, variable-length UTF-8 strings, sequences, legacy object references, legacy region references and same-file version-4 object/region/attribute references, plus references and variable-length values nested in compound, fixed-array and sequence members; object-header/OCHK and fractal-heap direct-block checksums are refreshed, variable-length, region-reference and indirect version-4 values append a global heap and update the superblock EOF, while attribute message metadata and indexes remain unchanged |
| Attribute editing | Add, rename and delete on v1/v2 object headers, including compact-to-dense and dense-to-compact conversion, ordered-dense creation-order preservation, variable-length string and sequence addition, committed datatypes, shared compact/dense attributes, 4-byte-address dense heaps/B-trees, aligned continuation chunks, OCHK/OHDR checksum refresh, dense heap/name/order index rebuilding and superblock EOF repair; v1, 4-byte and shared fixture edits are independently readable by h5py 3.13.0/HDF5 1.14.6 |
| Link editing | Add, rename and delete for hard, soft and external links in v2 link-message groups and symbol-table groups, including symbol-table-to-link-message conversion, compact/dense conversion, creation-order preservation, 4-/8-byte target addresses, mixed 4-byte-address/8-byte-length files, in-place version 1 to version 2 object-header upgrades, multi-block dense heaps, huge external-link objects, huge-object v2 B-tree records, aligned continuation chunks, OCHK/OHDR checksum refresh and superblock EOF repair; writer-demo edits and separate 4/8-width compact and dense edits are independently readable by h5py 3.13.0/HDF5 1.14.6 |
| Object creation | `add_group`, fixed-size contiguous `add_dataset`, finite or resizable `add_chunked_dataset` and `add_external_dataset` append version 2 object headers to bytes-backed files, link them into the parent, honor 4-/8-byte address and length widths, support v1/v2 and symbol-table parents, and remain editable through the attribute, link and object-creation APIs; datasets support numeric, fixed-string, complex, fixed-size structured, variable-length UTF-8 string, variable-length sequence, object-reference and region-reference values, including members nested in compound, fixed-array and fixed-base sequence datatypes, with variable-length descriptors, reference slots and heap addresses sized to the file address width; all creation forms accept fill values, chunked creation supports fixed-array indexes, extensible-array indexes for one unlimited axis and v2 B-tree indexes for multiple unlimited axes, built-in Shuffle, deflate, Fletcher32 and scale-offset filters, caller-registered custom filters passed through `custom_filters`, plus nested and top-level vlen/reference snapshots, and external creation writes unfiltered fixed-size raw-data slots through an appended external-data local heap. MoonBit tests cover nested group/dataset creation, 4-byte symbol-table parents, subsequent attribute/link edits, variable-length strings and sequences, object and region references, filtered chunk values, custom filters on all three chunk-index paths, vlen/reference chunks, resizable maximum dimensions and external resolver/selection reads in 4- and 8-byte files; the mixed 4/8 release demo is independently readable and resizable by h5py 3.13.0/HDF5 1.14.6, with raw reference slots used because h5py's high-level multi-reference conversion misreads 4-byte slots |
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

Fixed-point storage wider than 8 bytes is decoded without truncation into
`BigInt` values, including bit offsets and both byte orders. Extended binary
floating layouts, including x87 80-bit storage and explicit- or implied-MSB
normalization, are converted to `Double`. The writer emits wide integers through
`BigIntegerValues` and structured `BigInteger`/`BigTime`/`BigBitfield` values;
extended floating-point output remains outside the writer boundary. Attribute
value decoding propagates unsupported datatype errors instead of reporting an
empty value. HDF5 2.0 datatype-message v5 is accepted as forward-compatibility
evidence and is outside the 1.14.6 baseline.

## Writer findings

The builder validates absolute paths, duplicates, datatype nesting, shapes,
value sizes, filters, fill values and encoded link/message limits before
encoding. Caller-owned arrays are copied. All object, continuation,
dense-metadata, global-heap, external-slot and raw-data addresses are assigned
before checksummed output is written.

The writer emits a v3 superblock and v2 object headers. Large compact headers use
checksummed OCHK continuations. Dense collections use FHDB direct blocks, an
FHIB indirect root when multiple blocks are needed, and a type-1 v2 B-tree for
huge objects. Multi-level BTHD/BTIN/BTLF name and type 6/9 creation-order
indexes are limited to 9–65,535 records. A free-space manager is not emitted.

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
when patched values equal the effective fill; non-fill values require a writable
slot or page and are rejected otherwise. Caller-defined filters can be
re-encoded when registered with an encoder. Relocation is append-only and does
not write a free-space manager or reclaim the old payload.

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

Created variable-length string datasets append their global heap before the
contiguous payload and encode { length, heap address, heap index } descriptors
with the file's address width. Created object-reference datasets map the root
and every traversed object to a writable address and write each reference slot
with that same width. In 4-byte-address files the on-disk slot is four bytes even
though HDF5 reports a host-sized reference type; the release verifier reads the
raw slots through `H5Dread` with `STD_REF_OBJ` instead of relying on h5py's
NumPy multi-reference conversion.

Created chunked datasets encode finite shapes through a fixed-array index and
apply the requested built-in filter pipeline when each chunk is written. Fill
values are encoded with the same datatype rules as dataset values, including
global-heap allocation for top-level variable-length strings. Created external
datasets append an external-data local heap and encode one unmodified raw-data
slot per element; slot name offsets use the file's length width, and both the
external-data message payload and local-heap data offset are eight-byte aligned.
All new dataset layouts use the file's offset and length widths, including
fixed-array headers and entries, external slot addresses and heap offsets.

## SOHM review

The reader resolves the superblock-extension SOHM table, SMTB indexes and fractal
heaps. Wrapper versions 1–3 cover heap IDs and committed object-header targets.
Dense attribute records referencing SOHM read the payload from the shared heap.
Whole fill and filter-pipeline messages use the same heap-ID path.
Reserved indexes with no entries may omit their fractal heap; an index that
claims entries must provide one. Attribute datatype/dataspace flags are resolved
separately. Cycles, malformed indexes and checksum failures are rejected.

The writer optionally emits one SOHM master table with independent fill, filter
and attribute indexes and managed-object heaps. Messages up to 65,535 bytes are
shared; equal messages within one type reuse a heap ID. The heap encoder uses
FHDB direct blocks and FHIB indirect roots when the managed objects exceed one
64 KiB direct block. Named committed datatypes are written as datatype objects
and referenced through wrapper version 3 storage type 2; datasets and attributes
with matching encoded datatypes use the shared flag and target address. This path
also works without a superblock-extension SOHM table.

The remaining SOHM boundary is one master table, which is the only table the
superblock format can reference. Independent indexes and heaps cover the shared
message types emitted by this writer.

## Residual risks

- Detection of unsupported input is not exhaustive.
- Whole-dataset and single-selection results are memory resident.
- Custom filter callbacks cover dataset chunk and fractal-heap metadata decoder
  pipelines. Writing and patching require a registered encoder; overrides of
  built-in filter IDs are not accepted.
- External raw data requires caller registration and caller-managed files.
- SOHM output uses one master table with typed indexes and multi-block heaps.
  Dense metadata output supports multiple direct blocks and huge objects but does
  not emit a free-space manager or tiny heap objects.
- V2 B-tree output is bulk-built for a completed in-memory shape; arbitrary
  incremental insertion at interior coordinates is not exposed.
- Value patching can allocate raw data only into existing writable layout or
  chunk-index storage, including a missing fixed-array data block, a missing
  extensible-array data, EAIB or EASB block, an unallocated fixed- or
  extensible-array page inside an existing data block, or a missing v1/v2 B-tree
  chunk record. Missing B-tree records cause the complete tree to be rebuilt
  from the final records; v1 uses the superblock K and v2 preserves its node
  size and split/merge thresholds. Rebuilding a missing EAIB may create its
  direct/super hierarchy and derives or extends the EAHD maximum set. It does
  not create unrelated metadata structures.
- Variable-length patching accepts string datatypes and sequences whose base is
  fixed-size, including variable-length values nested in compound, fixed-array
  and sequence members. Reference-bearing bases are encoded recursively and
  share the same same-file target rules as other nested references; their
  indirect payloads and sequence objects share one global heap collection.
  Chunked vlen data supports unfiltered storage, Shuffle or
  deflate, not Fletcher32 or scale-offset, and cannot create a missing
  fixed-array data block or extensible-array metadata. Each patch appends a new
  global heap collection whose size is a multiple of 4096 bytes; old heaps and
  relocated values remain unreferenced because no free-space reclamation is
  written.
- Legacy object-reference patching accepts only version-3-or-earlier object
  references whose slot width equals the file address size. Every non-null path
  must resolve to an object already present in the file; null slots are
  zero-filled. Legacy region-reference patching accepts version-3-or-earlier
  slots whose width is the address size plus four bytes. Every non-null target
  must be an existing dataset with a non-null dataspace, and point, hyperslab,
  block, all and empty selections are supported. Same-file version-4 object,
  region and attribute references accept 18-byte slots, write only flags zero,
  and validate dataset and existing-attribute targets before appending indirect
  heap payloads. Compound, fixed-array and sequence values recurse through
  nested members and elements, including nested variable-length values.
  External-file version-4 references remain read-only.
- Attribute patching is limited to replacing values of existing fixed-size,
  variable-length UTF-8 string, fixed-base sequence, legacy object-reference,
  legacy region-reference or same-file version-4 object/region/attribute
  reference attributes. Object-reference slots must match the file address size;
  region-reference slots must be four bytes wider and target existing non-null
  datasets. References and variable-length values nested in compound,
  fixed-array and sequence members are patched recursively. Version-4
  attribute targets must already exist, and external-file flags are rejected,
  as are shared attribute messages and filtered fractal-heap storage for direct
  value patching. Attribute addition,
  deletion and renaming use a rebuild path that supports v1/v2 object headers,
  compact/dense conversion, ordered-dense creation order, shared compact/dense
  attribute messages (rebuilt as local messages), 4-byte-address dense heaps and
  legacy or version-4 object/region/attribute-reference additions, including
  references nested in compound and fixed-array members.
  Added reference values must target existing same-file objects, datasets or
  attributes and use the file's address width. Registered custom filters can
  decode existing dense metadata heaps. Rebuilt dense heaps preserve the
  existing filter pipeline and re-apply it to direct blocks and huge objects;
  deflate, shuffle, Fletcher32 and registered custom encoders are supported,
  while metadata scale-offset is rejected. Old attribute metadata and global
  heaps are not reclaimed.
- Link editing also accepts symbol-table groups and converts them to compact or
  dense link-message storage. Hard-link targets must already exist in the same
  bytes-backed file, while soft and external targets are stored without
  resolution. `delete_object` removes every hard link to the resolved object,
  leaves soft/external links untouched and rejects the root group;
  external-target creation and physical free-space reclamation are not
  performed. Compact, dense and symbol-table link metadata is rebuilt from the
  final link set, so old link messages, heaps, symbol tables and indexes remain
  unreferenced.
- Object creation appends empty groups, fixed-size contiguous datasets, finite
  or resizable chunked datasets and unfiltered fixed-size external datasets,
  including variable-length UTF-8 strings, sequences and object/region
  references, recursively nested in compound, fixed-array and fixed-base
  sequence values.
  Contiguous and chunked creation accept fill values; chunked creation supports
  built-in Shuffle, deflate, Fletcher32 and scale-offset pipelines, while
  external creation writes raw-data slots and an external-data local heap.
  `maximum_dimensions` selects an extensible-array index for one unlimited axis
  or a v2 B-tree for multiple unlimited axes, allowing subsequent HDF5
  resizing. Addresses, lengths, chunk entries and external heap fields follow
  the file's 4- or 8-byte widths. Four-byte files reject custom fills for
  structured values containing variable-length members because HDF5 1.14.6
  cannot represent the fill size consistently. Appended chunked datasets accept
  caller-registered filters through `custom_filters`; virtual storage and
  physical free-space reclamation are not supported. New parents can be edited
  afterwards, and superseded parent metadata remains unreferenced because no
  free-space manager is written.
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
- 2026-11-10: raw-data allocation for unallocated contiguous datasets and
  undefined slots in existing single, fixed-array, extensible-array and v1/v2
  B-tree indexes, including filtered and registered custom-filter chunks, with
  layout or slot updates and checksum/EOF repair. Missing fixed-array blocks,
  extensible-array pages and B-tree records remained outside that first
  allocation boundary.
- 2026-11-10: allocation into unallocated fixed-array pages inside existing data
  blocks, including whole-page initialization, bitmap and FADB checksum updates,
  filtered entries and page checksums.
- 2026-11-10: allocation into unallocated extensible-array pages inside existing
  data blocks, including whole-page initialization, EASB bitmap updates, page and
  EASB checksums and filtered entries. Missing fixed- or extensible-array data
  blocks, super/index blocks and B-tree records remain outside the boundary.
- 2026-11-10: creation of a missing fixed-array data block when the FAHD already
  exists, including undefined entry initialization, bitmap and page/block
  checksums, FAHD address and checksum updates, and later patching through the
  new index. Missing extensible-array data blocks, extensible-array super/index
  blocks and B-tree records remain outside the boundary.
- 2026-11-11: creation of missing extensible-array data blocks linked from an
  existing EAIB direct pointer or EASB block pointer, including undefined entry
  initialization, paged and non-paged block checksums, EASB page-bitmap updates,
  pointer and parent-checksum updates, and later patching through the new index.
  Missing EAIB/EASB or other super/index blocks and B-tree records remained
  outside the boundary.
- 2026-11-11: creation of missing extensible-array index and super blocks,
  including EAHD `maximum_set` recovery from non-fill payload slots, direct,
  super-block and paged child storage, block-count/size and realized-element
  accounting, parent/index checksums, and later patching through the rebuilt
  hierarchy. V1/v2 B-tree record creation remained outside that boundary.
- 2026-11-11: creation of missing v2 B-tree chunk records, including filtered
  variable-width stored sizes and masks, complete BTHD/leaf/internal tree
  rebuilds with the index's node size and split/merge thresholds, data-layout
  address and object-header checksum updates, and later patching through the
  rebuilt tree. V1 B-tree record creation remained outside that boundary.
- 2026-11-11: creation of missing v1 B-tree chunk records, including sparse and
  filtered datasets, complete multi-level rebuilds with the superblock's
  indexed-storage K, balanced same-level nodes, sibling links and datatype-sized
  termination keys, plus layout-address, object-header-checksum and superblock
  EOF updates. Default-K output and a non-default K=4 tree with 256 chunks were
  independently re-read by h5py 3.13.0/HDF5 1.14.6.
- 2026-11-11: in-place attribute value patching for existing fixed-size inline
  and dense attributes, including creation-order indexes, compact attributes,
  big-endian numeric values, managed fractal-heap writes and direct-block
  checksum repair. Compact, dense, ordered and 7000-element output was
  independently re-read by h5py 3.13.0/HDF5 1.14.6.
- 2026-11-11: variable-length UTF-8 string attribute patching for inline and
  dense storage, including empty values and repeated updates. Patching appends
  a global heap collection whose size is a multiple of 4096 bytes, rewrites
  fixed-size heap references and updates the superblock EOF/checksum. Compact and dense h5py
  3.12.1/HDF5 1.14.6 inputs were patched and independently re-read.
- 2026-11-11: variable-length UTF-8 string dataset patching for compact,
  contiguous and chunked storage, including unallocated contiguous data, empty
  and repeated values, Shuffle/deflate pipelines, appended global heaps and
  repeated patches. h5py 3.12.1/HDF5 1.14.6 inputs and outputs were independently
  re-read. Shuffle pipelines with omitted client data, as written by HDF5, are
  accepted.
- 2026-11-13: variable-length sequence patching for datasets and attributes
  with fixed-size, reference-free bases, including empty and repeated sequences,
  contiguous and filtered chunked storage, global-heap deduplication and
  back-to-back patches. The path is covered by MoonBit round-trip tests; nested
  vlen and reference-bearing sequence bases remain rejected.
- 2026-11-13: legacy object- and region-reference patching for datasets and
  attributes, including repeated updates, null slots, group/dataset/committed
  datatype targets, point/hyperslab/block/all/empty region selections and 4- and
  8-byte address files. Region payloads append deduplicated target-address plus
  selection objects to a global heap and rewrite the fixed heap-address/index
  slots. Version-4 reference datatypes and references nested in compound,
  array or sequence datatypes remain rejected. The path is covered by MoonBit
  round-trip tests against writer output and the 32-bit reference-layout
  fixture.
- 2026-11-13: same-file version-4 object, region and attribute reference
  patching for datasets and attributes, including compact, contiguous and
  filtered chunked storage, direct object tokens, indirect global-heap region
  and attribute payloads, existing-attribute validation and null slots.
  External-file flags and references nested in compound, array or sequence
  datatypes remain rejected.
- 2026-11-15: compound, fixed-array and fixed-base sequence nested-reference
  patching for datasets and attributes, including legacy and same-file
  version-4 object, region and attribute references, null slots, deduplicated
  heap payloads and 4-byte legacy layouts. At that stage nested variable-length
  values were still outside this patch path; the later recursive-heap entry
  extends both patching and object creation to them. Parsed fixed-string padding
  is now honored when patch values are encoded.
- 2026-11-15: attribute addition, deletion and renaming on bytes-backed files,
  including compact/dense storage conversion, ordered-dense creation-order
  preservation, v1 and v2 object-header continuation rewrites, variable-length
  string attributes, dense index rebuilding and superblock EOF repair. Dense
  rebuilding now uses the file's address/length widths, including 4-byte-address
  files, and resolves compact and dense shared attributes into local messages.
  MoonBit coverage includes v1, 4-byte-address and shared-message fixtures plus
  invalid-name and missing-target errors; edited v1, 4-byte dense and shared
  outputs were independently re-read by h5py 3.13.0/HDF5 1.14.6.
- 2026-11-15: variable-length strings and sequences nested in compound,
  fixed-array and sequence datatypes can be patched in datasets and attributes,
  including chunked Shuffle/deflate storage and deduplicated recursive global
  heap payloads. Structured writes expose `Datatype::write_vlen_string` and
  accept vlen strings nested in compound and array values. MoonBit round-trip
  tests cover datasets, attributes, filtered chunks and recursive heap objects;
  h5py 3.13.0/HDF5 1.14.6 independently reads the writer demo's nested vlen
  datasets and attributes.
- 2026-11-15: dense writer name and creation-order indexes now build multi-level
  v2 B-trees with BTIN internal nodes instead of sizing every index to one
  BTLF leaf. New writes and dense attribute rebuilding share the same balanced
  encoder, including 4-byte address and length widths. A 1,400-link dense group
  and a 2,500-attribute collection are independently re-read by h5py
  3.13.0/HDF5 1.14.6; MoonBit tests exercise link and attribute indexes whose
  roots are internal nodes.
- 2026-11-15: dense metadata writer storage now spans multiple FHDB direct
  blocks with an FHIB indirect root and stores encoded messages larger than
  65,535 bytes as huge heap objects indexed by a type-1 v2 B-tree. New writes
  and dense attribute rebuilding share the same storage path. MoonBit tests
  cover a multi-block 600,000-byte attribute collection and a huge attribute
  message; writer demo output with dense collections remains independently
  readable by h5py 3.13.0/HDF5 1.14.6.
- 2026-11-15: hard, soft and external links can be added, renamed and deleted
  in bytes-backed v2 link-message groups. Rebuilding preserves creation order,
  converts between compact and dense storage, honors 4- and 8-byte address
  widths, and reuses multi-block and huge heap object encoding for large link
  messages. MoonBit tests cover add/rename/delete, compact-to-dense-to-compact
  conversion, duplicate/missing/invalid names, 4-byte addresses, high creation
  orders, multi-block soft links and a huge external link. The writer demo's
  edited hard/soft/external links and a separate 4-byte-address edit are
  independently re-read by h5py 3.13.0/HDF5 1.14.6.
- 2026-11-15: empty groups and fixed-size contiguous datasets can be created in
  bytes-backed files through `add_group` and `add_dataset`. New version-2 object
  headers use the file's address/length widths, are attached through the link
  rebuild path, and remain editable afterwards. MoonBit tests cover nested
  creation, 4-byte symbol-table parents and later attribute/link edits; release
  demo output is independently re-read by h5py 3.13.0/HDF5 1.14.6.
- 2026-11-15: `delete_object` removes every hard link that names an object so
  aliases cannot keep it reachable, while soft/external links are left dangling
  and the root group is rejected. MoonBit tests cover dataset/group aliases and
  root rejection; a symbol-table release demo is independently re-read by h5py
  3.13.0/HDF5 1.14.6. Superseded object storage remains unreferenced because
  writes are append-only.
- 2026-11-15: `add_dataset` now creates top-level variable-length UTF-8 string
  and object-reference datasets. String payloads use an appended global heap,
  and variable-length descriptors plus reference slots use the file's address
  width, including 4-byte-address files. MoonBit tests cover both address widths
  and null references; 4- and 8-byte release demo outputs are independently
  re-read by h5py 3.13.0/HDF5 1.14.6 through raw reference-slot reads.
- 2026-11-15: finite chunked/filtered and external-storage datasets can be
  created in existing bytes-backed files through `add_chunked_dataset` and
  `add_external_dataset`, and every creation form accepts a fill value. Chunked
  creation emits fixed-array indexes with built-in Shuffle, deflate, Fletcher32
  and scale-offset pipelines, including top-level variable-length string and
  object-reference snapshots. External creation appends an external-data local
  heap and writes unfiltered fixed-size slots. MoonBit tests cover 4- and
  8-byte address/length widths, filters, fills, vlen/reference chunks, resolver
  reads and point selections; release-demo outputs are independently re-read by
  h5py 3.13.0/HDF5 1.14.6. The format investigation also confirmed that external
  slot name offsets use the length width, and both the external-data message
  payload and local-heap data offset are eight-byte aligned.
- 2026-11-15: chunked creation accepts `maximum_dimensions`. One unlimited axis
  emits a width-aware extensible-array index; multiple unlimited axes emit a
  width-aware v2 B-tree. MoonBit tests cover filtered resizable datasets in both
  4- and 8-byte files, and the release demo's extent, fill and chunk metadata are
  independently read, resized and written by h5py 3.13.0/HDF5 1.14.6.
- 2026-11-15: object creation now recursively encodes variable-length UTF-8
  strings, fixed-base sequences, legacy object references and legacy region
  references nested in compound, fixed-array and sequence datatypes. Variable-
  length descriptors, reference slots, region payload slots and global-heap
  addresses use the file's 4- or 8-byte address width. Contiguous, filtered
  chunked and resizable chunked outputs are covered by MoonBit tests in both
  address layouts, including null references and empty variable-length values.
  Fill encoding uses the native datatype size as a lower bound and the reader
  truncates longer stored fills to the file datatype size. HDF5 1.14.6 requires
  incompatible fill sizes for four-byte structured values containing vlen
  members, so those custom fills are explicitly rejected; eight-byte nested
  fills survive reopen and HDF5 resize, with the new gap reading back as the
  fill record. The release demo's nested values are independently read,
  filtered and resized by h5py 3.13.0/HDF5 1.14.6.
- 2026-11-15: existing-file `add_chunked_dataset` accepts caller-registered
  `Hdf5CustomFilter` implementations through `custom_filters`. The registry
  validates IDs, names and duplicates before encoding the pipeline, and the
  filter is applied through fixed-array, extensible-array and v2 B-tree chunk
  storage alike. MoonBit tests create and read custom-filtered datasets for all
  three index paths, including a named ID above 255, and cover duplicate,
  built-in-ID and unregistered-filter errors.
