# Design

moon-hdf5 separates parsing, metadata interpretation, value decoding and
new-file encoding. The same datatype and value model is used by the reader and
writer so structured values do not require parallel implementations.

## Layers

| Layer | Responsibility |
| --- | --- |
| Framing | Superblocks, object headers, messages, address widths and checksums |
| Navigation | Groups, links, paths, dense metadata and cross-file resolution |
| Metadata | Dataspaces, datatypes, layouts, filters, fills and shared messages |
| Values | Numeric/structured decoding, selections, references and chunk iteration |
| Sources | In-memory bytes, bounded random-access reads and external-link registries |
| Writer | Validation, address planning and complete-file encoding |

## Reader

`Hdf5File::parse` accepts bytes; `parse_source` accepts an `Hdf5Source` and keeps
file addresses as `UInt64`. Superblocks, object headers, links, dense indexes,
heaps and selected values use bounded source reads. A full dataset or selection
result is materialized in memory, while `chunk_iterator` bounds memory to one
tile plus an intersecting storage chunk.

Dataset metadata and values are separate paths. `read_dataset_metadata` returns
the dataspace, datatype and layout without touching payload storage.
`read_dataset` reuses that path before decoding values. Selections validate rank,
bounds and stride, preserve point order and duplicates, and decode only
intersecting allocated chunks.

Groups use symbol tables or link messages. Modern groups select compact or dense
storage from link-info addresses; dense groups store links in an FRHP fractal
heap with metadata v2 B-tree name and creation-order indexes. Attributes use the
same dense machinery. Every named hard-link path is discovered, with recursion
stopped when an address repeats in the current ancestor chain.

Paths and relative soft-link targets are normalized by removing empty and `.`
components and folding `..` into the current segment stack before traversal.
This keeps `Hdf5File` and `Hdf5Resolver` behavior aligned.

Chunk indexes are normalized behind `DataLayout` and `DatasetLayoutKind`.
Supported readers cover v1 B-trees, single-chunk, implicit, fixed-array,
extensible-array and v2 B-tree indexes. Headers, block signatures, backlinks,
entry counts, coordinates, offsets and checksums are validated. Missing or
undefined chunks use fill values. Undefined fill messages and fill-time `NEVER`
are treated as no stored payload, so unallocated storage reads as zero, matching
HDF5. A custom fixed-size payload remains parsed and size-checked even when the
writing policy disables its use.

Filters are decoded through a shared pipeline model: shuffle, deflate,
Fletcher32, integer scale-offset and f32/f64 decimal scale-offset. Per-chunk
masks follow HDF5 filter order. Non-finite scale-offset inputs are rejected.
Dataset chunk pipelines may also contain caller-registered IDs. Registration is
owned by `Hdf5File` or `Hdf5Resolver`; callbacks receive the filter parameters,
element width and expected decoded size, and the pipeline bounds intermediate
output as well as verifying the final size. The same trait has an optional
encoder used by the writer and value patching; decoder-only implementations keep
working for reads and raise `UnsupportedFeature` when output is requested.
Built-in IDs cannot be overridden. Filters attached to metadata heaps do not use
the caller registry and still require built-in support.

Datatype decoding recursively handles fixed-point layouts, IEEE and normalized
binary floats, fixed/vlen strings, compound, enum, array, time, bitfield, opaque,
sequence, reference and complex classes. Fixed-point values honor byte order,
bit offset, precision, sign extension and padding. Variable-length values fetch
global-heap objects on demand.

Legacy object references resolve to an object header and deterministic hard-link
path. Legacy dataset-region references resolve a global-heap selection against a
same-file dataset; selected-value reads reuse the normal selection and filter
path. Null references return `None`; dangling targets are rejected.

Version-4 references reuse the same public encoded-byte API. Object references
carry an address token directly; region and attribute references carry a blob
size plus a global-heap address and index. Attribute payloads add a UTF-8
attribute name, and resolved values go through the normal attribute decoder.
The writer selects version-4 encodings only when the caller constructs the
matching datatype; legacy 8-byte objects and 12-byte regions remain the default.
References marked as pointing into external files are rejected until the
resolver-backed reference path is extended.

## External sources

`Hdf5Resolver` owns named `Hdf5Source` values. The primary source is `<root>`;
`add_source` registers external-link targets. Resolution follows hard, soft and
external links, tries exact, relative and basename matches, rejects cycles and
applies an indirection bound.

Contiguous external raw-data storage uses the same registry. Slots are
concatenated in declaration order with each file offset applied, so reads may
cross files and element boundaries. A short source contributes its available
prefix and zero-fills the declared remainder. `Hdf5File` leaves both external
links and external raw data unresolved.

## Shared messages

The reader resolves the superblock-extension SOHM table through its SMTB indexes
and fractal heaps. Wrapper versions 1–3 cover SOHM heap IDs and committed
object-header targets. Dense shared attribute records use SOHM heap IDs directly;
attribute flags may also share embedded datatype or dataspace messages. Whole
fill and filter-pipeline messages resolve through the same SOHM heap-ID path.
HDF5 may reserve a configured index without creating a heap when it has no
entries; that form is accepted, while an index with a nonzero entry count and no
heap is rejected. Cycles, malformed indexes and bad checksums are rejected.

The writer can emit:

- whole attribute, custom fill and filter-pipeline messages through one SOHM
  table and managed heap, interning equal encoded fills, attributes and
  pipelines;
- named committed datatypes as ordinary datatype objects referenced through
  wrapper version 3 storage type 2;
- the SMTB type mask for the message kinds actually emitted.

Field-level attribute sharing is not emitted because every attribute the writer
can represent fits the whole-message path.

## Writer

`Hdf5Writer` is an in-memory builder. Adding an object validates paths, names,
shapes, datatypes, values, filters, fill values and link sizes immediately, and
copies caller-owned arrays. Encoding then computes the complete layout before
writing checksummed metadata.

The output uses superblock v3 and v2 object headers. The initial OHDR chunk uses
a 16-bit size; additional messages continue through checksummed OCHK chunks.
Object, continuation, dense-metadata, global-heap, external-slot and raw-data
addresses are assigned before final encoding.

Numeric encoding covers signed and unsigned 1–8-byte integers and IEEE
float16/32/64 in either byte order. Half-precision values are rounded to
nearest-even from their f32 representation, including subnormals, infinities and
NaNs.

Dense link and attribute collections are used for 9–65,535 records when every
encoded heap object is at most 65,535 bytes. Each collection has one FRHP header,
one FHDB direct block, one BTHD header and BTLF name index, plus a type 6 or 9
creation-order index sharing the same heap IDs. Larger sets remain compact and
use OCHK continuation when needed. Indirect/huge heap blocks and a free-space
manager are not emitted.

The writer supports compact datasets, contiguous datasets, contiguous external
raw-data slots, finite chunked datasets with fixed-array indexes,
one-unlimited-dimension datasets with extensible-array indexes, and
multi-unlimited datasets with v2 B-tree chunk indexes. Compact payloads are
embedded in the layout message and limited to 65,535 bytes. Extensible arrays
use inline, direct, super-block or paged forms. The v2 B-tree is bulk-built in
row-major chunk order, with client type 10 for unfiltered chunks and 11 for
filtered chunks. Fixed axes must match the current shape when appending; each
unlimited axis grows by the block extent and the block is written at the old
shape's trailing corner. Fill-only chunks keep undefined addresses except for
variable-length datasets, which allocate in-shape chunks so empty strings do not
depend on undefined vlen storage.

Integer scale-offset supports automatic or fixed bit widths; floating-point
scale-offset supports f32/f64 decimal digits. Scale-offset must lead the
pipeline, and floating-point scale-offset cannot be combined with Fletcher32.
Variable-length values are filtered as fixed-width heap descriptors: shuffle
and deflate are supported, while Fletcher32 and scale-offset are rejected, as
in HDF5 1.14.6.

Caller-defined filters are registered on `Hdf5Writer` before the affected
dataset is added. A pipeline entry carries only the registered ID and its
little-endian client parameters; the encoder receives the current chunk bytes,
parameters, element width and original decoded chunk size. Version-2 pipeline
encoding writes the optional UTF-8 filter name for IDs at or above 256, and the
same entry works through normal or shared filter-pipeline messages. Encoders can
change the chunk size; the resulting stored size is recorded by each chunk
index.
`set_chunk_filter_mask` records per-chunk skips in the fixed-array,
extensible-array or v2 B-tree index. Masks are keyed by chunk-grid coordinates,
not linear chunk positions, so resize and append preserve their logical target.

Structured values use one recursive encoder. Validation covers nesting depth,
member names and offsets, enum values, array dimensions, compound bounds and
exact encoded sizes. References store paths and resolve to addresses after final
layout, so forward targets work. Region references use a dedicated global heap
because mixing them with string or sequence heap objects can corrupt HDF5 1.14.6
bookkeeping.
Version-4 region and attribute payloads share that dedicated heap. The writer
validates attribute-reference targets before encoding and emits zero-filled
slots for null references.

Variable-length strings and sequences use 16-byte descriptors and one shared
global heap; repeated payloads and custom fills share heap objects. Sequence
bases must be fixed-size. Sequence bases containing region references are
rejected because HDF5 1.14.6 cannot safely consume them.

In-place value patching keeps object identity and layout. Allocated contiguous
payloads are overwritten at their assigned address. Compact payloads are located
from the stored object-header message offset; when the containing OHDR chunk or
OCHK continuation has a checksum, it is recomputed after the bytes change. An
unallocated contiguous payload is collected in a tail buffer, appended at an
aligned file offset, and linked by updating the layout address and contiguous
stored size. The chunked path walks the active chunk index while recording the
stored size, filter mask and writable index location for every represented
chunk, then rewrites only in-bounds elements so padding survives. Unfiltered
chunks use their native size. Filtered chunks are decoded and re-encoded with
the existing mask and registered pipeline. Caller-defined filters are resolved
through the `Hdf5File` registrations and require an encoder. Scale-offset input
is normalized to little endian when the dataset is big-endian, while the stored
filter metadata keeps the original byte order and fill marker. Results that fit
overwrite the original payload and update its size field when needed. Larger
results are collected in the tail buffer, appended at aligned file offsets, then
linked by updating the chunk address and size fields, covering checksums and the
superblock EOF address and checksum.

An undefined chunk slot represented by the active index is handled similarly:
the patch constructs a fill-initialized chunk, overlays the requested values,
applies the pipeline and appends the encoded payload. It then updates the slot
address, stored size and filter mask and recomputes the containing index
checksum. This covers the single-chunk layout and fixed- and extensible-array
entries in existing blocks or pages.
When a fixed-array header exists but its data-block address is undefined,
patching appends an FADB containing undefined entries for every maximum slot.
It initializes the page bitmap and page or whole-block checksums, writes the new
block address into the FAHD, and refreshes the FAHD checksum before linking
chunks through the normal slot path.
For a page whose bitmap bit is clear, patching first initializes all reserved
entries to the undefined address and writes the page checksum. A fixed-array
page then sets its bit and refreshes the FADB checksum; an extensible-array page
sets its bit in the EASB bitmap and refreshes the EASB checksum. The requested
chunk is then linked into that page; other entries continue to read as fill
values.
Missing chunks stay unallocated only when their patched values equal the
effective fill element. Non-fill chunks whose extensible-array data block,
extensible-array super/index block or B-tree record does not exist are rejected;
the patcher does not construct those index structures. Relocation and allocation
are append-only and leave old chunk bytes unreferenced.
`patch_dataset_values` accepts all three layouts, while
`patch_contiguous_dataset` is the contiguous-only compatibility entry point.

## Errors and verification

Unsupported structures raise `UnsupportedFeature`; malformed metadata normally
raises `InvalidField` or `UnexpectedEof`; invalid caller input raises
`InvalidArgument`. Detection is not exhaustive.

Tests run against committed h5py/hdf5storage fixtures on wasm and native. The
release gate also runs documentation/package checks, the writer demo and
independent h5py verification with HDF5 1.14.6. Fixture regeneration is in
[testdata/README.md](../testdata/README.md); audit evidence is in
[audit.md](audit.md).
