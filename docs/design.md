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

Virtual datasets use a version-4 layout whose global-heap object contains the
source/virtual selection mappings. The reader validates the mapping-block
checksum, selection ranks and bounds, source registration, element counts and a
recursive datatype match before copying source elements into the requested
virtual indices. Unmapped elements use the declared fill value. The same path
serves full reads, arbitrary selections, chunk tiles, selection pages and
region-reference targets; nested virtual datasets recurse through the same
dependency graph with a fixed depth bound. All and non-overlapping hyperslab
mappings are addressed directly, so a page request reads only the corresponding
source elements even when the mapping itself contains very many elements. Point
and irregular block mappings retain the materialized mapping path.

The writer emits the same layout-v4 form. Each virtual dataset appends a
checksummed mapping block as a global-heap object, then stores its heap address
and object index in the dataset layout. All, empty and hyperslab selections are
encoded; HDF5 1.14.6 rejects point and block virtual mappings, so the writer
rejects them too. Source files are recorded by name but are not opened or
validated during creation.

Filters are decoded through a shared pipeline model: shuffle, deflate,
Fletcher32, integer scale-offset and f32/f64 decimal scale-offset. Per-chunk
masks follow HDF5 filter order. Non-finite scale-offset inputs are rejected.
Dataset chunk pipelines may also contain caller-registered IDs. Registration is
owned by `Hdf5File` or `Hdf5Resolver`; callbacks receive the filter parameters,
element width and expected decoded size, and the pipeline bounds intermediate
output as well as verifying the final size. The same trait has an optional
encoder used by the writer and value patching; decoder-only implementations keep
working for reads and raise `UnsupportedFeature` when output is requested.
Built-in IDs cannot be overridden. Filters attached to fractal-heap metadata use
the same caller registry, so dense groups, dense attributes and SOHM heap
objects can use caller-supplied decoders.

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
`Hdf5File` rejects references marked as pointing into external files; the same
encoded bytes resolve through `Hdf5Resolver` when the named file is registered.

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
  master table with one index and managed heap per message type, interning
  equal messages within each type and extending heaps through FHIB roots;
- named committed datatypes as ordinary datatype objects referenced through
  wrapper version 3 storage type 2;
- the SMTB index records for the message kinds actually emitted.

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

Dense link and attribute collections are used for 9–65,535 records. Managed
heap objects up to 65,535 bytes are packed into one or more FHDB direct blocks,
with an FHIB indirect root when more than one block is needed. Larger encoded
objects use huge heap IDs and a type-1 v2 B-tree. Each collection has one FRHP
header, a BTHD header and BTLF name index, plus a type 6 or 9 creation-order
index sharing the same heap IDs. Larger sets remain compact and use OCHK
continuation when needed. A free-space manager is not emitted.

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
slots for null references. External version-4 references resolve their targets
from sources registered with `Hdf5Writer::add_external_source`. Object payloads
store the target file's address width, target address, file-name length and
UTF-8 name. Region payloads append the target dataspace rank and encoded
selection; attribute payloads append the UTF-8 attribute name. External slot
flags are one, and the payload address and object index refer to the same
region-reference global heap. Mixed same-file and external references may share
one file.

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

Variable-length UTF-8 string and fixed-base sequence datasets use the same three
layouts. Patching first deduplicates all new object payloads into one global heap
collection whose size is a multiple of 4096 bytes. Unless every value is empty,
that collection is appended at an 8-byte-aligned file offset before any
relocated chunk or rebuilt index payload, so later addresses are computed from
the complete tail. Dataset elements are rewritten as fixed-size references
containing a string byte length or sequence element count, the collection
address and a one-based object index; empty values use a null collection address
and index. Compact values keep their inline references, contiguous values stay
at their address or are allocated with the payload after the heap, and chunked
values are encoded through the normal chunk path. Chunked variable-length data
accepts unfiltered storage, Shuffle or deflate.
The patcher rejects Fletcher32, scale-offset and attempts that would require
creating a missing fixed-array data block or extensible-array metadata. Old heap
collections are not reclaimed.

Legacy object-reference values keep their fixed-width slots. Patching resolves
each non-null path against the file's existing object graph and writes the
target object-header address in file address-width bytes; null values are
zero-filled. Groups, datasets and committed datatypes can be targets. This path
is limited to version-3-or-earlier object references whose slot width equals the
file address size.

Legacy region references use a slot containing a global-heap address and a
one-based object index. Patching first resolves each non-null target to an
existing dataset with a non-null dataspace, then encodes the target object
header address followed by its empty, all, point, hyperslab or block selection.
Equal payloads share one heap object, and the collection is appended before the
dataset payload or relocated chunk data. The fixed slots are then rewritten with
the heap address and object index; null references are zero-filled. The path is
limited to version-3-or-earlier region references whose slot width is the file
address size plus four bytes.

Version-4 references keep their 18-byte slots. Same-file object references write
the reference kind, zero flags, address-token size and target address directly.
Same-file region and attribute references append an indirect payload to the same
global heap mechanism used for variable-length data: a region payload contains
the token size, target-dataset address, selection length, rank and encoded
selection, while an attribute payload contains the token size, target-object
address and UTF-8 attribute name. External object, region and attribute
references instead write flag one plus the payload length, heap address and
object index; their payloads carry the normalized external file name and the
target file's address width. Attribute-reference targets are checked against
the target object's existing attributes. Null slots are zero-filled. Compound,
fixed-array and sequence values recurse through nested references.

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
effective fill element. When an EAIB direct pointer or an EASB block pointer is
undefined, patching appends an EADB containing undefined entries and initial
page or block checksums, writes the new block address into the owning EAIB or
EASB, and refreshes its checksum. Paged blocks also set their page bits in the
existing EASB bitmap. If the EAIB address is undefined, patching creates an EAIB
with undefined inline/direct/super slots, scans the requested payload for
non-fill chunks, derives `maximum_set` when the header watermark is too small,
and appends the required direct EASBs, EADBs and paged block content. It then
updates the EAHD index address, maximum set, secondary/data block counts and
sizes, realized-element count and checksum before normal slot linking. An
undefined EASB pointer is repaired by appending an EASB with an empty bitmap and
undefined child pointers, linking it into the EAIB, and creating only the child
EADBs that contain non-fill data. A non-fill chunk whose v2 B-tree record does
not exist is encoded and appended, after which the complete BTHD/node tree is
rebuilt from the final addresses, stored sizes, masks and coordinates and linked
through the data-layout address. The v1 path is similar: a missing record's
payload is appended, then the complete v1 tree is rebuilt using the
indexed-storage K from the superblock, including a non-default K carried by the
superblock-extension B-tree K message. Rebuilt nodes balance records by capacity,
link same-level left and right siblings, and use termination keys whose final
field is the datatype storage size. The layout address and object-header
checksum are updated after the tree is placed. Indexes without a writable record
path continue to require fill values for absent chunks. Relocation and allocation
are append-only and leave old chunk bytes unreferenced.
`patch_dataset_values` accepts all three layouts, while
`patch_contiguous_dataset` is the contiguous-only compatibility entry point.

`patch_attribute_values` locates attributes through the object header's attribute
info message and its name or creation-order B-tree. Inline values are rewritten
inside the containing object-header chunk, after which the OHDR or continuation
checksum is refreshed. Dense values are written through the managed fractal-heap
object identified by the dense attribute record; the direct block's checksum is
recomputed with the checksum field treated as zero, matching the HDF5 direct
block layout. The attribute message, name, creation order, datatype, dataspace
and index records are not changed. For variable-length UTF-8 strings and
fixed-base sequences, patching deduplicates the new object payloads in a global
heap collection whose size is a multiple of 4096 bytes, appends it at an
8-byte-aligned file offset and rewrites only the fixed-size global heap
references in the value area. String references store byte lengths, while
sequence references store element counts; the fixed-size payload of each
non-empty sequence is encoded from its base datatype, recursively applying the
reference rules when the base contains references. Legacy object references
resolve paths against the existing object graph and overwrite their
address-width slots, with zero-filled nulls. Legacy region references resolve to
existing non-null datasets and append deduplicated target-address plus selection
payloads to the same heap before rewriting their heap-address/index slots.
Same-file version-4 object references write direct address tokens; version-4
region and attribute references append their indirect payloads to the heap,
validate region dataspaces and existing attribute names, and rewrite kind,
payload length, heap address and object index. External version-4 references
resolve against registered sources, validate the same target properties, use
flag one and embed each target file's address width and normalized file name.
The superblock EOF and checksum are updated; all-empty variable-length,
null-only region-reference and all-null version-4 values skip the append.
Compound, fixed-array and sequence values are traversed recursively, so
references and variable-length values in members, nested members, array
elements and sequence elements use the same address-width and heap rules.
Shared attribute messages, filtered heap storage and unregistered external
reference files are rejected. Old global heaps remain unreferenced because no
free-space reclamation is written.

`add_attribute`, `rename_attribute` and `delete_attribute` use a separate
metadata-rebuild path. The reader collects compact and dense attribute messages,
their creation order and their original encoded payloads, then writes a new
complete attribute collection. Up to eight attributes use compact storage; a
larger collection is rebuilt as dense storage, including multiple direct heap
blocks and huge objects when needed. Dense heap and B-tree headers use the
file's address and length widths, so rebuilding works for 4- and 8-byte-address
layouts. Compact and dense shared attribute messages are resolved through SOHM
and emitted as local attribute messages; editing does not mutate the existing
shared-message table or heaps. Added variable-length strings and fixed-base
sequences receive a newly appended global heap before their descriptors are
encoded. Added legacy and version-4 object, region and attribute references
resolve their targets against the existing hierarchy; region targets use the
target dataset's dataspace, attribute targets must already exist, and new
indirect-reference payloads use a second appended global heap. Reference slots
and indirect payload addresses use the file's 4- or 8-byte address width.
Dense rebuilds also preserve an existing fractal-heap filter pipeline and
re-apply it to direct blocks and huge objects. This covers deflate, shuffle,
Fletcher32 and registered custom encoders; metadata scale-offset remains
unsupported. The link rebuild path uses the same filtered-heap encoder.

The object's initial header chunk is replaced by an object-continuation message
followed by NIL padding. Every non-attribute message from the original header and
its continuations is copied into a new continuation chunk appended at an aligned
EOF. Version-2 continuation chunks carry an OCHK checksum and refresh the OHDR
checksum; version-1 chunks use the original message-block layout and update the
header message count. Dense rebuilds append a new fractal heap plus name and
creation-order indexes; compact rebuilds emit an attribute-info message followed
by the attribute messages. Existing creation order is retained, including when a
dense ordered attribute is renamed. The superblock EOF and checksum are updated
after the tail is placed. This path is append-only: old attribute messages,
dense heaps and B-trees remain unreferenced because no free-space manager is
written.

`add_hard_link`, `add_soft_link`, `add_external_link`, `rename_link` and
`delete_link` use the same object-message rebuild machinery. The reader
collects compact link messages or dense-heap link objects, their creation order
and their targets, then re-encodes the complete group link set. Older groups
whose links live in a symbol table are read through the v1 B-tree and local
heap, normalized to hard or soft links, and converted to compact or dense
link-message storage; symbol-table entries do not have a creation order, so
their traversal order becomes the fallback order. Up to eight links use compact
link-info and v1 link messages; larger sets use a newly appended fractal heap
plus name and creation-order v2 B-tree indexes. Dense storage reuses the
multi-block/heap-object encoder, so link payloads larger than 65,535 bytes use
huge heap IDs. Link-info addresses, hard-link target addresses and dense B-tree
pointers all honor the file's 4- or 8-byte address width, including files whose
symbol-table name offsets use the 8-byte length width alongside 4-byte object
addresses. Hard-link targets are resolved to an existing object before
rebuilding; soft and external targets are stored without resolution. Rebuilding
preserves creation order where it exists. When a version 1 object header cannot
hold an eight-byte-aligned continuation message for the rebuilt link messages,
its initial chunk is upgraded in place to a version 2 `OHDR` that continues into
a checksummed `OCHK` chunk. Object creation uses the same attachment path.
`delete_object` walks reachable groups and removes every hard link whose target
address is the requested object, so hard aliases cannot keep it reachable;
soft/external links remain and may dangle, and the root is rejected. Link edits
and deletion are append-only and leave the previous compact messages, dense
heap, symbol tables and indexes unreferenced because no free-space manager is
written.

`add_group` and `add_dataset` append new objects at an aligned EOF and then use
the same link rebuild path to attach them to an existing parent. A new object
uses a version 2 `OHDR` header with a padded initial chunk so later attribute or
link edits can replace it with an aligned continuation. Group headers contain a
compact link-info message. Dataset headers contain dataspace, datatype and
contiguous-layout messages; their payload is appended first and the layout uses
the file's offset and length widths. Fixed-size numeric, string, complex and
structured values are supported. Structured writes recursively encode
variable-length strings and sequences plus object and region references nested
in compound, fixed-array and fixed-base sequence members. Variable-length
descriptors, reference slots and heap addresses use the file address width, so
a descriptor is `4 + address_width + 4` bytes and a legacy object reference is
`address_width` bytes. Object references resolve against the current root and
traversal nodes; the writer uses the same address width for every slot,
including 4-byte-address files where h5py's high-level multi-reference
conversion expects 8-byte slots. The release verifier therefore reads raw
reference slots with the HDF5 low-level API.
Finite chunked datasets append a width-aware fixed-array header, data block and
chunk payloads before the object header; built-in Shuffle, deflate, Fletcher32
and scale-offset messages use the normal filter pipeline. Supplying
`maximum_dimensions` selects a width-aware extensible-array index for one
unlimited axis or a width-aware v2 B-tree for multiple unlimited axes, and the
dataspace advertises those maxima so HDF5 can resize the dataset later. External
datasets append a width-aware local heap for slot names and write the
external-data message plus an undefined contiguous layout. Fill values use the
shared fill message encoder for contiguous, chunked and external storage.
Custom fills for structured values containing variable-length members are
accepted only with eight-byte file addresses; HDF5 1.14.6 requires a fill size
that matches the file datatype while its fill initializer also uses the native
datatype size, so a four-byte structured layout cannot satisfy both and is
rejected instead of producing a file HDF5 may corrupt on resize. Custom filters
for appended chunked datasets are supplied through the `custom_filters`
argument, registered with the same ID/name validation as `Hdf5Writer`, and used
by the fixed-array, extensible-array and v2 B-tree storage encoders. Virtual
dataset creation appends a width-aware mapping heap and layout-v4 message and
supports an optional fill value.
The parent may use link messages or a symbol table; symbol-table parents are
converted during attachment. Created objects participate in later attribute,
link and object-creation edits. Object deletion is available as hard-link
removal; physical free-space reclamation is not implemented.

Structured floating-point datatypes accept IEEE float16/32/64 and x87 80-bit
extended values stored in 10 or 16 bytes. Extended values use explicit-integer
MSB normalization, support both byte orders, and are encoded from `Double` with
their exponent range and signed zero preserved. The same encoder is used for
new datasets, attributes, chunks and in-place value patches.

## Errors and verification

Unsupported structures raise `UnsupportedFeature`; malformed metadata normally
raises `InvalidField` or `UnexpectedEof`; invalid caller input raises
`InvalidArgument`. Detection is not exhaustive.

Tests run against committed h5py/hdf5storage fixtures on wasm and native. The
release gate also runs documentation/package checks, the writer demo and
independent h5py verification with HDF5 1.14.6. Fixture regeneration is in
[testdata/README.md](../testdata/README.md); audit evidence is in
[audit.md](audit.md).
