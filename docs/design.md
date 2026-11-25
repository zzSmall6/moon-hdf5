# Design

moon-hdf5 separates framing, navigation, metadata, value decoding and encoding,
while sharing one datatype/value model between reader and writer.

## Layers

| Layer | Responsibility |
| --- | --- |
| Framing | Superblocks, object headers, messages, address widths and checksums |
| Navigation | Groups, links, paths, dense collections and cross-file resolution |
| Metadata | Dataspaces, datatypes, layouts, filters, fills and shared messages |
| Values | Numeric/structured decoding, selections, references and chunk iteration |
| Sources | In-memory bytes, bounded random-access reads and external-file registries |
| Writer/editor | Validation, address planning, complete-file encoding and append-only edits |

## Reader

`Hdf5File::parse` accepts bytes; `parse_source` accepts bounded random-access
storage. File addresses remain `UInt64`. Metadata and value reads are separate:
`read_dataset_metadata` avoids payload I/O, while `read_dataset` decodes only the
requested values. A full dataset or single selection result is materialized;
`chunk_iterator` and `selection_iterator` keep memory bounded.

Groups use symbol tables or v1/v2 link messages. Compact groups store inline
links; dense groups use a fractal heap with v2 B-tree name and creation-order
indexes. Attributes use the same dense machinery. Hard-link traversal stops when
an address repeats in the current ancestor chain. Paths normalize empty and `.`
segments and fold `..` lexically; `Hdf5File` and `Hdf5Resolver` share this path
logic.

Chunked storage is normalized behind one layout model. Readers cover v1
B-trees, single-chunk, implicit, fixed-array, extensible-array and v2 B-tree
indexes, validating headers, signatures, counts, coordinates, addresses and
checksums. Missing or undefined chunks use the effective fill value, including
undefined and `FILL_TIME_NEVER` policies.

Filters use one pipeline model: shuffle, deflate, Fletcher32, N-bit, integer
scale-offset and f32/f64 decimal scale-offset. Per-chunk masks follow filter
order. N-bit decoding restores declared precision and padding bits. Caller
filters implement `Hdf5CustomFilter`; the same registry can decode dataset
chunks and fractal-heap metadata. Pipeline output size and bounds are validated.

Datatype decoding recursively handles fixed-point (including arbitrary precision
through `BigInt`), IEEE and extended floats, fixed/vlen strings, compound, enum,
array, time, bitfield, opaque, sequence, reference and complex classes.
Fixed-point values honor byte order, offset, precision, sign extension and
padding. Variable-length values fetch global-heap objects on demand.

Explicit conversion reuses the same datatype model. `Strict` requires
type-level losslessness and rejects narrowing, fractional integer results,
precision loss and range overflow. `Lossy` applies HDF5-style wrapping,
truncation and zero-fill, while `NoConversion` accepts only identical layouts.
Compound members are matched by name; missing target members require an option
and receive a typed default. Complex class-11, compound and two-element array
forms can bridge to one another when both component types are compatible under
the selected policy. Dataset, selection, tile, page and attribute
`convert_to` methods share the converter; `read_*_as`, typed writer/editor
methods and `chunk_iterator_as` / `selection_iterator_as` expose it across the
API without changing storage layout.

Legacy object references resolve to an object header and deterministic hard-link
path. Legacy dataset-region references resolve a heap selection through the
normal dataset selection path. Version-4 object references carry an address
token; region and attribute references carry a heap payload. `Hdf5File`
resolves same-file targets; `Hdf5Resolver` resolves registered external targets.

Virtual datasets use version-4 layouts whose global-heap mappings describe source
and virtual selections. The reader validates mapping checksums, ranks, bounds,
source registration and datatype compatibility. All, empty and hyperslab
mappings are supported, along with fill gaps, nested virtual datasets and one
unlimited axis with `%b`/`%%` substitutions. Numeric, enum, complex, array,
compound, fixed-string and fixed-to-vlen-string conversions are supported.
Directly addressable all/hyperslab mappings keep page and tile reads bounded;
point and irregular block mappings expand their mapping metadata.

## External sources

`Hdf5Resolver` owns named sources. The primary file is `<root>`; `add_source`
registers external targets. Resolution follows hard, soft and external links,
tries exact, relative and basename matches, rejects cycles and applies an
indirection bound.

Contiguous external raw-data slots use the same registry. Reads concatenate
declared slots and apply each file offset, so a read may cross files or element
boundaries. A short source contributes its prefix and zero-fills the remainder.
`Hdf5File` leaves external links and raw data unresolved.

## Shared messages

The reader resolves the superblock-extension SOHM table through SMTB indexes and
fractal heaps. Wrapper versions 1–3 cover heap IDs and committed object-header
targets. Dense attribute records may reference SOHM heap objects, and fill or
filter-pipeline messages use the same heap-ID path. A reserved index without a
heap is accepted when empty; an index claiming entries must provide one.
Cycles, malformed indexes and checksum failures are rejected.

The writer optionally emits one SOHM master table with per-message-type indexes
and managed heaps, interning equal messages. Committed datatypes use wrapper
version 3 type 2. The format exposes only one master table, which is the output
boundary.

## Writer

`Hdf5Writer` is a value-building API. Adding an object validates paths, names,
shapes, datatypes, values, filters and fill values immediately. Encoding plans
all addresses before writing checksummed metadata. Output uses superblock v3,
v2 object headers and checksummed OCHK continuations.

`write_to` accepts a random-access `Hdf5Sink` and emits metadata and payloads as
they are encoded, without retaining a complete output image. The sink contract
allows out-of-order writes, zero-filled gaps and final length truncation.
`encode` is the convenience path that writes to `Hdf5MemorySink` and returns a
snapshot. `Hdf5File::copy_to` uses the same sink to copy bytes- or source-backed
files in 1 MiB chunks.

`Hdf5ChunkSource` backs finite and extensible chunked datasets with repeatable
edge-clipped chunks. Address planning records just the stored size of each
chunk; final emission regenerates and filters one chunk at a time directly into
the sink. The complete dataset and the complete set of compressed chunks are
therefore never resident. Variable-length values are pre-scanned to build the
global heaps. Source-backed datasets are immutable in the builder because its
resize and patch APIs require resident values.

`Hdf5ValueSource` provides the same bounded-memory model for contiguous
datasets. Logical blocks are edge-clipped and requested in row-major order;
each encoded block is written into its row-major destination, so a partial
block never overwrites adjacent dataset rows. Block dimensions affect memory
use only, not the on-disk contiguous layout. Variable-length and structured
blocks are likewise pre-scanned for global-heap planning.

The three `*_from_source_typed` methods wrap their sources with per-block
datatype conversion using the existing strict, lossy or no-conversion policy.
Prototype and fill conversion happen at registration; block conversion errors
are reported during output planning or emission.

`set_user_block` reserves a zero-filled 512 B–1 GiB prefix;
`set_user_block_content` stores caller bytes in the smallest legal size and
zero-pads the remainder. Layout addresses are relative to the user-block base,
while the v3 superblock records the base and an absolute EOF.

New objects include groups, contiguous, compact, chunked, external and virtual
datasets, committed datatypes, attributes and links. Compact payloads are
limited to 65,524 bytes. Chunked output supports fixed-array indexes, an
extensible-array index for one unlimited axis and a v2 B-tree for multiple
unlimited axes. The v2 tree is bulk-built from the final chunk set. Source-backed
chunks use the same indexes but keep only per-chunk sizes during planning.

Numeric output covers signed/unsigned integers, IEEE float16/32/64 and x87
80-bit extended floats in either byte order. Structured encoding recursively
validates compounds, enums, arrays, sequences, variable-length values and
references. Variable-length strings and sequences use 16-byte descriptors and
shared global heaps. References resolve after final layout, so forward targets
work. Region references use a dedicated heap to avoid mixing payload types.
Typed creation and editing methods convert memory values before the normal
writer validation and encoding, so contiguous, compact, chunked, external and
virtual layouts share the same conversion rules.

Built-in filters cover shuffle, deflate, Fletcher32, N-bit and scale-offset.
Caller filters are registered before use, receive the chunk parameters and
may change the stored size. Per-chunk masks are keyed by chunk-grid coordinates,
so resize and append do not move masks between logical chunks.

External dataset creation writes unfiltered raw-data slots for fixed-width values
and legacy object references; variable-length and heap-backed reference values
are rejected. Virtual dataset creation writes a checksummed mapping heap and
layout-v4 message, but records source names without opening or validating them.

Dense links are not capped at 65,535 records and use the format's 64-bit
creation-order field. Dense attributes are capped at 65,535 by the
attribute-info field. Dense output supports multiple direct heap blocks,
indirect roots and huge objects; no free-space manager is emitted.

## Editing

Value patching preserves object identity and layout. Allocated contiguous and
compact payloads are rewritten in place with checksum repair. Unallocated
contiguous payloads are appended and linked through the layout message.
Chunked patching walks the active index, decodes filtered chunks, overlays only
in-bounds elements and preserves edge padding. Relocated chunks are appended at
aligned offsets and all affected index, page, header and superblock checksums are
updated.

Patching can allocate into undefined slots and existing fixed- or
extensible-array pages. It can create missing fixed-array data blocks,
extensible-array index/super/data blocks and missing v1/v2 B-tree records when a
non-fill value requires them. Missing chunks stay sparse when patched values
equal the effective fill. Variable-length and reference patches append heap
payloads before rewriting fixed-width descriptors. Old payloads remain
unreferenced.

`resize_dataset` rewrites only the simple dataspace and keeps chunk metadata.
Shrinking hides chunks; regrowing restores them. `append_dataset` grows
unlimited axes and patches each affected chunk at the old trailing corner.
Variable-length and reference heap prefixes are copied verbatim from the prior
EOF to preserve alignment.

Attribute patching updates inline values or fractal-heap objects in place and
refreshes OCHK/direct-block checksums. Attribute add, rename and delete rebuild
the complete compact or dense collection, converting storage when needed and
preserving available creation order. Link add, rename and delete likewise
rebuild compact/dense links or convert symbol-table groups. Object creation
appends v2 headers and attaches through the same link-rebuild path.
`delete_object` removes every hard link to the target and rejects the root;
soft and external links remain unresolved.

All editing is append-only. Superseded object headers, heaps, symbol tables and
indexes remain unreferenced because physical reclamation is outside the profile.

## Errors and verification

Unsupported structures raise `UnsupportedFeature`; malformed metadata normally
raises `InvalidField` or `UnexpectedEof`; invalid caller input raises
`InvalidArgument`. Detection is not exhaustive.

Tests run against committed h5py/hdf5storage fixtures on wasm and native. The
release gate adds documentation/package checks, writer/editor demos and
independent h5py verification with HDF5 1.14.6. See
[compatibility.md](compatibility.md), [audit.md](audit.md) and
[testdata/README.md](../testdata/README.md).
