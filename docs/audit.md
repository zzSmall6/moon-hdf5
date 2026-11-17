# Code audit — 2026-11-17

Scope: current worktree, reader and writer, against the HDF5 1.14.6 common-file
profile and the documented post-profile extensions.

## Verdict

The release gate passes with MoonBit 0.10.19 (`moon 0.1.20261102`),
`moonbitlang/x@0.5.7` and `moonbitstack/moonzip@0.4.0`:

- wasm: 467 tests passed;
- native: 467 tests passed;
- interface, formatting, `moon check`, `moon doc` and package checks passed;
- h5py 3.13.0/HDF5 1.14.6 independently read, resized and patched release-demo
  output for all documented storage and metadata paths, including typed
  datatype conversion in both directions.

No known common-file feature gap remains inside the boundary in
[compatibility.md](compatibility.md). Unsupported input is rejected rather than
silently reinterpreted.

## Verification coverage

| Area | Evidence |
| --- | --- |
| Framing and navigation | Superblocks v0–v3, user blocks, 2/4/8-byte address/length combinations, compact/dense groups, hard/soft links, path normalization and source-backed reads; edited v0 and new v3 user-block files checked by h5py |
| Datasets and indexes | Compact, contiguous, external raw data, virtual storage and every supported chunk index; sparse/undefined chunks, resize and append; h5py reads release demos after writer and editor output |
| Virtual data | Multiple/strided mappings, fill gaps, nested virtual datasets, one unlimited axis, 4-/8-byte files and documented datatype conversions; new-file and edited outputs read by h5py |
| Filters and fills | Shuffle, deflate, Fletcher32, N-bit, integer/f32/f64 scale-offset, masks, fill markers and custom encoders/decoders across indexes and shared pipelines |
| Datatypes and values | Fixed-point, IEEE/extended floats, strings, compound, enum, array, time, bitfield, opaque, sequence, references and complex values, including nested variable-length/reference members; strict/lossy/no-conversion policies, complex-to-array/compound bridging, typed read/write/edit APIs and bounded conversion iterators |
| References | Legacy and version-4 object/region/attribute references in datasets and attributes; same-file and registered external targets; 4-/8-byte slots; compact, contiguous and filtered targets |
| Shared and dense metadata | SOHM table/indexes/heaps, wrappers v1–v3, shared attributes/fills/filter pipelines, dense direct/indirect blocks, huge objects and metadata v2 B-trees |
| Editing | Value and attribute patching; dataset resize/append; attribute/link add, rename and delete; object creation/deletion; v1/v2 headers and symbol-table conversion; release outputs independently read by h5py |

## Residual risks

- Detection of unsupported input is not exhaustive. Unsupported structures raise
  `UnsupportedFeature`; malformed input raises `InvalidField` or
  `UnexpectedEof`.
- Full datasets and single selection results are memory resident. Use
  `chunk_iterator` or `selection_iterator` for bounded memory.
- Editing is append-only. Superseded bytes, old heaps and rebuilt metadata are
  not reclaimed because no free-space manager is emitted.
- Value patching allocates only into existing writable layout or reconstructs
  missing chunk-index records. It does not create unrelated metadata.
- External raw data and external reference targets require caller-registered,
  caller-managed files. External raw storage is contiguous and unfiltered.
- Custom filters need registered decoders for reads and encoders for writes or
  patches. Built-in filter IDs cannot be overridden.
- Explicit conversion is not an unconditional cross-type cast. Reference kinds
  and widths must match, target enums must contain the converted value, and
  virtual-source vlen/reference conversions remain outside that mapping boundary.
- Virtual datasets support the documented reader conversions and one unlimited
  axis. HDF5 1.14.6 point/block mappings, multiple unlimited axes and vlen or
  reference source conversions remain unsupported.
- Dense attributes are capped at 65,535 records by the format; dense links use
  the 64-bit creation-order field and exceed that limit. Writer user blocks are
  capped at 1 GiB.
- Four-byte files reject structured variable-length or reference custom fills
  because HDF5 1.14.6 cannot represent them safely across host/file layouts.
- Path normalization is lexical; combinations where `..` follows a soft-link
  component are not fixture-verified against every HDF5 path rule.

## Release history

Historical milestone details are available in the Git history. Version `0.4.1`
was published before the explicit conversion work; the current worktree is
GitHub-only until a later registry release.
