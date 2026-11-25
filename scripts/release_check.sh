#!/bin/sh

set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"

echo "==> Update interfaces"
moon info

echo "==> Check formatting"
moon fmt --check

echo "==> Check"
moon check

echo "==> Test (wasm)"
moon test --target wasm --no-parallelize

echo "==> Test (native)"
moon test --target native --no-parallelize

echo "==> Documentation"
moon doc

echo "==> Package contents"
moon package --list >/dev/null 2>&1 || {
  moon package --list
  exit 1
}

echo "==> Writer compatibility"
output=$(mktemp "${TMPDIR:-/tmp}/moon-hdf5-release.XXXXXX")
user_block_new=$(mktemp "${TMPDIR:-/tmp}/moon-hdf5-user-block-new.XXXXXX")
user_block_edited=$(mktemp "${TMPDIR:-/tmp}/moon-hdf5-user-block-edited.XXXXXX")
user_block_existing=$(mktemp "${TMPDIR:-/tmp}/moon-hdf5-user-block-existing.XXXXXX")
dense_links=$(mktemp "${TMPDIR:-/tmp}/moon-hdf5-dense-links.XXXXXX")
dense_links_edited=$(mktemp "${TMPDIR:-/tmp}/moon-hdf5-dense-links-edited.XXXXXX")
symbol_compact=$(mktemp "${TMPDIR:-/tmp}/moon-hdf5-symbol-compact.XXXXXX")
symbol_dense=$(mktemp "${TMPDIR:-/tmp}/moon-hdf5-symbol-dense.XXXXXX")
filtered_links=$(mktemp "${TMPDIR:-/tmp}/moon-hdf5-filtered-links.XXXXXX")
extended_floats=$(mktemp "${TMPDIR:-/tmp}/moon-hdf5-extended-floats.XXXXXX")
external_references=$(mktemp -d "${TMPDIR:-/tmp}/moon-hdf5-external-references.XXXXXX")
virtual_new=$(mktemp "${TMPDIR:-/tmp}/moon-hdf5-virtual-new.XXXXXX")
virtual_edited=$(mktemp "${TMPDIR:-/tmp}/moon-hdf5-virtual-edited.XXXXXX")
created=$(mktemp "${TMPDIR:-/tmp}/moon-hdf5-created.XXXXXX")
created8=$(mktemp "${TMPDIR:-/tmp}/moon-hdf5-created8.XXXXXX")
deleted=$(mktemp "${TMPDIR:-/tmp}/moon-hdf5-deleted.XXXXXX")
resize_append=$(mktemp "${TMPDIR:-/tmp}/moon-hdf5-resize-append.XXXXXX")
conversion=$(mktemp "${TMPDIR:-/tmp}/moon-hdf5-conversion.XXXXXX")
trap 'rm -f "$output" "$user_block_new" "$user_block_edited" "$user_block_existing" "$dense_links" "$dense_links_edited" "$symbol_compact" "$symbol_dense" "$filtered_links" "$extended_floats" "$virtual_new" "$virtual_edited" "$created" "$created8" "$created.external.bin" "$created8.external.bin" "$created.external-refs.bin" "$created8.external-refs.bin" "$deleted" "$resize_append" "$conversion"; rm -rf "$external_references"' EXIT HUP INT TERM
moon run src/cmd/write_demo --target native -- "$output"

python=${H5PY_PYTHON:-}
if [ "${SKIP_H5PY:-0}" = "1" ]; then
  echo "==> h5py verification skipped (SKIP_H5PY=1)"
else
  if [ -z "$python" ]; then
    for candidate in python3 python; do
      if command -v "$candidate" >/dev/null 2>&1 &&
        "$candidate" -c "import h5py, numpy" >/dev/null 2>&1; then
        python=$candidate
        break
      fi
    done
  fi
  if [ -z "$python" ]; then
    echo "h5py and NumPy are required; set H5PY_PYTHON or SKIP_H5PY=1" >&2
    exit 1
  fi
  "$python" scripts/verify_writer_output.py "$output"
fi

echo "==> Datatype conversion interoperability"
moon run src/cmd/conversion_demo --target native -- \
  "$conversion" testdata/datatypes_v114.h5 testdata/numeric_layouts.h5 \
  testdata/wide_numeric.h5
if [ "${SKIP_H5PY:-0}" = "1" ]; then
  echo "==> Conversion verification skipped (SKIP_H5PY=1)"
else
  "$python" scripts/verify_conversion_interop.py "$conversion" \
    testdata/datatypes_v114.h5 testdata/numeric_layouts.h5 \
    testdata/wide_numeric.h5
fi

echo "==> User-block output"
moon run src/cmd/user_block_demo --target native -- \
  testdata/userblock.h5 "$user_block_new" "$user_block_edited" \
  "$user_block_existing"
if [ "${SKIP_H5PY:-0}" = "1" ]; then
  echo "==> User-block verification skipped (SKIP_H5PY=1)"
else
  "$python" scripts/verify_user_block_output.py \
    "$user_block_new" "$user_block_edited" "$user_block_existing"
fi

echo "==> Dense link output"
moon run src/cmd/dense_link_demo --target native -- \
  "$dense_links" "$dense_links_edited"
if [ "${SKIP_H5PY:-0}" = "1" ]; then
  echo "==> Dense-link verification skipped (SKIP_H5PY=1)"
else
  "$python" scripts/verify_dense_link_output.py \
    "$dense_links" "$dense_links_edited"
fi

echo "==> Virtual dataset output"
moon run src/cmd/virtual_dataset_demo --target native -- \
  "$virtual_new" "$virtual_edited"
if [ "${SKIP_H5PY:-0}" = "1" ]; then
  echo "==> Virtual-dataset writer verification skipped (SKIP_H5PY=1)"
else
  "$python" scripts/verify_virtual_dataset_write.py \
    "$virtual_new" "$virtual_edited"
fi

echo "==> Symbol-table editing"
moon run src/cmd/symbol_table_edit_demo --target native -- \
  testdata/symbol_table_32.h5 "$symbol_compact" compact
moon run src/cmd/symbol_table_edit_demo --target native -- \
  testdata/symbol_table_32.h5 "$symbol_dense" dense
if [ "${SKIP_H5PY:-0}" = "1" ]; then
  echo "==> Symbol-table edit verification skipped (SKIP_H5PY=1)"
else
  "$python" scripts/verify_symbol_table_edit.py "$symbol_compact" compact
  "$python" scripts/verify_symbol_table_edit.py "$symbol_dense" dense
fi

echo "==> Filtered link editing"
moon run src/cmd/filtered_link_edit_demo --target native -- \
  testdata/fractal_heap_advanced.h5 "$filtered_links"
if [ "${SKIP_H5PY:-0}" = "1" ]; then
  echo "==> Filtered link edit verification skipped (SKIP_H5PY=1)"
else
  "$python" scripts/verify_filtered_link_edit.py "$filtered_links"
fi

echo "==> Extended floating-point output"
moon run src/cmd/extended_float_demo --target native -- "$extended_floats"
if [ "${SKIP_H5PY:-0}" = "1" ]; then
  echo "==> Extended floating-point verification skipped (SKIP_H5PY=1)"
else
  "$python" scripts/verify_extended_float_output.py "$extended_floats"
fi

echo "==> External version-4 reference output"
moon run src/cmd/external_reference_demo --target native -- \
  "$external_references/external_reference_target.h5" \
  "$external_references/source.h5" \
  "$external_references/patched.h5"
if [ "${SKIP_H5PY:-0}" = "1" ]; then
  echo "==> External version-4 reference verification skipped (SKIP_H5PY=1)"
else
  "$python" scripts/verify_external_reference_output.py \
    "$external_references/external_reference_target.h5" \
    "$external_references/source.h5" \
    "$external_references/patched.h5"
fi

echo "==> Object creation"
moon run src/cmd/object_create_demo --target native -- \
  testdata/symbol_table_32.h5 "$created"
moon run src/cmd/object_create_demo --target native -- \
  testdata/references.h5 "$created8"
if [ "${SKIP_H5PY:-0}" = "1" ]; then
  echo "==> Object creation verification skipped (SKIP_H5PY=1)"
else
  "$python" scripts/verify_object_create.py "$created"
  "$python" scripts/verify_object_create.py "$created8"
fi

echo "==> Object deletion"
moon run src/cmd/object_delete_demo --target native -- \
  testdata/symbol_table_32.h5 "$deleted"
if [ "${SKIP_H5PY:-0}" = "1" ]; then
  echo "==> Object deletion verification skipped (SKIP_H5PY=1)"
else
  "$python" scripts/verify_object_delete.py "$deleted"
fi

echo "==> Existing-file resize and append"
moon run src/cmd/resize_append_demo --target native -- "$resize_append"
if [ "${SKIP_H5PY:-0}" = "1" ]; then
  echo "==> Resize/append verification skipped (SKIP_H5PY=1)"
else
  "$python" scripts/verify_resize_append_output.py "$resize_append"
fi

if [ "${SKIP_H5PY:-0}" = "1" ]; then
  echo "==> Numeric layout fixture verification skipped (SKIP_H5PY=1)"
elif [ ! -f testdata/numeric_layouts.h5 ]; then
  echo "==> Numeric layout fixture verification skipped (fixture unavailable)"
else
  "$python" scripts/verify_numeric_layouts_fixture.py testdata/numeric_layouts.h5
fi

if [ "${SKIP_H5PY:-0}" = "1" ]; then
  echo "==> Scale-offset fixture verification skipped (SKIP_H5PY=1)"
elif [ ! -f testdata/scaleoffset.h5 ]; then
  echo "==> Scale-offset fixture verification skipped (fixture unavailable)"
else
  "$python" scripts/verify_scaleoffset_fixture.py testdata/scaleoffset.h5
fi

if [ "${SKIP_H5PY:-0}" = "1" ]; then
  echo "==> N-bit fixture verification skipped (SKIP_H5PY=1)"
elif [ ! -f testdata/nbit_filters.h5 ]; then
  echo "==> N-bit fixture verification skipped (fixture unavailable)"
else
  "$python" scripts/verify_nbit_fixture.py testdata/nbit_filters.h5
fi

if [ "${SKIP_H5PY:-0}" = "1" ]; then
  echo "==> Variable-length fill verification skipped (SKIP_H5PY=1)"
elif [ ! -f testdata/vlen_fills.h5 ]; then
  echo "==> Variable-length fill verification skipped (fixture unavailable)"
else
  "$python" scripts/verify_vlen_fill_fixture.py testdata/vlen_fills.h5
fi

if [ "${SKIP_H5PY:-0}" = "1" ]; then
  echo "==> NetCDF-4 fixture verification skipped (SKIP_H5PY=1)"
elif [ ! -f testdata/netcdf4_1_7_5.h5 ]; then
  echo "==> NetCDF-4 fixture verification skipped (fixture unavailable)"
else
  "$python" scripts/verify_netcdf4_fixture.py testdata/netcdf4_1_7_5.h5
fi

if [ "${SKIP_H5PY:-0}" = "1" ]; then
  echo "==> External raw-data fixture verification skipped (SKIP_H5PY=1)"
elif [ ! -f testdata/external_storage.h5 ]; then
  echo "==> External raw-data fixture verification skipped (fixture unavailable)"
else
  "$python" scripts/verify_external_storage_fixture.py testdata/external_storage.h5
fi

if [ "${SKIP_H5PY:-0}" = "1" ]; then
  echo "==> External-reference fixture verification skipped (SKIP_H5PY=1)"
elif [ ! -f testdata/external_references.h5 ] ||
  [ ! -f testdata/external_reference_target.h5 ]; then
  echo "==> External-reference fixture verification skipped (fixture unavailable)"
else
  echo "==> External-reference fixture verification"
  "$python" scripts/verify_external_reference_fixture.py \
    testdata/external_references.h5
fi

if [ "${SKIP_H5PY:-0}" = "1" ]; then
  echo "==> Shared-message fixture verification skipped (SKIP_H5PY=1)"
elif [ ! -f testdata/shared_messages.h5 ]; then
  echo "==> Shared-message fixture verification skipped (fixture unavailable)"
else
  "$python" scripts/verify_shared_messages_fixture.py testdata/shared_messages.h5
fi

if [ "${SKIP_H5PY:-0}" = "1" ]; then
  echo "==> Virtual-dataset fixture verification skipped (SKIP_H5PY=1)"
elif [ ! -f testdata/virtual_dataset.h5 ] ||
  [ ! -f testdata/virtual_dataset_32.h5 ] ||
  [ ! -f testdata/virtual_dataset_nested.h5 ] ||
  [ ! -f testdata/virtual_dataset_printf.h5 ] ||
  [ ! -f testdata/virtual_dataset_conversion.h5 ] ||
  [ ! -f testdata/virtual_dataset_source.h5 ] ||
  [ ! -f testdata/virtual_dataset_source_32.h5 ] ||
  [ ! -f testdata/virtual_printf_source_0.h5 ] ||
  [ ! -f testdata/virtual_printf_source_1.h5 ]; then
  echo "==> Virtual-dataset fixture verification skipped (fixture unavailable)"
else
  "$python" scripts/verify_virtual_dataset_fixture.py
fi

if [ "${SKIP_H5PY:-0}" = "1" ]; then
  echo "==> Address/length width matrix verification skipped (SKIP_H5PY=1)"
elif ! ls testdata/size_22.h5 testdata/size_24.h5 testdata/size_28.h5 \
  testdata/size_42.h5 testdata/size_44.h5 testdata/size_48.h5 \
  testdata/size_82.h5 testdata/size_84.h5 testdata/size_88.h5 \
  >/dev/null 2>&1; then
  echo "==> Address/length width matrix verification skipped (fixture unavailable)"
else
  "$python" scripts/verify_size_matrix_fixtures.py
fi

if [ "${SKIP_H5PY:-0}" = "1" ]; then
  echo "==> Hard-link fixture verification skipped (SKIP_H5PY=1)"
elif [ ! -f testdata/hard_links.h5 ]; then
  echo "==> Hard-link fixture verification skipped (fixture unavailable)"
else
  "$python" scripts/verify_hard_links_fixture.py testdata/hard_links.h5
fi

if [ "${SKIP_H5PY:-0}" = "1" ]; then
  echo "==> MATLAB-compatible fixture verification skipped (SKIP_H5PY=1)"
elif [ ! -f testdata/matlab73_compatible.mat ]; then
  echo "==> MATLAB-compatible fixture verification skipped (fixture unavailable)"
else
  "$python" scripts/verify_matlab73_fixture.py \
    testdata/matlab73_compatible.mat
fi

if [ "${SKIP_H5PY:-0}" = "1" ]; then
  echo "==> Reference fixture verification skipped (SKIP_H5PY=1)"
elif [ ! -f testdata/references.h5 ] ||
  [ ! -f testdata/references_v108.h5 ] ||
  [ ! -f testdata/references_layouts_32.h5 ]; then
  echo "==> Reference fixture verification skipped (fixture unavailable)"
else
  "$python" scripts/verify_reference_fixture.py testdata/references.h5
  "$python" scripts/verify_reference_fixture.py testdata/references_v108.h5
  "$python" scripts/verify_region_layout_fixture.py \
    testdata/references_layouts_32.h5
fi

if [ "${SKIP_H5PY:-0}" = "1" ]; then
  echo "==> Wide numeric fixture verification skipped (SKIP_H5PY=1)"
elif [ ! -f testdata/wide_numeric.h5 ]; then
  echo "==> Wide numeric fixture verification skipped (fixture unavailable)"
else
  "$python" scripts/verify_wide_numeric_fixture.py testdata/wide_numeric.h5
fi

if [ "${SKIP_H5PY:-0}" != "1" ]; then
  echo "==> Persistent free-space inspection"
  "$python" scripts/verify_free_space.py
  "$python" scripts/verify_space_reuse.py
fi

echo "Release checks passed."
