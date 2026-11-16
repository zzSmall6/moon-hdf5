"""Verify committed virtual-dataset fixtures with HDF5 1.14.6."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TESTDATA = ROOT / "testdata"


def verify() -> None:
    if h5py.version.hdf5_version != "1.14.6":
        raise SystemExit(
            "virtual dataset fixtures require HDF5 1.14.6, found "
            + h5py.version.hdf5_version
        )

    with h5py.File(TESTDATA / "virtual_dataset_source.h5", "r") as source:
        np.testing.assert_array_equal(
            source["data"][...], np.arange(20, dtype="<i4").reshape(4, 5)
        )
        np.testing.assert_array_equal(
            source["matrix"][...], np.arange(100, 136, dtype="<i4").reshape(6, 6)
        )
        np.testing.assert_array_equal(
            source["vector"][...], np.arange(6, dtype="<i4")
        )

    expected = np.full((8, 10), -1, dtype="<i4")
    values = np.arange(20, dtype="<i4").reshape(4, 5)
    expected[0:4, 0:5] = values
    expected[4:8, 5:10] = values
    with h5py.File(TESTDATA / "virtual_dataset.h5", "r") as handle:
        np.testing.assert_array_equal(handle["mapped"][...], expected)
        np.testing.assert_array_equal(
            handle["same_file"][...], np.arange(12, dtype="<i4").reshape(3, 4)
        )
        np.testing.assert_array_equal(
            handle["strided"][...],
            np.arange(100, 136, dtype="<i4").reshape(6, 6)[::2, ::2],
        )
        np.testing.assert_array_equal(
            handle["multiple_mappings"][...], np.arange(6, dtype="<i4")
        )
        for name in ("mapped", "same_file", "strided", "multiple_mappings"):
            assert handle[name].id.get_create_plist().get_layout() == 3

    expected_32 = np.full((4, 6), -9, dtype="<i4")
    expected_32[0:3, 0:4] = np.arange(200, 212, dtype="<i4").reshape(3, 4)
    with h5py.File(TESTDATA / "virtual_dataset_32.h5", "r") as handle:
        np.testing.assert_array_equal(handle["mapped"][...], expected_32)
        assert handle["mapped"].id.get_create_plist().get_layout() == 3

    with h5py.File(TESTDATA / "virtual_dataset_nested.h5", "r") as handle:
        np.testing.assert_array_equal(
            handle["nested"][...], np.arange(20, dtype="<i4").reshape(4, 5)
        )
        assert handle["nested"].id.get_create_plist().get_layout() == 3

    expected_printf = np.array(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [100, 101, 102, 103],
            [104, 105, 106, 107],
        ],
        dtype="<i8",
    )
    with h5py.File(TESTDATA / "virtual_dataset_printf.h5", "r") as handle:
        np.testing.assert_array_equal(handle["printf_grid"][...], expected_printf)
        np.testing.assert_array_equal(
            handle["literal_percent"][...], np.arange(4, dtype="<i2")
        )
        np.testing.assert_array_equal(
            handle["printf_dataset"][...], np.array([10, 11, 20, 21], dtype="<i4")
        )
        assert handle["printf_grid"].maxshape == (None, 4)
        assert handle["printf_dataset"].maxshape == (None,)
        for name in ("printf_grid", "literal_percent", "printf_dataset"):
            assert handle[name].is_virtual

    with h5py.File(TESTDATA / "virtual_dataset_conversion.h5", "r") as handle:
        np.testing.assert_array_equal(
            handle["i16_i64"][...],
            np.array([[1, -2, 3], [4, 5, -6]], dtype="<i8"),
        )
        np.testing.assert_array_equal(
            handle["i16_f64"][...],
            np.array([[1, -2, 3], [4, 5, -6]], dtype="<f8"),
        )
        np.testing.assert_array_equal(
            handle["f32_i32"][...],
            np.array([[1, -2, 3], [-4, 5, -6]], dtype="<i4"),
        )
        np.testing.assert_array_equal(
            handle["s3_s8"][...], np.array([b"ab", b"c", b"def"], dtype="S8")
        )
        for name in ("i16_i64", "i16_f64", "f32_i32", "s3_s8"):
            assert handle[name].is_virtual

    print(
        "Verified virtual-dataset fixtures with h5py "
        f"{h5py.__version__}/HDF5 {h5py.version.hdf5_version}"
    )


if __name__ == "__main__":
    verify()
