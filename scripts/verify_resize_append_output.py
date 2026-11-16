#!/usr/bin/env python3
"""Verify existing-file dataset resize and append output with h5py."""

import sys
from pathlib import Path

import h5py
import numpy as np


def strings(values) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else value
        for value in values
    ]


def verify(path: Path) -> None:
    with h5py.File(path, "r") as handle:
        numbers = handle["numbers"]
        assert numbers.shape == (4,)
        assert numbers.maxshape == (None,)
        np.testing.assert_array_equal(numbers[:], [1, 2, 3, 4])

        filtered = handle["filtered"]
        expected_filtered = np.array([1, 2, 3, 4] + [-11] + [9] * 512, dtype=np.int32)
        assert filtered.shape == (517,)
        np.testing.assert_array_equal(filtered[:], expected_filtered)

        matrix = handle["matrix"]
        assert matrix.shape == (3, 3)
        expected_matrix = np.full((3, 3), -1, dtype=np.int32)
        expected_matrix[0, 0] = 1
        expected_matrix[0, 1] = 2
        expected_matrix[1, 0] = 3
        expected_matrix[1, 1] = 4
        expected_matrix[2, 2] = 5
        np.testing.assert_array_equal(matrix[:], expected_matrix)

        expected_names = ["moon"] + [f"value-{index}" for index in range(200)]
        assert strings(handle["names"][:]) == expected_names
        assert strings(handle["filtered_names"][:]) == [
            "moon",
            "数据",
            "",
            "tail",
        ]

        np.testing.assert_array_equal(handle["resized"][:], [10, 20, -1, -1, -1])
        np.testing.assert_array_equal(handle["hidden"][:], [1, 2, 3, 4])

    print(
        f"Verified {path} with h5py {h5py.__version__} "
        f"and HDF5 {h5py.version.hdf5_version}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_resize_append_output.py <file.h5>")
    verify(Path(sys.argv[1]))
