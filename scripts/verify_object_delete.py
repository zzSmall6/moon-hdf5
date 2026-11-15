#!/usr/bin/env python3
"""Verify hard-link object deletion with h5py."""

import sys
from pathlib import Path

import h5py
import numpy as np


def verify(path: Path) -> None:
    with h5py.File(path, "r") as handle:
        assert "matrix" not in handle
        assert "matrix_alias" not in handle
        assert isinstance(handle["group"], h5py.Group)
        np.testing.assert_array_equal(handle["group/values"][:], np.arange(5))
        soft = handle.get("soft", getlink=True)
        assert isinstance(soft, h5py.SoftLink)
        assert soft.path == "/group/values"
        assert handle.attrs["title"] == "old-style 32-bit offsets"
    print(f"Verified {path} with h5py {h5py.__version__} and HDF5 {h5py.version.hdf5_version}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_object_delete.py <file.h5>")
    verify(Path(sys.argv[1]))
