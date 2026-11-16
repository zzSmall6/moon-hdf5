"""Verify dense link output beyond the 16-bit boundary with HDF5 1.14.6."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np


def verify_group(group: h5py.Group, expected_count: int) -> None:
    assert len(group) == expected_count
    assert group.id.get_create_plist().get_link_creation_order() == 3
    samples = ["link_0", f"link_{expected_count // 2}"]
    samples.append(f"link_{expected_count - 1}")
    for name in samples:
        link = group.get(name, getlink=True)
        assert isinstance(link, h5py.SoftLink)
        assert link.path == "/target"
    assert group[f"link_{expected_count - 1}"][()] == np.uint64(42)


def main(created_path: Path, edited_path: Path) -> None:
    if h5py.version.hdf5_version != "1.14.6":
        raise RuntimeError(
            f"expected HDF5 1.14.6, found {h5py.version.hdf5_version}"
        )
    with h5py.File(created_path, "r") as handle:
        verify_group(handle["/links"], 65536)
    with h5py.File(edited_path, "r") as handle:
        verify_group(handle["/links"], 65537)
        link = handle["/links"].get("link_65536", getlink=True)
        assert isinstance(link, h5py.SoftLink)
        assert link.path == "/target"
    print("Verified dense link output with h5py", h5py.__version__)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: verify_dense_link_output.py <created.h5> <edited.h5>"
        )
    main(Path(sys.argv[1]), Path(sys.argv[2]))
