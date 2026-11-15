"""Verify a MoonBit rewrite of a deflate- and Fletcher32-filtered link heap."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np


def main(path: Path) -> None:
    data = path.read_bytes()
    filtered_headers = []
    offset = 0
    while True:
        offset = data.find(b"FRHP", offset)
        if offset < 0:
            break
        filtered_headers.append(int.from_bytes(data[offset + 7 : offset + 9], "little"))
        offset += 4
    if not any(size > 0 for size in filtered_headers):
        raise RuntimeError("rewritten file has no filtered fractal heap")

    with h5py.File(path, "r") as handle:
        group = handle["filtered"]
        names = list(group.keys())
        if len(names) != 1201:
            raise RuntimeError(f"filtered group has {len(names)} links")
        if "renamed_item" not in group:
            raise RuntimeError("renamed link is missing")
        if "item_0000" in group:
            raise RuntimeError("old link name is still present")
        np.testing.assert_array_equal(group["renamed_item"][:], [0])
        long_name = "long_" + "x" * 5000
        if long_name not in group:
            raise RuntimeError("huge filtered link is missing")
        np.testing.assert_array_equal(group[long_name][:], [7])
    print(f"Verified filtered link editing with h5py {h5py.__version__}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
