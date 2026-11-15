"""Verify MoonBit x87 extended floating-point output with HDF5 1.14.6."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import h5py
import numpy as np


def main(path: Path) -> None:
    if h5py.version.hdf5_version != "1.14.6":
        raise RuntimeError(
            f"expected HDF5 1.14.6, found {h5py.version.hdf5_version}"
        )
    minimum = np.longdouble(2.0) ** -1074
    with h5py.File(path, "r") as handle:
        for size in (10, 16):
            for suffix in ("le", "be"):
                name = f"x87_{size}_{suffix}"
                dataset = handle[name]
                datatype = dataset.id.get_type()
                assert datatype.get_class() == h5py.h5t.FLOAT
                assert datatype.get_size() == size
                assert datatype.get_precision() == 80
                assert datatype.get_fields() == (79, 64, 15, 0, 63)
                values = dataset[:]
                assert values[0] == np.longdouble(1.5)
                assert values[1] == np.longdouble(-2.25)
                assert values[2] == minimum
                assert math.isinf(float(values[3])) and values[3] > 0
                assert math.isinf(float(values[4])) and values[4] < 0
                assert math.isnan(float(values[5]))
                assert values[6] == 0 and np.signbit(values[6])
    print(f"Verified extended floating-point output with h5py {h5py.__version__}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: verify_extended_float_output.py <file.h5>"
        )
    main(Path(sys.argv[1]))
