"""Generate dense attributes stored in a deflate-filtered fractal heap."""

from __future__ import annotations

import ctypes
from pathlib import Path

import h5py
import numpy as np


def load_hdf5() -> ctypes.CDLL:
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split()
        if len(fields) >= 6 and "libhdf5" in fields[-1] and ".so" in fields[-1]:
            library = ctypes.CDLL(fields[-1])
            library.H5Pset_filter.argtypes = [
                ctypes.c_int64,
                ctypes.c_int,
                ctypes.c_uint,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_uint),
            ]
            library.H5Pset_filter.restype = ctypes.c_int
            return library
    raise RuntimeError("unable to locate the bundled HDF5 library")


def generate(destination: Path) -> None:
    library = load_hdf5()
    with h5py.File(destination, "w", libver="v110") as handle:
        properties = h5py.h5p.create(h5py.h5p.GROUP_CREATE)
        parameters = (ctypes.c_uint * 1)(6)
        if library.H5Pset_filter(
            properties.id, 1, 0, len(parameters), parameters
        ) < 0:
            raise RuntimeError("unable to install the deflate filter")
        group = h5py.Group(
            h5py.h5g.create(handle.id, b"filtered", gcpl=properties)
        )
        for index in range(24):
            group.attrs.create(
                f"value_{index:02d}",
                np.arange(16, dtype="<i4") + index,
            )

    with h5py.File(destination, "r") as handle:
        attributes = handle["filtered"].attrs
        if len(attributes) != 24:
            raise RuntimeError("filtered attribute fixture is incomplete")
        for index in range(24):
            np.testing.assert_array_equal(
                attributes[f"value_{index:02d}"],
                np.arange(16, dtype="<i4") + index,
            )

    data = destination.read_bytes()
    if b"FRHP" not in data or b"deflate" not in data:
        raise RuntimeError("fixture does not contain a filtered fractal heap")
    print(f"Verified {destination} with h5py {h5py.__version__}")


if __name__ == "__main__":
    directory = Path(__file__).resolve().parents[1] / "testdata"
    generate(directory / "filtered_attributes.h5")
