"""Generate HDF5 1.14.6 fixtures for 2/4/8-byte address and length widths."""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TESTDATA = ROOT / "testdata"
SIZES = (2, 4, 8)


def fixture_path(offset_size: int, length_size: int) -> Path:
    return TESTDATA / f"size_{offset_size}{length_size}.h5"


def create_fixture(offset_size: int, length_size: int) -> Path:
    path = fixture_path(offset_size, length_size)
    if path.exists():
        path.unlink()
    fcpl = h5py.h5p.create(h5py.h5p.FILE_CREATE)
    fcpl.set_sizes(offset_size, length_size)
    fapl = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    fapl.set_libver_bounds(h5py.h5f.LIBVER_LATEST, h5py.h5f.LIBVER_LATEST)
    fid = h5py.h5f.create(
        bytes(path), h5py.h5f.ACC_TRUNC, fcpl=fcpl, fapl=fapl
    )
    with h5py.File(fid) as handle:
        group = handle.create_group("g")
        data = np.arange(12, dtype="<i4").reshape(3, 4)
        contiguous = group.create_dataset("data", data=data, track_times=False)
        group.create_dataset(
            "chunked",
            data=data,
            chunks=(2, 2),
            compression="gzip",
            track_times=False,
        )
        contiguous.attrs["answer"] = np.int32(42)
        group.attrs["label"] = np.array("matrix", dtype="S8")
    return path


def verify_fixture(offset_size: int, length_size: int) -> None:
    path = fixture_path(offset_size, length_size)
    raw = path.read_bytes()
    if raw[:8] != b"\x89HDF\r\n\x1a\n":
        raise AssertionError(f"{path} does not have an HDF5 signature")
    if raw[8] != 3 or raw[9] != offset_size or raw[10] != length_size:
        raise AssertionError(f"{path} has unexpected superblock sizes")
    expected = np.arange(12, dtype="<i4").reshape(3, 4)
    with h5py.File(path, "r") as handle:
        np.testing.assert_array_equal(handle["g/data"][...], expected)
        np.testing.assert_array_equal(handle["g/chunked"][...], expected)
        assert handle["g/data"].attrs["answer"] == 42
        assert handle["g"].attrs["label"] == b"matrix"


def generate() -> None:
    if h5py.version.hdf5_version != "1.14.6":
        raise SystemExit(
            "size matrix fixtures require HDF5 1.14.6, found "
            + h5py.version.hdf5_version
        )
    TESTDATA.mkdir(exist_ok=True)
    for offset_size in SIZES:
        for length_size in SIZES:
            path = create_fixture(offset_size, length_size)
            verify_fixture(offset_size, length_size)
            print(
                f"Generated {path.name} with {offset_size}-byte addresses and "
                f"{length_size}-byte lengths"
            )


if __name__ == "__main__":
    os.chdir(ROOT)
    generate()
