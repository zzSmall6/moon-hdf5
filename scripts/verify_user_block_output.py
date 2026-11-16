"""Verify user-block writer and patch output with HDF5 1.14.6."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import h5py
import numpy as np


def superblock_offsets(data: bytes) -> tuple[int, int, int, int]:
    signature = b"\x89HDF\r\n\x1a\n"
    assert data[:512] == bytes(512)
    assert data[512 : 520] == signature
    version = data[520]
    if version == 3:
        assert data[521] == 8 and data[522] == 8
        base_address, extension_address, eof, _root = struct.unpack_from(
            "<QQQQ", data, 524
        )
        assert extension_address == 0xFFFFFFFFFFFFFFFF
    elif version in (0, 1):
        width = data[525]
        assert width == 8 and data[526] == 8
        base_address = int.from_bytes(data[536:544], "little")
        eof = int.from_bytes(data[552:560], "little")
    else:
        raise AssertionError(f"unsupported superblock version {version}")
    assert base_address == 512
    return base_address, eof, 512, version


def verify_created(path: Path, edited: bool) -> None:
    data = path.read_bytes()
    base_address, eof, signature_offset, version = superblock_offsets(data)
    assert version == 3
    assert eof == len(data)

    with h5py.File(path, "r") as handle:
        assert handle.userblock_size == base_address
        assert handle["/group/values"].dtype == np.dtype("<i4")
        expected = np.array([10, 20, 30, 40] if edited else [1, 2, 3, 4])
        assert np.array_equal(handle["/group/values"][:], expected)
        expected_labels = (
            ["a longer value", "second"] if edited else ["one", "two"]
        )
        assert [value.decode("utf-8") for value in handle["/labels"][:]] == expected_labels
        assert handle["/group/values"].attrs["unit"].decode("utf-8") == "count"
    assert signature_offset == 512


def verify_existing_edited(path: Path) -> None:
    data = path.read_bytes()
    _, eof, _, version = superblock_offsets(data)
    assert version in (0, 1)
    assert eof == len(data)
    with h5py.File(path, "r") as handle:
        assert handle.userblock_size == 512
        assert "/created" in handle
        assert np.array_equal(handle["/x"][:], np.array([10, 20, 30, 40]))


def main(
    created_path: Path,
    edited_path: Path,
    existing_edited_path: Path,
) -> None:
    if h5py.version.hdf5_version != "1.14.6":
        raise RuntimeError(
            f"expected HDF5 1.14.6, found {h5py.version.hdf5_version}"
        )
    created = created_path.read_bytes()
    edited = edited_path.read_bytes()
    assert len(edited) > len(created)
    verify_created(created_path, edited=False)
    verify_created(edited_path, edited=True)
    verify_existing_edited(existing_edited_path)
    print("Verified user-block output with h5py", h5py.__version__)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: verify_user_block_output.py <new.h5> <edited.h5> <existing-edited.h5>"
        )
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
