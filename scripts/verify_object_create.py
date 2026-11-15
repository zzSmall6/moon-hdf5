#!/usr/bin/env python3
"""Verify groups and datasets created in an existing file with h5py."""

import sys
from pathlib import Path

import h5py
import numpy as np

from hdf5_raw import object_header_messages


def message_types(data, address, offset_size, length_size):
    return [
        message_type
        for message_type, _, _ in object_header_messages(
            data, address, offset_size, length_size
        )
    ]


def verify(path: Path) -> None:
    data = path.read_bytes()
    offset_size = data[13]
    length_size = data[14]
    assert offset_size == 4
    assert length_size == 8

    with h5py.File(path, "r") as handle:
        assert isinstance(handle["created_root"], h5py.Group)
        assert isinstance(handle["group/created_leaf"], h5py.Group)
        assert sorted(handle["created_root"]) == ["numbers"]
        assert sorted(handle["group/created_leaf"]) == ["fixed"]
        assert handle["created_root"].attrs["note"] == 11
        np.testing.assert_array_equal(
            handle["created_root/numbers"][:],
            np.arange(1, 7).reshape(2, 3),
        )
        assert handle["created_root/numbers"].attrs["unit"].tobytes() == b"count"
        np.testing.assert_array_equal(
            handle["group/created_leaf/fixed"][:], [100, 200]
        )
        assert handle.attrs["title"] == "old-style 32-bit offsets"
        np.testing.assert_array_equal(handle["alias"][:], np.arange(5))
        np.testing.assert_array_equal(
            handle["matrix"][:], np.arange(6).reshape(2, 3)
        )
        np.testing.assert_array_equal(handle["group/values"][:], np.arange(5))
        group_soft = handle.get("soft", getlink=True)
        assert isinstance(group_soft, h5py.SoftLink)
        assert group_soft.path == "/group/values"

        root_address = h5py.h5o.get_info(handle.id).addr
        group_address = h5py.h5o.get_info(handle["group"].id).addr
        created_address = h5py.h5o.get_info(handle["created_root"].id).addr
        leaf_address = h5py.h5o.get_info(handle["group/created_leaf"].id).addr
        dataset_address = h5py.h5o.get_info(
            handle["created_root/numbers"].id
        ).addr
        leaf_dataset_address = h5py.h5o.get_info(
            handle["group/created_leaf/fixed"].id
        ).addr

    for address in (root_address, group_address):
        messages = message_types(data, address, offset_size, length_size)
        assert 0x11 not in messages
        assert 0x02 in messages
    for address in (created_address, leaf_address):
        messages = message_types(data, address, offset_size, length_size)
        assert 0x02 in messages
        assert 0x11 not in messages
    for address in (dataset_address, leaf_dataset_address):
        messages = message_types(data, address, offset_size, length_size)
        assert set(messages) >= {0x01, 0x03, 0x08}
    print(
        f"Verified {path} with h5py {h5py.__version__} "
        f"and HDF5 {h5py.version.hdf5_version}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_object_create.py <file.h5>")
    verify(Path(sys.argv[1]))
