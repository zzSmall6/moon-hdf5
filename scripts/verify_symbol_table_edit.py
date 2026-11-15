#!/usr/bin/env python3
"""Verify symbol-table group edits using h5py and raw header messages."""

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


def verify(path: Path, dense: bool) -> None:
    data = path.read_bytes()
    assert data[8] == 0
    offset_size = data[13]
    length_size = data[14]
    assert offset_size == 4
    assert length_size == 8

    with h5py.File(path, "r") as handle:
        expected = [
            "alias_renamed",
            "external",
            "group",
            "group_soft",
            "matrix",
            "soft",
        ]
        if dense:
            expected += ["extra_0", "extra_1", "extra_2", "extra_3"]
        assert sorted(handle) == sorted(expected)
        assert handle.attrs["title"] == "old-style 32-bit offsets"
        np.testing.assert_array_equal(handle["alias_renamed"][:], np.arange(5))
        np.testing.assert_array_equal(handle["soft"][:], np.arange(5))
        assert h5py.h5o.get_info(
            handle["alias_renamed"].id
        ).addr == h5py.h5o.get_info(handle["group/values"].id).addr

        external = handle.get("external", getlink=True)
        assert isinstance(external, h5py.ExternalLink)
        assert external.filename == "external-target.h5"
        assert external.path == "/matrix"
        group_soft = handle.get("group_soft", getlink=True)
        assert isinstance(group_soft, h5py.SoftLink)
        assert group_soft.path == "group"
        soft = handle.get("soft", getlink=True)
        assert isinstance(soft, h5py.SoftLink)
        assert soft.path == "/group/values"

        group = handle["group"]
        assert group.attrs["answer"] == 42
        assert group.attrs["note"] == 7
        assert group["values"].attrs["unit"] == "count"
        np.testing.assert_array_equal(group["values"][:], np.arange(5))
        np.testing.assert_array_equal(group["values_alias"][:], np.arange(5))
        assert h5py.h5o.get_info(
            group["values_alias"].id
        ).addr == h5py.h5o.get_info(group["values"].id).addr
        assert group["self"].name == "/group/self"

        root_address = h5py.h5o.get_info(handle.id).addr
        group_address = h5py.h5o.get_info(group.id).addr

    root_messages = message_types(
        data, root_address, offset_size, length_size
    )
    group_messages = message_types(
        data, group_address, offset_size, length_size
    )
    assert 0x11 not in root_messages
    assert 0x02 in root_messages
    assert 0x11 not in group_messages
    assert 0x02 in group_messages
    if dense:
        assert 0x06 not in root_messages
    else:
        assert 0x06 in root_messages
    assert 0x06 in group_messages
    print(
        f"Verified {path} with h5py {h5py.__version__} "
        f"and HDF5 {h5py.version.hdf5_version}"
    )


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit(
            "usage: verify_symbol_table_edit.py <file.h5> [dense|compact]"
        )
    verify(Path(sys.argv[1]), len(sys.argv) == 3 and sys.argv[2] == "dense")
