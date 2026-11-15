#!/usr/bin/env python3
"""Verify groups and datasets created in an existing file with h5py."""

import shutil
import sys
import tempfile
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


def reference_addresses(handle, dataset):
    """Read on-disk reference slots without h5py's multi-reference conversion."""
    address_size = handle.id.get_create_plist().get_sizes()[0]
    count = int(np.prod(dataset.shape)) if dataset.shape else 1
    storage_size = dataset.id.get_storage_size()
    assert storage_size == count * address_size
    raw = np.zeros(storage_size, dtype=np.uint8)
    dataset.id.read(
        h5py.h5s.ALL,
        h5py.h5s.ALL,
        raw,
        mtype=h5py.h5t.STD_REF_OBJ,
    )
    return [
        int.from_bytes(raw[index * address_size : (index + 1) * address_size], "little")
        for index in range(count)
    ]


def reference_name(handle, reference):
    """Return the target path for a non-null HDF5 reference."""
    try:
        name = h5py.h5r.get_name(reference, handle.id)
    except RuntimeError:
        return None
    return name.decode() if name else None


def assert_first_nested_record(record, handle, offset_size):
    assert record[0] == b"moon"
    np.testing.assert_array_equal(record[1], np.array([1, -2], dtype=np.int32))
    assert [label.decode() for label in record[2]] == ["一", ""]
    assert reference_name(handle, record[3]) == "/matrix"
    assert reference_name(handle, record[4][0]) == "/matrix"
    # HDF5 1.14.6 cannot reliably convert the second slot of a legacy
    # four-byte reference array in a compound. The file test covers the raw
    # slot, while h5py can verify the eight-byte layout end to end.
    if offset_size == 8:
        assert reference_name(handle, record[4][1]) is None
    assert reference_name(handle, record[5]) == "/matrix"


def assert_second_nested_record(record, handle, offset_size):
    assert record[0] == b""
    assert record[1].size == 0
    assert [label.decode() for label in record[2]] == ["", "二"]
    assert reference_name(handle, record[3]) is None
    if offset_size == 8:
        pair = [reference_name(handle, item) for item in record[4]]
        assert pair == [None, "/matrix"]
    assert reference_name(handle, record[5]) is None


def verify(path: Path) -> None:
    data = path.read_bytes()
    version = data[8]
    if version in (0, 1):
        offset_size = data[13]
        length_size = data[14]
    elif version in (2, 3):
        offset_size = data[9]
        length_size = data[10]
    else:
        raise AssertionError(f"unsupported superblock version {version}")
    assert offset_size in (4, 8)
    assert length_size == 8

    with h5py.File(path, "r") as handle:
        assert isinstance(handle["created_root"], h5py.Group)
        assert isinstance(handle["group/created_leaf"], h5py.Group)
        assert sorted(handle["created_root"]) == [
            "chunked",
            "chunked_labels",
            "external",
            "growable",
            "labels",
            "nested_chunked",
            "nested_records",
            "nested_resizable",
            "numbers",
            "references",
            "resizable",
        ]
        assert sorted(handle["group/created_leaf"]) == ["fixed"]
        assert handle["created_root"].attrs["note"] == 11
        np.testing.assert_array_equal(
            handle["created_root/numbers"][:],
            np.arange(1, 7).reshape(2, 3),
        )
        assert handle["created_root/numbers"].fillvalue == -1
        assert handle["created_root/labels"].asstr()[:].tolist() == [
            "moon",
            "数据",
            "",
            "moon",
        ]
        chunked = handle["created_root/chunked"]
        assert chunked.shape == (4, 3)
        assert chunked.chunks == (2, 3)
        assert chunked.fillvalue == 99
        assert chunked.shuffle
        assert chunked.compression == "gzip"
        assert chunked.compression_opts == 6
        assert chunked.fletcher32
        np.testing.assert_array_equal(
            chunked[:],
            np.arange(12, dtype=np.uint32).reshape(4, 3),
        )
        chunked_labels = handle["created_root/chunked_labels"]
        assert chunked_labels.chunks == (2,)
        assert chunked_labels.compression == "gzip"
        assert chunked_labels.asstr()[:].tolist() == [
            "moon",
            "数据",
            "",
            "moon",
        ]
        resizable = handle["created_root/resizable"]
        assert resizable.shape == (4, 3)
        assert resizable.maxshape == (None, 3)
        assert resizable.chunks == (2, 3)
        assert resizable.fillvalue == -1
        assert resizable.shuffle
        assert resizable.compression == "gzip"
        np.testing.assert_array_equal(
            resizable[:], np.arange(12, dtype=np.int32).reshape(4, 3)
        )
        growable = handle["created_root/growable"]
        assert growable.shape == (3, 4)
        assert growable.maxshape == (None, None)
        assert growable.chunks == (2, 2)
        assert growable.fillvalue == 99
        assert growable.compression == "gzip"
        np.testing.assert_array_equal(
            growable[:], np.arange(1, 13, dtype=np.uint16).reshape(3, 4)
        )
        external = handle["created_root/external"]
        external_path = str(path) + ".external.bin"
        assert external.external == [(external_path, 4, 24)]
        assert external.fillvalue == -1
        np.testing.assert_array_equal(
            external[:], [10, 20, 30, 40, 50, 60]
        )
        second_target = (
            "group/values" if "group/values" in handle else "group/child"
        )
        references = reference_addresses(
            handle, handle["created_root/references"]
        )
        assert references == [
            h5py.h5o.get_info(handle["matrix"].id).addr,
            0,
            h5py.h5o.get_info(handle[second_target].id).addr,
        ]
        nested_records = handle["created_root/nested_records"]
        assert nested_records.shape == (2,)
        assert_first_nested_record(nested_records[0], handle, offset_size)
        assert_second_nested_record(nested_records[1], handle, offset_size)
        nested_chunked = handle["created_root/nested_chunked"]
        assert nested_chunked.shape == (3,)
        assert nested_chunked.chunks == (2,)
        assert nested_chunked.shuffle
        assert nested_chunked.compression == "gzip"
        assert_first_nested_record(nested_chunked[0], handle, offset_size)
        assert_first_nested_record(nested_chunked[2], handle, offset_size)
        assert_second_nested_record(nested_chunked[1], handle, offset_size)
        nested_resizable = handle["created_root/nested_resizable"]
        assert nested_resizable.shape == (1,)
        assert nested_resizable.maxshape == (None,)
        assert nested_resizable.chunks == (1,)
        assert not nested_resizable.shuffle
        assert nested_resizable.compression == "gzip"
        assert_first_nested_record(nested_resizable[0], handle, offset_size)
        if offset_size == 4:
            try:
                nested_resizable.fillvalue
            except RuntimeError:
                pass
            else:
                raise AssertionError(
                    "four-byte nested variable-length fill should be undefined"
                )
        assert handle["created_root/numbers"].attrs["unit"].tobytes() == b"count"
        np.testing.assert_array_equal(
            handle["group/created_leaf/fixed"][:], [100, 200]
        )
        if offset_size == 4:
            assert handle.attrs["title"] == "old-style 32-bit offsets"
            np.testing.assert_array_equal(handle["alias"][:], np.arange(5))
            np.testing.assert_array_equal(handle["group/values"][:], np.arange(5))
            group_soft = handle.get("soft", getlink=True)
            assert isinstance(group_soft, h5py.SoftLink)
            assert group_soft.path == "/group/values"
        if offset_size == 4:
            expected_matrix = np.arange(6).reshape(2, 3)
        else:
            expected_matrix = np.arange(30).reshape(6, 5)
        np.testing.assert_array_equal(handle["matrix"][:], expected_matrix)

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
    with tempfile.TemporaryDirectory() as directory:
        resized_path = Path(directory) / "resized.h5"
        shutil.copyfile(path, resized_path)
        with h5py.File(resized_path, "r+") as handle:
            resizable = handle["created_root/resizable"]
            resizable.resize((5, 3))
            resizable[4, :] = [12, 13, 14]
            assert resizable.shape == (5, 3)
            np.testing.assert_array_equal(
                resizable[4, :], [12, 13, 14]
            )
            growable = handle["created_root/growable"]
            growable.resize((5, 4))
            assert growable.shape == (5, 4)
            np.testing.assert_array_equal(growable[4, :], [99, 99, 99, 99])
            nested_resizable = handle["created_root/nested_resizable"]
            nested_resizable.resize((3,))
            assert nested_resizable.shape == (3,)
            assert_first_nested_record(nested_resizable[0], handle, offset_size)
            if offset_size == 8:
                assert_second_nested_record(nested_resizable[2], handle, offset_size)
    Path(str(path) + ".external.bin").unlink(missing_ok=True)
    print(
        f"Verified {path} with h5py {h5py.__version__} "
        f"and HDF5 {h5py.version.hdf5_version}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_object_create.py <file.h5>")
    verify(Path(sys.argv[1]))
