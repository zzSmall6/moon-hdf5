"""Verify MoonBit datatype conversion output with HDF5 1.14.6 and h5py."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np


def _raw_values(dataset: h5py.Dataset) -> tuple[h5py.h5t.TypeID, list[bytes]]:
    datatype = dataset.id.get_type()
    space = dataset.id.get_space()
    count = space.get_simple_extent_npoints()
    payload = np.zeros((count, datatype.get_size()), dtype=np.uint8)
    dataset.id.read(h5py.h5s.ALL, h5py.h5s.ALL, payload, mtype=datatype)
    space.close()
    return datatype, [bytes(row) for row in payload]


def _decode_integer(raw: bytes, datatype: h5py.h5t.TypeID, signed: bool) -> int:
    precision = datatype.get_precision()
    offset = datatype.get_offset()
    order = datatype.get_order()
    value = int.from_bytes(
        raw, "little" if order == h5py.h5t.ORDER_LE else "big"
    )
    value = (value >> offset) & ((1 << precision) - 1)
    if signed and value & (1 << (precision - 1)):
        value -= 1 << precision
    return value


def _assert_compound_value(actual: np.void, expected: dict[str, object]) -> None:
    for name, value in expected.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(actual[name], value)
        elif isinstance(value, float):
            assert actual[name] == value, (name, actual[name], value)
        else:
            assert actual[name] == value, (name, actual[name], value)


def verify(path: Path) -> None:
    if h5py.version.hdf5_version != "1.14.6":
        raise SystemExit(
            "conversion verification requires HDF5 1.14.6, found "
            + h5py.version.hdf5_version
        )

    with h5py.File(path, "r") as handle:
        assert set(handle) == {"generated", "imported"}
        generated = handle["generated"]
        imported = handle["imported"]

        narrowed = generated["narrowed"]
        assert narrowed.dtype == np.dtype("u1")
        np.testing.assert_array_equal(narrowed[:], [4, 5, 6])
        assert float(narrowed.attrs["scale"]) == 4.0

        lossy = generated["lossy"]
        assert lossy.dtype == np.dtype("u1")
        np.testing.assert_array_equal(lossy[:], [44, 255])

        float32 = generated["float32"]
        assert float32.dtype == np.dtype("<f4")
        np.testing.assert_array_equal(float32[:], np.array(
            [1.5, -2.25, 3.0], dtype="<f4"
        ))

        wide_type, wide_rows = _raw_values(generated["wide"])
        assert wide_type.get_class() == h5py.h5t.INTEGER
        assert wide_type.get_size() == 16
        wide_expected = [123, -456, 2**80, -(2**80)]
        wide_actual = [
            _decode_integer(row, wide_type, signed=True) for row in wide_rows
        ]
        assert wide_actual == wide_expected, (wide_actual, wide_expected)

        time_type, time_rows = _raw_values(generated["time"])
        assert time_type.get_class() == h5py.h5t.TIME
        assert time_type.get_size() == 8
        assert [int.from_bytes(row, "little") for row in time_rows] == [
            0,
            1_700_000_000,
        ]

        bitfield_type, bitfield_rows = _raw_values(generated["bitfield"])
        assert bitfield_type.get_class() == h5py.h5t.BITFIELD
        assert [
            int.from_bytes(row, "little") & 0xFFF
            for row in bitfield_rows
        ] == [0, 240, 4095]

        assert generated["fixed_names"].asstr()[:].tolist() == [
            "alpha",
            "月亮",
            "",
        ]
        assert generated["vlen_names"].asstr()[:].tolist() == ["moon", "数据"]

        states = generated["states"]
        assert states.dtype.metadata == {
            "enum": {"UNKNOWN": 0, "OPEN": 1, "CLOSED": 2}
        }
        np.testing.assert_array_equal(states[:], [1, 2, 0])

        records = generated["records"]
        assert records.dtype.names == ("weight", "id", "state", "note")
        assert records.dtype["weight"] == np.dtype("<f8")
        assert records.dtype["id"] == np.dtype("<i8")
        assert records.dtype["note"] == np.dtype("S6")
        assert records.dtype["state"].metadata == {
            "enum": {"UNKNOWN": 0, "OPEN": 1, "CLOSED": 2}
        }
        record_values = records[:]
        np.testing.assert_array_equal(
            record_values["weight"], [1.5, -2.25]
        )
        np.testing.assert_array_equal(record_values["id"], [7, -2])
        np.testing.assert_array_equal(record_values["state"], [0, 0])
        np.testing.assert_array_equal(record_values["note"], [b"", b""])

        arrays = generated["arrays"]
        assert arrays.dtype == np.dtype(("<f8", (2, 2)))
        np.testing.assert_array_equal(
            arrays[:],
            np.array(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[-1.0, -2.0], [-3.0, -4.0]],
                ],
                dtype="<f8",
            ),
        )

        sequences = generated["sequences"][:]
        assert [len(item) for item in sequences] == [3, 0, 2]
        np.testing.assert_array_equal(sequences[0], [1.0, -2.0, 3.0])
        np.testing.assert_array_equal(sequences[1], np.array([], dtype="<f8"))
        np.testing.assert_array_equal(sequences[2], [300.0, 400.0])

        complex_compound = generated["complex_compound"]
        compound_type = complex_compound.id.get_type()
        assert compound_type.get_class() == h5py.h5t.COMPOUND
        assert [
            compound_type.get_member_name(index)
            for index in range(compound_type.get_nmembers())
        ] == [b"r", b"i"]
        np.testing.assert_array_equal(complex_compound[:], [5.0 - 6.0j])

        complex_array = generated["complex_array"]
        assert complex_array.dtype == np.dtype(("<f8", (2,)))
        np.testing.assert_array_equal(
            complex_array[:],
            np.array([[7.0, -8.0], [9.0, 10.0]], dtype="<f8"),
        )

        chunked = generated["chunked"]
        assert chunked.dtype == np.dtype("<i4")
        assert chunked.chunks == (2,)
        assert chunked.compression == "gzip"
        np.testing.assert_array_equal(chunked[:], [10, 11, 12, 13, 14])

        growing = generated["growing"]
        assert growing.dtype == np.dtype("<f8")
        assert growing.chunks == (2,)
        np.testing.assert_array_equal(growing[:], [1.0, 2.0, 3.0, 4.0])

        file_added = generated["file_added"]
        assert file_added.dtype == np.dtype("<i2")
        np.testing.assert_array_equal(file_added[:], [7, 8])
        assert file_added.attrs["labels"].tolist() == [b"x", b"yz"]

        imported_records = imported["records"]
        assert imported_records.dtype.names == (
            "label",
            "id",
            "extra",
            "state",
            "samples",
        )
        imported_values = imported_records[:]
        np.testing.assert_array_equal(
            imported_values["label"], [b"alpha", b"xy", b"z"]
        )
        np.testing.assert_array_equal(imported_values["id"], [7, -2, 99])
        np.testing.assert_array_equal(
            imported_values["extra"], [b"", b"", b""]
        )
        np.testing.assert_array_equal(imported_values["state"], [1, 0, 2])
        np.testing.assert_array_equal(
            imported_values["samples"],
            np.array(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[-1.0, -2.0], [-3.0, -4.0]],
                    [[10.0, 20.0], [30.0, 40.0]],
                ],
                dtype="<f8",
            ),
        )

        prototype = imported.attrs["prototype"]
        _assert_compound_value(
            prototype,
            {
                "label": b"alpha",
                "id": 7,
                "extra": b"",
                "state": 1,
                "samples": np.array([[1.0, 2.0], [3.0, 4.0]], dtype="<f8"),
            },
        )

        points = imported["states_points"]
        assert points.dtype == np.dtype("<i8")
        np.testing.assert_array_equal(points[:], [0, 1])
        assert imported["names_fixed"].asstr()[:].tolist() == [
            "月亮",
            "HDF5",
            "",
        ]
        assert imported["vectors"].dtype == np.dtype(("<f8", (2,)))
        np.testing.assert_array_equal(
            imported["vectors"][:],
            np.array([[1.5, -2.0], [3.25, 4.5]], dtype="<f8"),
        )

        imported_sequences = imported["sequences"][:]
        assert [len(item) for item in imported_sequences] == [2, 1]
        assert all(
            item.dtype == np.dtype([("id", "<i8"), ("value", "<f8")])
            for item in imported_sequences
        )
        np.testing.assert_array_equal(
            imported_sequences[0],
            np.array([(1, 1.5), (2, 2.5)], dtype=[("id", "<i8"), ("value", "<f8")]),
        )
        np.testing.assert_array_equal(
            imported_sequences[1],
            np.array([(3, 3.5)], dtype=[("id", "<i8"), ("value", "<f8")]),
        )

    print(
        f"Verified {path.name} conversion interoperability with h5py "
        f"{h5py.__version__} and HDF5 {h5py.version.hdf5_version}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: verify_conversion_interop.py <conversion-output.h5>"
        )
    verify(Path(sys.argv[1]))
