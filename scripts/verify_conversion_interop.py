"""Verify MoonBit datatype conversion output with HDF5 1.14.6 and h5py."""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np


HALF_VALUES = [
    0.0,
    -0.0,
    1.0 / 67108864.0,
    1.0 / 33554432.0,
    1.0 / 33554432.0 + 1.0 / 1152921504606846976.0,
    3.0 / 33554432.0,
    1.0 / 16777216.0,
    1.0 + 1.0 / 2048.0,
    1.0 + 1.0 / 2048.0 + 1.0 / 1099511627776.0,
    1.0 + 1.0 / 1024.0,
    65504.0,
    65519.0,
    65520.0,
    -65504.0,
    -65519.0,
    -65520.0,
]

EXTENDED_VALUES = [
    1.5,
    -2.25,
    1.0 / 3.0,
    0.0,
    -0.0,
    math.inf,
    -math.inf,
    math.nan,
]


def _raw_values(dataset: h5py.Dataset) -> tuple[h5py.h5t.TypeID, list[bytes]]:
    datatype = dataset.id.get_type()
    space = dataset.id.get_space()
    count = space.get_simple_extent_npoints()
    payload = np.zeros((count, datatype.get_size()), dtype=np.uint8)
    dataset.id.read(h5py.h5s.ALL, h5py.h5s.ALL, payload, mtype=datatype)
    space.close()
    return datatype, [bytes(row) for row in payload]


def _raw_attribute(
    attribute: h5py.h5a.AttrID,
) -> tuple[h5py.h5t.TypeID, list[bytes]]:
    datatype = attribute.get_type()
    space = attribute.get_space()
    count = space.get_simple_extent_npoints()
    payload = np.zeros((count, datatype.get_size()), dtype=np.uint8)
    attribute.read(payload, mtype=datatype)
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


def _integer_values(dataset: h5py.Dataset) -> list[int]:
    datatype, rows = _raw_values(dataset)
    assert datatype.get_class() == h5py.h5t.INTEGER
    signed = datatype.get_sign() > 0
    return [_decode_integer(row, datatype, signed) for row in rows]


def _hdf5_conversion_rows(
    values: list[object], source_dtype: str, target_type: h5py.h5t.TypeID
) -> list[bytes]:
    array = np.asarray(values, dtype=source_dtype)
    with h5py.File("conversion-reference.h5", "w", driver="core", backing_store=False) as reference:
        dataset = reference.create_dataset("values", data=array)
        payload = np.zeros((len(values), target_type.get_size()), dtype=np.uint8)
        dataset.id.read(h5py.h5s.ALL, h5py.h5s.ALL, payload, mtype=target_type)
    return [bytes(row) for row in payload]


def _assert_hdf5_conversion(
    dataset: h5py.Dataset, values: list[object], source_dtype: str
) -> None:
    target_type = dataset.id.get_type()
    expected = _hdf5_conversion_rows(values, source_dtype, target_type)
    _, actual = _raw_values(dataset)
    assert actual == expected, (dataset.name, actual, expected)


def _assert_float_vector(actual: np.ndarray, expected: list[float]) -> None:
    assert len(actual) == len(expected), (actual, expected)
    for index, (converted, source) in enumerate(zip(actual, expected)):
        if math.isnan(source):
            assert math.isnan(float(converted)), (index, converted)
        elif math.isinf(source):
            assert math.isinf(float(converted)), (index, converted)
            assert math.copysign(1.0, float(converted)) == math.copysign(1.0, source)
        else:
            assert float(converted) == source, (index, converted, source)
        if source == 0.0:
            assert bool(np.signbit(converted)) == (
                math.copysign(1.0, source) < 0.0
            )


def _assert_compound_value(actual: np.void, expected: dict[str, object]) -> None:
    for name, value in expected.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(actual[name], value)
        elif isinstance(value, float):
            assert actual[name] == value, (name, actual[name], value)
        else:
            assert actual[name] == value, (name, actual[name], value)


def verify(
    path: Path,
    numeric_fixture: Path | None = None,
    wide_fixture: Path | None = None,
) -> None:
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
        _assert_hdf5_conversion(float32, [1.5, -2.25, 3.0], "<f8")

        float16 = generated["float16_boundaries"]
        assert float16.dtype == np.dtype("<f2")
        float16_type, float16_rows = _raw_values(float16)
        assert float16_type.get_class() == h5py.h5t.FLOAT
        assert float16_type.get_size() == 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            expected_half = np.asarray(HALF_VALUES, dtype="<f8").astype("<f2")
        assert [int.from_bytes(row, "little") for row in float16_rows] == (
            expected_half.view("<u2").tolist()
        )
        _assert_float_vector(float16[:], [float(value) for value in expected_half])
        half_attribute_type, half_attribute_rows = _raw_attribute(
            float16.attrs.get_id("edge_values")
        )
        assert half_attribute_type.get_class() == h5py.h5t.FLOAT
        assert half_attribute_type.get_size() == 2
        assert [
            int.from_bytes(row, "little") for row in half_attribute_rows
        ] == expected_half[:5].view("<u2").tolist()

        for name in ("x87_16_le", "x87_16_be"):
            extended = generated[name]
            extended_type = extended.id.get_type()
            assert extended_type.get_class() == h5py.h5t.FLOAT
            assert extended_type.get_size() == 16
            assert extended_type.get_precision() == 80
            assert extended_type.get_fields() == (79, 64, 15, 0, 63)
            _assert_float_vector(extended[:], EXTENDED_VALUES)

        integer_be = generated["integer_be"]
        integer_be_type = integer_be.id.get_type()
        assert integer_be_type.get_order() == h5py.h5t.ORDER_BE
        assert _integer_values(integer_be) == [-32768, -1, 0, 32767]
        _assert_hdf5_conversion(
            integer_be, [-32768, -1, 0, 32767], "<i8"
        )

        padded = generated["integer_padded"]
        padded_type = padded.id.get_type()
        assert padded_type.get_precision() == 12
        assert padded_type.get_offset() == 4
        assert _integer_values(padded) == [-2048, -1, 0, 2047]
        _assert_hdf5_conversion(
            padded, [-2048, -1, 0, 2047], "<i8"
        )

        widened = generated["integer_widened"]
        widened_type = widened.id.get_type()
        assert widened_type.get_size() == 16
        assert widened_type.get_precision() == 128
        assert _integer_values(widened) == [-1, 0, 1]
        _assert_hdf5_conversion(widened, [-1, 0, 1], "<i8")

        unsigned_be = generated["integer_unsigned_be"]
        unsigned_be_type = unsigned_be.id.get_type()
        assert unsigned_be_type.get_order() == h5py.h5t.ORDER_BE
        assert unsigned_be_type.get_sign() == 0
        assert _integer_values(unsigned_be) == [0, 1, 2147483647]
        _assert_hdf5_conversion(
            unsigned_be, [0, 1, 2147483647], "<i8"
        )

        opaque_widened = generated["opaque_widened"]
        widened_type, widened_rows = _raw_values(opaque_widened)
        assert widened_type.get_class() == h5py.h5t.OPAQUE
        assert widened_type.get_size() == 6
        assert widened_rows == [b"ABCD\x00\x00", b"WXYZ\x00\x00"]
        opaque_truncated = generated["opaque_truncated"]
        truncated_type, truncated_rows = _raw_values(opaque_truncated)
        assert truncated_type.get_class() == h5py.h5t.OPAQUE
        assert truncated_type.get_size() == 2
        assert truncated_rows == [b"AB", b"WX"]

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

        complex_float32 = generated["complex_float32"]
        assert complex_float32.dtype == np.dtype("<c8")
        complex32_type = complex_float32.id.get_type()
        assert [
            complex32_type.get_member_name(index)
            for index in range(complex32_type.get_nmembers())
        ] == [b"r", b"i"]
        assert complex32_type.get_member_type(0).get_size() == 4
        assert complex32_type.get_member_type(1).get_size() == 4
        np.testing.assert_array_equal(
            complex_float32[:],
            np.array(
                [complex(1.5, -2.25), complex(1.0 / 3.0, -7.0 / 9.0)],
                dtype="<c8",
            ),
        )

        chunked = generated["chunked"]
        assert chunked.dtype == np.dtype("<i4")
        assert chunked.chunks == (2,)
        assert chunked.compression == "gzip"
        np.testing.assert_array_equal(chunked[:], [10, 11, 12, 13, 14])
        np.testing.assert_array_equal(
            generated["chunk_bridge"][:],
            [10, 11, 12, 13, 14],
        )
        np.testing.assert_array_equal(
            generated["selection_bridge"][:],
            [14, 10, 12],
        )

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

        if numeric_fixture is not None:
            with h5py.File(numeric_fixture, "r") as numeric_source:
                for output_name, source_name in (
                    ("numeric_offset", "/offset"),
                    ("numeric_precision", "/precision"),
                    ("numeric_big_endian", "/big_endian"),
                    ("numeric_signed20", "/signed_20"),
                    ("numeric_unsigned40", "/unsigned_40_be"),
                ):
                    assert _integer_values(imported[output_name]) == _integer_values(
                        numeric_source[source_name]
                    ), output_name
                for output_name, source_name in (
                    ("numeric_half", "/half_special"),
                    ("numeric_custom_float", "/custom_binary_float"),
                ):
                    output = imported[output_name]
                    assert output.dtype == np.dtype("<f8")
                    _assert_float_vector(
                        output[:],
                        [float(value) for value in numeric_source[source_name][:]],
                    )

        if wide_fixture is not None:
            with h5py.File(wide_fixture, "r") as wide_source:
                for output_name, source_name in (
                    ("wide_int128_le", "/int128_le"),
                    ("wide_int128_be", "/int128_be"),
                    ("wide_uint128_le", "/uint128_le"),
                    ("wide_int96_offset", "/int96_offset_le"),
                ):
                    assert _integer_values(imported[output_name]) == _integer_values(
                        wide_source[source_name]
                    ), output_name
                for output_name, source_name in (
                    ("wide_float80_le", "/float80_le"),
                    ("wide_float80_be", "/float80_be"),
                ):
                    output = imported[output_name]
                    assert output.dtype == np.dtype("<f8")
                    _assert_float_vector(
                        output[:],
                        [float(value) for value in wide_source[source_name][:]],
                    )

    print(
        f"Verified {path.name} conversion interoperability with h5py "
        f"{h5py.__version__} and HDF5 {h5py.version.hdf5_version}"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 5:
        raise SystemExit(
            "usage: verify_conversion_interop.py <conversion-output.h5> "
            "[datatypes.h5] [numeric-layouts.h5] [wide-numeric.h5]"
        )
    numeric = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    wide = Path(sys.argv[4]) if len(sys.argv) > 4 else None
    verify(Path(sys.argv[1]), numeric, wide)
