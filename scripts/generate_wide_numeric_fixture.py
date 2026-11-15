"""Generate fixtures for numeric storage and precision wider than 64 bits."""

from __future__ import annotations

import os
import struct
from pathlib import Path

import h5py
import numpy as np


def _little(value: int, size: int, signed: bool = True) -> bytes:
    return int(value).to_bytes(size, "little", signed=signed)


def _big(value: int, size: int, signed: bool = True) -> bytes:
    return int(value).to_bytes(size, "big", signed=signed)


def _write_raw(dataset_id, rows: list[bytes]) -> None:
    width = rows[0].__len__()
    payload = np.frombuffer(b"".join(rows), dtype=np.uint8).reshape(len(rows), width)
    dataset_id.write(h5py.h5s.ALL, h5py.h5s.ALL, payload, mtype=dataset_id.get_type())


def _create_raw(file_id, name: str, datatype, rows: list[bytes]) -> None:
    space = h5py.h5s.create_simple((len(rows),))
    dataset = h5py.h5d.create(file_id, name.encode(), datatype, space)
    try:
        _write_raw(dataset, rows)
    finally:
        dataset.close()
        space.close()


def _copy_integer(signed: bool, size: int, precision: int, offset: int = 0):
    base = h5py.h5t.py_create(np.dtype("int64" if signed else "uint64")).copy()
    base.set_size(size)
    base.set_precision(precision)
    base.set_offset(offset)
    return base


def _x87_implied_layout(datatype):
    """Return a copy that stores only the explicit x87 fraction bits."""
    implied = datatype.copy()
    implied.set_norm(h5py.h5t.NORM_IMPLIED)
    implied.set_fields(79, 64, 15, 0, 63)
    implied.set_ebias(16383)
    implied.set_precision(80)
    implied.set_size(16)
    return implied


def _x87_to_implied(raw: bytes) -> bytes:
    bits = int.from_bytes(raw[:10], "little")
    sign = (bits >> 79) & 1
    exponent = (bits >> 64) & 0x7FFF
    mantissa = bits & ((1 << 64) - 1)
    fraction = mantissa & ((1 << 63) - 1)
    packed = (sign << 79) | (exponent << 64) | fraction
    return packed.to_bytes(10, "little") + bytes(6)


def generate(destination: Path) -> None:
    if h5py.version.hdf5_version != "1.14.6":
        raise SystemExit(
            "wide_numeric.h5 requires HDF5 1.14.6, found "
            + h5py.version.hdf5_version
        )
    if np.dtype(np.longdouble).itemsize != 16 or np.finfo(np.longdouble).nmant != 63:
        raise SystemExit("wide_numeric.h5 requires 80-bit x86 extended precision")

    destination = destination.resolve()
    previous = Path.cwd()
    os.chdir(destination.parent)
    try:
        with h5py.File(destination.name, "w", libver="latest") as file:
            signed_values = [123, -456, 2**100, -(2**100)]
            int128 = _copy_integer(True, 16, 128)
            _create_raw(file.id, "int128_le", int128, [_little(v, 16) for v in signed_values])
            int128_be = int128.copy()
            int128_be.set_order(h5py.h5t.ORDER_BE)
            _create_raw(file.id, "int128_be", int128_be, [_big(v, 16) for v in signed_values])

            unsigned_values = [0, 2**64, 2**127 + 5, 2**128 - 1]
            uint128 = _copy_integer(False, 16, 128)
            _create_raw(
                file.id,
                "uint128_le",
                uint128,
                [_little(v, 16, signed=False) for v in unsigned_values],
            )

            int96 = _copy_integer(True, 12, 80, offset=4)
            offset_values = [123, -456, 2**70]
            _create_raw(
                file.id,
                "int96_offset_le",
                int96,
                [((v % (1 << 80)) << 4).to_bytes(12, "little") for v in offset_values],
            )

            base = h5py.h5t.py_create(np.dtype(np.longdouble)).copy()
            float_values = [
                1.5,
                -2.25,
                0.125,
                0.0,
                -0.0,
                np.inf,
                -np.inf,
                np.nan,
                float(np.nextafter(np.longdouble(0), np.longdouble(1))),
                1e4000,
            ]
            longdouble_rows = [np.longdouble(value).tobytes() for value in float_values]
            _create_raw(file.id, "float128", base, longdouble_rows)

            packed = base.copy()
            packed.set_size(10)
            _create_raw(file.id, "float80_le", packed, [row[:10] for row in longdouble_rows])
            packed_be = packed.copy()
            packed_be.set_order(h5py.h5t.ORDER_BE)
            _create_raw(
                file.id,
                "float80_be",
                packed_be,
                [row[:10][::-1] for row in longdouble_rows],
            )

            implied = _x87_implied_layout(base)
            _create_raw(
                file.id,
                "float_implied_wide",
                implied,
                [_x87_to_implied(row) for row in longdouble_rows],
            )

            msbset = base.copy()
            msbset.set_norm(h5py.h5t.NORM_MSBSET)
            _create_raw(file.id, "float_msbset", msbset, longdouble_rows)
    finally:
        os.chdir(previous)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 2:
        generate(Path(sys.argv[1]))
    elif len(sys.argv) == 1:
        generate(Path(__file__).resolve().parents[1] / "testdata" / "wide_numeric.h5")
    else:
        raise SystemExit("usage: generate_wide_numeric_fixture.py [file.h5]")
