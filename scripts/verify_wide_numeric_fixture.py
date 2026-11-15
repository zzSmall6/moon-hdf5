"""Independently verify wide integer and extended floating-point fixtures."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import h5py
import numpy as np


def _raw_values(dataset) -> tuple[object, list[bytes]]:
    datatype = dataset.id.get_type()
    space = dataset.id.get_space()
    count = space.get_simple_extent_npoints()
    payload = np.zeros((count, datatype.get_size()), dtype=np.uint8)
    dataset.id.read(h5py.h5s.ALL, h5py.h5s.ALL, payload, mtype=datatype)
    space.close()
    return datatype, [bytes(row) for row in payload]


def _decode_integer(raw: bytes, datatype, signed: bool) -> int:
    size = datatype.get_size()
    precision = datatype.get_precision()
    offset = datatype.get_offset()
    order = datatype.get_order()
    value = int.from_bytes(raw, "little" if order == h5py.h5t.ORDER_LE else "big")
    value = (value >> offset) & ((1 << precision) - 1)
    if signed and value & (1 << (precision - 1)):
        value -= 1 << precision
    return value


def _decode_float(raw: bytes, datatype) -> float:
    size = datatype.get_size()
    precision = datatype.get_precision()
    offset = datatype.get_offset()
    sign_position, exponent_position, exponent_size, mantissa_position, mantissa_size = (
        datatype.get_fields()
    )
    order = datatype.get_order()
    bits = int.from_bytes(raw, "little" if order == h5py.h5t.ORDER_LE else "big")
    bits = (bits >> offset) & ((1 << precision) - 1)
    negative = (bits >> sign_position) & 1
    exponent = (bits >> exponent_position) & ((1 << exponent_size) - 1)
    mantissa = (bits >> mantissa_position) & ((1 << mantissa_size) - 1)
    normalization = datatype.get_norm()
    exponent_bias = datatype.get_ebias()
    max_exponent = (1 << exponent_size) - 1
    fraction = (
        mantissa
        if normalization == h5py.h5t.NORM_IMPLIED
        else mantissa & ((1 << (mantissa_size - 1)) - 1)
    )
    if exponent == max_exponent:
        value = math.inf if fraction == 0 else math.nan
        return -value if negative else value
    if normalization == h5py.h5t.NORM_IMPLIED:
        if exponent == 0:
            significand = mantissa
            binary_exponent = 1 - exponent_bias - mantissa_size
        else:
            significand = mantissa | (1 << mantissa_size)
            binary_exponent = exponent - exponent_bias - mantissa_size
    else:
        significand = mantissa
        binary_exponent = (
            1 - exponent_bias - (mantissa_size - 1)
            if exponent == 0
            else exponent - exponent_bias - (mantissa_size - 1)
        )
    try:
        value = math.ldexp(float(significand), binary_exponent)
    except OverflowError:
        value = math.inf
    return -value if negative else value


def _assert_same(expected: float, actual: float) -> None:
    if math.isnan(expected):
        if not math.isnan(actual):
            raise AssertionError(f"expected NaN, found {actual!r}")
    elif expected == 0.0:
        if actual != 0.0 or math.copysign(1.0, expected) != math.copysign(1.0, actual):
            raise AssertionError(f"expected {expected!r}, found {actual!r}")
    elif math.isinf(expected) or math.isinf(actual):
        if expected != actual:
            raise AssertionError(f"expected {expected!r}, found {actual!r}")
    elif not math.isclose(expected, actual, rel_tol=0.0, abs_tol=0.0):
        raise AssertionError(f"expected {expected!r}, found {actual!r}")


def verify(path: Path) -> None:
    if h5py.version.hdf5_version != "1.14.6":
        raise SystemExit(
            "fixture verification requires HDF5 1.14.6, found "
            + h5py.version.hdf5_version
        )
    signed_expected = [123, -456, 2**100, -(2**100)]
    unsigned_expected = [0, 2**64, 2**127 + 5, 2**128 - 1]
    offset_expected = [123, -456, 2**70]
    float_expected = [
        1.5,
        -2.25,
        0.125,
        0.0,
        -0.0,
        math.inf,
        -math.inf,
        math.nan,
        0.0,
        math.inf,
    ]
    with h5py.File(path, "r") as file:
        for name, expected, signed in [
            ("int128_le", signed_expected, True),
            ("int128_be", signed_expected, True),
            ("uint128_le", unsigned_expected, False),
            ("int96_offset_le", offset_expected, True),
        ]:
            datatype, rows = _raw_values(file[name])
            actual = [_decode_integer(row, datatype, signed) for row in rows]
            if actual != expected:
                raise AssertionError(f"{name}: expected {expected}, found {actual}")
        for name in [
            "float128",
            "float80_le",
            "float80_be",
            "float_implied_wide",
            "float_msbset",
        ]:
            datatype, rows = _raw_values(file[name])
            actual = [_decode_float(row, datatype) for row in rows]
            if len(actual) != len(float_expected):
                raise AssertionError(f"{name}: unexpected element count")
            for expected, found in zip(float_expected, actual):
                _assert_same(expected, found)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_wide_numeric_fixture.py <file.h5>")
    verify(Path(sys.argv[1]))
