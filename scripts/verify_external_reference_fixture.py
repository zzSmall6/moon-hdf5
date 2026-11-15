"""Verify external-file version-4 references with the native HDF5 library."""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import h5py
import numpy as np

from generate_v4_reference_fixture import load_hdf5

HID_T = ctypes.c_longlong


def read_references(
    library: ctypes.CDLL, dataset: h5py.Dataset, count: int
) -> ctypes.Array:
    reference_type = ctypes.c_longlong.in_dll(
        library, "H5T_STD_REF_g"
    ).value
    buffer = (ctypes.c_ubyte * 64 * count)()
    if (
        library.H5Dread(
            dataset.id.id,
            reference_type,
            0,
            0,
            0,
            ctypes.byref(buffer),
        )
        < 0
    ):
        raise AssertionError("H5Dread failed for " + dataset.name)
    return buffer


def reference_file_name(
    library: ctypes.CDLL, buffer: ctypes.Array, index: int
) -> str | None:
    name = ctypes.create_string_buffer(512)
    length = library.H5Rget_file_name(
        ctypes.byref(buffer, index * 64), name, len(name)
    )
    if length < 0:
        return None
    return name.value.decode("utf-8")


def verify(path: Path) -> None:
    path = path.resolve()
    target_path = path.with_name("external_reference_target.h5")
    if not target_path.exists():
        raise SystemExit("missing external reference target fixture")

    library = load_hdf5()
    library.H5Dread.argtypes = [
        HID_T,
        HID_T,
        HID_T,
        HID_T,
        HID_T,
        ctypes.c_void_p,
    ]
    library.H5Dread.restype = ctypes.c_int
    library.H5Rget_file_name.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.H5Rget_file_name.restype = ctypes.c_ssize_t

    previous = Path.cwd()
    os.chdir(path.parent)
    try:
        with h5py.File(target_path.name, "r") as target:
            np.testing.assert_array_equal(
                target["target"][:], np.arange(6, dtype="<i4")
            )
            with h5py.File(path.name, "r") as source:
                object_refs = source["object_refs"]
                objects = read_references(library, object_refs, 5)
                expected = [
                    "external_reference_target.h5",
                    "external_reference_target.h5",
                    "external_reference_target.h5",
                    "external_references.h5",
                    None,
                ]
                assert [
                    reference_file_name(library, objects, index)
                    for index in range(5)
                ] == expected

                region_refs = source["region_refs"]
                regions = read_references(library, region_refs, 5)
                assert [
                    reference_file_name(library, regions, index)
                    for index in range(4)
                ] == [
                    "external_reference_target.h5",
                    "external_reference_target.h5",
                    "external_reference_target.h5",
                    "external_reference_target.h5",
                ]
                assert reference_file_name(library, regions, 4) is None

                attribute_refs = source["attribute_refs"]
                attributes = read_references(library, attribute_refs, 2)
                assert (
                    reference_file_name(library, attributes, 0)
                    == "external_reference_target.h5"
                )
                assert reference_file_name(library, attributes, 1) is None
    finally:
        os.chdir(previous)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: verify_external_reference_fixture.py <file.h5>"
        )
    verify(Path(sys.argv[1]))
