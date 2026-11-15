"""Verify MoonBit external version-4 reference output with HDF5 1.14.6."""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import h5py
import numpy as np

from generate_v4_reference_fixture import load_hdf5

HID_T = ctypes.c_longlong


def configure(library: ctypes.CDLL) -> None:
    library.H5Dread.argtypes = [
        HID_T,
        HID_T,
        HID_T,
        HID_T,
        HID_T,
        ctypes.c_void_p,
    ]
    library.H5Dread.restype = ctypes.c_int
    library.H5Aread.argtypes = [HID_T, HID_T, ctypes.c_void_p]
    library.H5Aread.restype = ctypes.c_int
    library.H5Rget_file_name.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.H5Rget_file_name.restype = ctypes.c_ssize_t
    library.H5Ropen_object.argtypes = [ctypes.c_void_p, HID_T, HID_T]
    library.H5Ropen_object.restype = HID_T
    library.H5Ropen_attr.argtypes = [ctypes.c_void_p, HID_T, HID_T]
    library.H5Ropen_attr.restype = HID_T
    library.H5Iget_name.argtypes = [HID_T, ctypes.c_char_p, ctypes.c_size_t]
    library.H5Iget_name.restype = ctypes.c_ssize_t
    library.H5Aget_name.argtypes = [HID_T, ctypes.c_size_t, ctypes.c_char_p]
    library.H5Aget_name.restype = ctypes.c_ssize_t
    library.H5Oclose.argtypes = [HID_T]
    library.H5Aclose.argtypes = [HID_T]
    library.H5Rdestroy.argtypes = [ctypes.c_void_p]
    library.H5Rdestroy.restype = ctypes.c_int


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


def read_attribute_reference(
    library: ctypes.CDLL, attribute: h5py.AttributeManager, name: str
) -> ctypes.Array:
    reference_type = ctypes.c_longlong.in_dll(
        library, "H5T_STD_REF_g"
    ).value
    buffer = (ctypes.c_ubyte * 64)()
    attribute_id = attribute.get_id(name)
    if library.H5Aread(attribute_id.id, reference_type, ctypes.byref(buffer)) < 0:
        raise AssertionError("H5Aread failed for attribute " + name)
    return buffer


def destroy_references(
    library: ctypes.CDLL, buffer: ctypes.Array, count: int
) -> None:
    for index in range(count):
        if library.H5Rdestroy(ctypes.byref(buffer, index * 64)) < 0:
            raise AssertionError("H5Rdestroy failed")


def reference_file_name(
    library: ctypes.CDLL, buffer: ctypes.Array, index: int
) -> str | None:
    name = ctypes.create_string_buffer(512)
    length = library.H5Rget_file_name(
        ctypes.byref(buffer, index * 64), name, len(name)
    )
    return None if length < 0 else name.value.decode("utf-8")


def reference_object_name(
    library: ctypes.CDLL, buffer: ctypes.Array, index: int
) -> str | None:
    object_id = library.H5Ropen_object(
        ctypes.byref(buffer, index * 64), 0, 0
    )
    if object_id < 0:
        return None
    try:
        name = ctypes.create_string_buffer(512)
        length = library.H5Iget_name(object_id, name, len(name))
        return None if length < 0 else name.value.decode("utf-8")
    finally:
        library.H5Oclose(object_id)


def reference_attribute_name(
    library: ctypes.CDLL, buffer: ctypes.Array, index: int
) -> str | None:
    attribute_id = library.H5Ropen_attr(
        ctypes.byref(buffer, index * 64), 0, 0
    )
    if attribute_id < 0:
        return None
    try:
        name = ctypes.create_string_buffer(512)
        length = library.H5Aget_name(attribute_id, len(name), name)
        return None if length < 0 else name.value.decode("utf-8")
    finally:
        library.H5Aclose(attribute_id)


def main(target_path: Path, source_path: Path, patched_path: Path) -> None:
    if h5py.version.hdf5_version != "1.14.6":
        raise RuntimeError(
            f"expected HDF5 1.14.6, found {h5py.version.hdf5_version}"
        )
    target_path = target_path.resolve()
    source_path = source_path.resolve()
    patched_path = patched_path.resolve()
    library = load_hdf5()
    configure(library)

    previous = Path.cwd()
    os.chdir(target_path.parent)
    try:
        with h5py.File(target_path.name, "r") as target:
            np.testing.assert_array_equal(target["matrix"][:], [[0, 1, 2], [3, 4, 5]])
            assert target["matrix"].attrs["unit"] == 7
            assert bytes(target["matrix"].attrs["label"]).startswith(b"matrix")

        with h5py.File(source_path.name, "r") as source:
            objects = read_references(library, source["object_refs"], 3)
            assert [
                reference_file_name(library, objects, index)
                for index in range(3)
            ] == [target_path.name] * 3
            assert [
                reference_object_name(library, objects, index)
                for index in range(3)
            ] == ["/matrix", "/group", "/group/child"]
            destroy_references(library, objects, 3)

            regions = read_references(library, source["region_refs"], 2)
            assert [
                reference_file_name(library, regions, index)
                for index in range(2)
            ] == [target_path.name] * 2
            assert [
                reference_object_name(library, regions, index)
                for index in range(2)
            ] == ["/matrix", "/matrix"]
            destroy_references(library, regions, 2)

            attributes = read_references(library, source["attribute_refs"], 1)
            assert reference_file_name(library, attributes, 0) == target_path.name
            assert reference_attribute_name(library, attributes, 0) == "unit"
            destroy_references(library, attributes, 1)

            remote = read_attribute_reference(
                library, source["object_refs"].attrs, "remote"
            )
            assert reference_file_name(library, remote, 0) == target_path.name
            assert reference_attribute_name(library, remote, 0) == "unit"
            destroy_references(library, remote, 1)

        patched_output = patched_path.read_bytes()
        source_output = source_path.read_bytes()
        assert patched_output != source_output
        with h5py.File(patched_path.name, "r") as patched:
            objects = read_references(library, patched["object_refs"], 3)
            assert [
                reference_object_name(library, objects, index)
                for index in range(3)
            ] == ["/group/child", "/matrix", "/group"]
            destroy_references(library, objects, 3)
            regions = read_references(library, patched["region_refs"], 2)
            assert [
                reference_file_name(library, regions, index)
                for index in range(2)
            ] == [target_path.name] * 2
            destroy_references(library, regions, 2)
            regional = read_references(library, patched["attribute_refs"], 1)
            assert reference_attribute_name(library, regional, 0) == "unit"
            destroy_references(library, regional, 1)
            remote = read_attribute_reference(
                library, patched["object_refs"].attrs, "remote"
            )
            assert reference_file_name(library, remote, 0) == target_path.name
            assert reference_attribute_name(library, remote, 0) == "label"
            destroy_references(library, remote, 1)
    finally:
        os.chdir(previous)

    print(f"Verified external reference output with h5py {h5py.__version__}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: verify_external_reference_output.py <target.h5> <source.h5> <patched.h5>"
        )
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
