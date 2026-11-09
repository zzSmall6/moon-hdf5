"""Generate HDF5 1.14.6 version-4 reference fixtures with h5py's native library."""

from __future__ import annotations

import ctypes
import glob
from pathlib import Path

import h5py
import numpy as np

HID_T = ctypes.c_longlong
H5P_DEFAULT = 0
H5S_SELECT_SET = 0


def load_hdf5() -> ctypes.CDLL:
    package = Path(h5py.__file__).resolve().parent
    candidates = glob.glob(str(package.parent / "h5py.libs" / "libhdf5-*.so*"))
    if not candidates:
        raise SystemExit("h5py does not provide a bundled libhdf5")
    return ctypes.CDLL(candidates[0])


def configure(library: ctypes.CDLL) -> None:
    library.H5Screate_simple.restype = HID_T
    library.H5Screate_simple.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_ulonglong),
        ctypes.POINTER(ctypes.c_ulonglong),
    ]
    library.H5Scopy.restype = HID_T
    library.H5Scopy.argtypes = [HID_T]
    library.H5Dget_space.restype = HID_T
    library.H5Dget_space.argtypes = [HID_T]
    library.H5Sclose.argtypes = [HID_T]
    library.H5Sselect_all.restype = ctypes.c_int
    library.H5Sselect_all.argtypes = [HID_T]
    library.H5Sselect_none.restype = ctypes.c_int
    library.H5Sselect_none.argtypes = [HID_T]
    library.H5Sselect_hyperslab.restype = ctypes.c_int
    library.H5Sselect_hyperslab.argtypes = [
        HID_T,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_ulonglong),
        ctypes.POINTER(ctypes.c_ulonglong),
        ctypes.POINTER(ctypes.c_ulonglong),
        ctypes.POINTER(ctypes.c_ulonglong),
    ]
    library.H5Sselect_elements.restype = ctypes.c_int
    library.H5Sselect_elements.argtypes = [
        HID_T,
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_ulonglong),
    ]
    library.H5Dcreate2.restype = HID_T
    library.H5Dcreate2.argtypes = [
        HID_T,
        ctypes.c_char_p,
        HID_T,
        HID_T,
        HID_T,
        HID_T,
        HID_T,
    ]
    library.H5Dwrite.restype = ctypes.c_int
    library.H5Dwrite.argtypes = [HID_T, HID_T, HID_T, HID_T, HID_T, ctypes.c_void_p]
    library.H5Dclose.argtypes = [HID_T]
    library.H5Acreate2.restype = HID_T
    library.H5Acreate2.argtypes = [HID_T, ctypes.c_char_p, HID_T, HID_T, HID_T, HID_T]
    library.H5Awrite.restype = ctypes.c_int
    library.H5Awrite.argtypes = [HID_T, HID_T, ctypes.c_void_p]
    library.H5Aclose.argtypes = [HID_T]
    library.H5Rcreate_object.restype = ctypes.c_int
    library.H5Rcreate_object.argtypes = [HID_T, ctypes.c_char_p, HID_T, ctypes.c_void_p]
    library.H5Rcreate_region.restype = ctypes.c_int
    library.H5Rcreate_region.argtypes = [
        HID_T,
        ctypes.c_char_p,
        HID_T,
        HID_T,
        ctypes.c_void_p,
    ]
    library.H5Rcreate_attr.restype = ctypes.c_int
    library.H5Rcreate_attr.argtypes = [
        HID_T,
        ctypes.c_char_p,
        ctypes.c_char_p,
        HID_T,
        ctypes.c_void_p,
    ]
    library.H5Rdestroy.restype = ctypes.c_int
    library.H5Rdestroy.argtypes = [ctypes.c_void_p]


class NativeReferences:
    def __init__(self, library: ctypes.CDLL) -> None:
        self.library = library
        self.reference_type = ctypes.c_longlong.in_dll(library, "H5T_STD_REF_g").value

    def reference(self) -> ctypes.Array:
        return (ctypes.c_ubyte * 64)()

    def object(self, file_id: int, path: bytes) -> ctypes.Array:
        value = self.reference()
        if self.library.H5Rcreate_object(file_id, path, H5P_DEFAULT, ctypes.byref(value)) < 0:
            raise RuntimeError("H5Rcreate_object failed")
        return value

    def region(self, dataset_id: int, space_id: int) -> ctypes.Array:
        value = self.reference()
        if (
            self.library.H5Rcreate_region(
                dataset_id, b".", space_id, H5P_DEFAULT, ctypes.byref(value)
            )
            < 0
        ):
            raise RuntimeError("H5Rcreate_region failed")
        return value

    def attribute(self, object_id: int, name: bytes) -> ctypes.Array:
        value = self.reference()
        if (
            self.library.H5Rcreate_attr(
                object_id, b".", name, H5P_DEFAULT, ctypes.byref(value)
            )
            < 0
        ):
            raise RuntimeError("H5Rcreate_attr failed")
        return value

    def null(self) -> ctypes.Array:
        return self.reference()

    def destroy(self, references: list[ctypes.Array]) -> None:
        for value in references:
            self.library.H5Rdestroy(ctypes.byref(value))

    def write_dataset(
        self,
        file_id: int,
        name: bytes,
        references: list[ctypes.Array],
        dcpl_id: int = H5P_DEFAULT,
    ) -> None:
        dimensions = (ctypes.c_ulonglong * 1)(len(references))
        space = self.library.H5Screate_simple(1, dimensions, None)
        dataset = self.library.H5Dcreate2(
            file_id,
            name,
            self.reference_type,
            space,
            H5P_DEFAULT,
            dcpl_id,
            H5P_DEFAULT,
        )
        buffer = (ctypes.c_ubyte * 64 * len(references))()
        for index, reference in enumerate(references):
            ctypes.memmove(ctypes.addressof(buffer) + index * 64, reference, 64)
        if (
            self.library.H5Dwrite(
                dataset, self.reference_type, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT, buffer
            )
            < 0
        ):
            raise RuntimeError("H5Dwrite failed for version-4 references")
        self.library.H5Dclose(dataset)
        self.library.H5Sclose(space)

    def write_scalar_attribute(
        self, object_id: int, name: bytes, reference: ctypes.Array
    ) -> None:
        space = self.library.H5Screate_simple(1, (ctypes.c_ulonglong * 1)(1), None)
        attribute = self.library.H5Acreate2(
            object_id,
            name,
            self.reference_type,
            space,
            H5P_DEFAULT,
            H5P_DEFAULT,
        )
        if (
            self.library.H5Awrite(attribute, self.reference_type, ctypes.byref(reference))
            < 0
        ):
            raise RuntimeError("H5Awrite failed for the version-4 reference")
        self.library.H5Aclose(attribute)
        self.library.H5Sclose(space)

    def copy_space(self, dataset_id: int) -> int:
        space = self.library.H5Dget_space(dataset_id)
        if space < 0:
            raise RuntimeError("H5Dget_space failed")
        return space

    def select_hyperslab(self, space_id: int, start, stride, count, block) -> None:
        def vector(values):
            return (ctypes.c_ulonglong * len(values))(*values)

        if (
            self.library.H5Sselect_hyperslab(
                space_id,
                H5S_SELECT_SET,
                vector(start),
                vector(stride),
                vector(count),
                vector(block),
            )
            < 0
        ):
            raise RuntimeError("H5Sselect_hyperslab failed")

    def select_points(self, space_id: int, points) -> None:
        values = [coordinate for point in points for coordinate in point]
        coordinates = (ctypes.c_ulonglong * len(values))(*values)
        if (
            self.library.H5Sselect_elements(
                space_id, H5S_SELECT_SET, len(points), coordinates
            )
            < 0
        ):
            raise RuntimeError("H5Sselect_elements failed")

    def select_all(self, space_id: int) -> None:
        if self.library.H5Sselect_all(space_id) < 0:
            raise RuntimeError("H5Sselect_all failed")

    def select_none(self, space_id: int) -> None:
        if self.library.H5Sselect_none(space_id) < 0:
            raise RuntimeError("H5Sselect_none failed")

    def close_space(self, space_id: int) -> None:
        self.library.H5Sclose(space_id)


def generate(destination: Path) -> None:
    if h5py.version.hdf5_version != "1.14.6":
        raise SystemExit(
            "references_v4.h5 requires HDF5 1.14.6, found " + h5py.version.hdf5_version
        )

    library = load_hdf5()
    configure(library)
    native = NativeReferences(library)

    with h5py.File(destination, "w", libver="latest") as handle:
        target = handle.create_dataset(
            "target", data=np.arange(6, dtype="<i4"), track_times=False
        )
        matrix = handle.create_dataset(
            "matrix",
            data=np.arange(30, dtype="<i4").reshape(6, 5),
            track_times=False,
        )
        group = handle.create_group("group")
        child = group.create_dataset(
            "child", data=np.array([4, 5], dtype="<i4"), track_times=False
        )
        target.attrs.create("unit", np.array(7, dtype="<i4"))
        matrix.attrs.create("unit", np.array(11, dtype="<i4"))
        datatype = h5py.h5t.py_create(np.dtype("<i4"), logical=True)
        datatype.commit(handle.id, b"named_i4")
        handle.flush()

        references = [native.object(handle.id.id, path) for path in (
            b"/target",
            b"/group",
            b"/group/child",
            b"/",
            b"/named_i4",
            b"/matrix",
        )]
        references.append(native.null())
        chunked = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        chunked.set_chunk((2,))
        chunked.set_shuffle()
        chunked.set_deflate(6)
        native.write_dataset(handle.id.id, b"object_refs", references, chunked.id)

        region_references = []
        space = native.copy_space(target.id.id)
        native.select_hyperslab(space, (1,), (2,), (2,), (1,))
        region_references.append(native.region(target.id.id, space))
        native.close_space(space)

        space = native.copy_space(target.id.id)
        native.select_points(space, ((4,), (0,), (3,)))
        region_references.append(native.region(target.id.id, space))
        native.close_space(space)

        space = native.copy_space(target.id.id)
        native.select_all(space)
        region_references.append(native.region(target.id.id, space))
        native.close_space(space)

        space = native.copy_space(target.id.id)
        native.select_none(space)
        region_references.append(native.region(target.id.id, space))
        native.close_space(space)

        space = native.copy_space(matrix.id.id)
        native.select_hyperslab(space, (0, 0), (2, 3), (2, 2), (2, 1))
        region_references.append(native.region(matrix.id.id, space))
        native.close_space(space)
        region_references.append(native.null())
        native.write_dataset(handle.id.id, b"region_refs", region_references)

        attribute_references = [
            native.attribute(target.id.id, b"unit"),
            native.attribute(matrix.id.id, b"unit"),
            native.null(),
        ]
        native.write_dataset(handle.id.id, b"attribute_refs", attribute_references)

        child_reference = native.object(handle.id.id, b"/group/child")
        native.write_scalar_attribute(target.id.id, b"child_ref", child_reference)

        all_references = (
            references
            + region_references
            + attribute_references
            + [child_reference]
        )
        handle.flush()
        native.destroy(all_references)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 2:
        generate(Path(sys.argv[1]))
    elif len(sys.argv) == 1:
        generate(Path(__file__).resolve().parents[1] / "testdata" / "references_v4.h5")
    else:
        raise SystemExit("usage: generate_v4_reference_fixture.py [file.h5]")
