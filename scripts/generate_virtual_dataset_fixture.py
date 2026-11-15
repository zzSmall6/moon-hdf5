"""Generate finite virtual-dataset fixtures with HDF5 1.14.6."""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TESTDATA = ROOT / "testdata"


def create_four_byte_file(path: Path, mode: str) -> h5py.File:
    fcpl = h5py.h5p.create(h5py.h5p.FILE_CREATE)
    fcpl.set_sizes(4, 8)
    fapl = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    fapl.set_libver_bounds(h5py.h5f.LIBVER_LATEST, h5py.h5f.LIBVER_LATEST)
    fid = h5py.h5f.create(bytes(path), h5py.h5f.ACC_TRUNC, fcpl=fcpl, fapl=fapl)
    handle = h5py.File(fid)
    _ = mode
    return handle


def generate(destination: Path) -> None:
    if h5py.version.hdf5_version != "1.14.6":
        raise SystemExit(
            "virtual_dataset.h5 requires HDF5 1.14.6, found "
            + h5py.version.hdf5_version
        )

    destination = destination.resolve()
    previous = Path.cwd()
    os.chdir(destination.parent)
    main_name = destination.name
    nested_name = "virtual_dataset_nested.h5"
    source_name = "virtual_dataset_source.h5"
    source_32_name = "virtual_dataset_source_32.h5"
    main_32_name = "virtual_dataset_32.h5"
    try:
        for name in (
            main_name,
            main_32_name,
            nested_name,
            source_name,
            source_32_name,
        ):
            path = Path(name)
            if path.exists():
                path.unlink()

        matrix = np.arange(100, 136, dtype="<i4").reshape(6, 6)
        with h5py.File(source_name, "w", libver="latest") as source:
            source.create_dataset(
                "data",
                data=np.arange(20, dtype="<i4").reshape(4, 5),
                track_times=False,
            )
            source.create_dataset("matrix", data=matrix, track_times=False)
            source.create_dataset(
                "vector",
                data=np.arange(6, dtype="<i4"),
                track_times=False,
            )

        source_32_values = np.arange(200, 212, dtype="<i4").reshape(3, 4)
        with create_four_byte_file(Path(source_32_name), "w") as source:
            source.create_dataset(
                "data", data=source_32_values, track_times=False
            )

        with h5py.File(main_name, "w", libver="latest") as handle:
            handle.create_dataset(
                "same_source",
                data=np.arange(12, dtype="<i4").reshape(3, 4),
                track_times=False,
            )

            mapped = h5py.VirtualLayout(shape=(8, 10), dtype="<i4")
            external = h5py.VirtualSource(source_name, "/data", shape=(4, 5))
            mapped[0:4, 0:5] = external
            mapped[4:8, 5:10] = external
            handle.create_virtual_dataset("mapped", mapped, fillvalue=-1)

            same = h5py.VirtualLayout(shape=(3, 4), dtype="<i4")
            same_source = h5py.VirtualSource(".", "/same_source", shape=(3, 4))
            same[...] = same_source
            handle.create_virtual_dataset("same_file", same)

            strided = h5py.VirtualLayout(shape=(3, 3), dtype="<i4")
            strided_source = h5py.VirtualSource(
                source_name, "/matrix", shape=(6, 6)
            )
            strided[:, :] = strided_source[::2, ::2]
            handle.create_virtual_dataset("strided", strided, fillvalue=-7)

            multiple = h5py.VirtualLayout(shape=(6,), dtype="<i4")
            vector = h5py.VirtualSource(source_name, "/vector", shape=(6,))
            multiple[0:3] = vector[0:3]
            multiple[3:6] = vector[3:6]
            handle.create_virtual_dataset("multiple_mappings", multiple, fillvalue=-1)

        with create_four_byte_file(Path(main_32_name), "w") as handle:
            layout = h5py.VirtualLayout(shape=(4, 6), dtype="<i4")
            external = h5py.VirtualSource(
                source_32_name, "/data", shape=source_32_values.shape
            )
            layout[0:3, 0:4] = external
            handle.create_virtual_dataset("mapped", layout, fillvalue=-9)

        with h5py.File(nested_name, "w", libver="latest") as nested:
            layout = h5py.VirtualLayout(shape=(4, 5), dtype="<i4")
            source = h5py.VirtualSource(main_name, "/mapped", shape=(8, 10))
            layout[0:4, 0:5] = source[4:8, 5:10]
            nested.create_virtual_dataset("nested", layout, fillvalue=-3)

        verify(Path(main_name), Path(main_32_name), Path(nested_name))
    finally:
        os.chdir(previous)

    print(
        "Generated virtual dataset fixtures with h5py "
        f"{h5py.__version__}/HDF5 {h5py.version.hdf5_version}"
    )


def verify(main: Path, main_32: Path, nested: Path) -> None:
    expected_mapped = np.full((8, 10), -1, dtype="<i4")
    values = np.arange(20, dtype="<i4").reshape(4, 5)
    expected_mapped[0:4, 0:5] = values
    expected_mapped[4:8, 5:10] = values
    with h5py.File(main, "r") as handle:
        np.testing.assert_array_equal(handle["mapped"][...], expected_mapped)
        np.testing.assert_array_equal(
            handle["same_file"][...], np.arange(12, dtype="<i4").reshape(3, 4)
        )
        np.testing.assert_array_equal(
            handle["strided"][...], np.arange(100, 136, dtype="<i4").reshape(6, 6)[
                ::2, ::2
            ]
        )
        np.testing.assert_array_equal(
            handle["multiple_mappings"][...], np.arange(6, dtype="<i4")
        )
        for name in ("mapped", "same_file", "strided", "multiple_mappings"):
            assert handle[name].id.get_create_plist().get_layout() == 3

    expected_32 = np.full((4, 6), -9, dtype="<i4")
    expected_32[0:3, 0:4] = np.arange(200, 212, dtype="<i4").reshape(3, 4)
    with h5py.File(main_32, "r") as handle:
        np.testing.assert_array_equal(handle["mapped"][...], expected_32)
        assert handle["mapped"].id.get_create_plist().get_layout() == 3

    with h5py.File(nested, "r") as handle:
        np.testing.assert_array_equal(
            handle["nested"][...], np.arange(20, dtype="<i4").reshape(4, 5)
        )
        assert handle["nested"].id.get_create_plist().get_layout() == 3


if __name__ == "__main__":
    import sys

    target = (
        Path(sys.argv[1])
        if len(sys.argv) == 2
        else TESTDATA / "virtual_dataset.h5"
    )
    if len(sys.argv) > 2:
        raise SystemExit("usage: generate_virtual_dataset_fixture.py [file.h5]")
    generate(target)
