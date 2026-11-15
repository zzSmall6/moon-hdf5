"""Verify virtual-dataset writer output with HDF5 1.14.6."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np


def require_virtual(dataset: h5py.Dataset, mappings: int) -> None:
    assert dataset.is_virtual
    dcpl = dataset.id.get_create_plist()
    assert dcpl.get_layout() == 3
    count = dcpl.get_virtual_count()
    assert count == mappings, (dataset.name, count, mappings)


def verify(new_path: Path, edited_path: Path) -> None:
    if h5py.version.hdf5_version != "1.14.6":
        raise SystemExit(
            "virtual-dataset output requires HDF5 1.14.6, found "
            + h5py.version.hdf5_version
        )

    source_values = np.arange(1, 13, dtype="<i4").reshape(3, 4)
    with h5py.File(new_path, "r") as handle:
        np.testing.assert_array_equal(handle["source"][...], source_values)

        expected = {
            "same": source_values,
            "mapped": np.array([[6, 7, -1], [10, 11, -1]], dtype="<i4"),
            "strided": np.array([[1, 3], [9, 11]], dtype="<i4"),
            "empty": np.full((2, 2), -5, dtype="<i4"),
            "multiple_mappings": np.arange(6, dtype="<i4"),
        }
        mapping_counts = {
            "same": 1,
            "mapped": 1,
            "strided": 1,
            "empty": 1,
            "multiple_mappings": 2,
        }
        for name, values in expected.items():
            dataset = handle[name]
            np.testing.assert_array_equal(dataset[...], values)
            require_virtual(dataset, mapping_counts[name])
            dcpl = dataset.id.get_create_plist()
            for index in range(dcpl.get_virtual_count()):
                assert dcpl.get_virtual_filename(index) == "."
                assert dcpl.get_virtual_dsetname(index) in ("/source", "/vector")

        mapped_source = handle["mapped"].virtual_sources()[0]
        assert mapped_source.dset_name == "/source"
        assert mapped_source.src_space.get_select_type() == h5py.h5s.SEL_HYPERSLABS
        assert mapped_source.vspace.get_select_type() == h5py.h5s.SEL_HYPERSLABS

    with h5py.File(edited_path, "r") as handle:
        np.testing.assert_array_equal(
            handle["group/virtual_source"][...], np.array([7, 8], dtype="<i4")
        )
        dataset = handle["group/virtual_mapped"]
        np.testing.assert_array_equal(dataset[...], np.array([7, 8], dtype="<i4"))
        require_virtual(dataset, 1)
        source = dataset.virtual_sources()[0]
        assert source.file_name == "."
        assert source.dset_name == "/group/virtual_source"

    print(
        "Verified virtual-dataset writer output with h5py "
        f"{h5py.__version__}/HDF5 {h5py.version.hdf5_version}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: verify_virtual_dataset_write.py <new.h5> <edited.h5>"
        )
    verify(Path(sys.argv[1]), Path(sys.argv[2]))
