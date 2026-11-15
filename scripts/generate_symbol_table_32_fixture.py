#!/usr/bin/env python3
"""Generate old-style group files with 4- and 8-byte offsets."""

from pathlib import Path

import h5py
import numpy as np


def generate(destination: Path, offset_size: int, length_size: int) -> None:
    fcpl = h5py.h5p.create(h5py.h5p.FILE_CREATE)
    fcpl.set_sizes(offset_size, length_size)
    fapl = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    fapl.set_libver_bounds(h5py.h5f.LIBVER_EARLIEST, h5py.h5f.LIBVER_V18)
    file_id = h5py.h5f.create(
        str(destination).encode(),
        h5py.h5f.ACC_TRUNC,
        fcpl,
        fapl,
    )
    with h5py.File(file_id) as handle:
        handle.attrs["title"] = "old-style 32-bit offsets"
        group = handle.create_group("group")
        group.attrs["answer"] = np.int32(42)
        values = group.create_dataset(
            "values",
            data=np.arange(5, dtype="<i4"),
        )
        values.attrs["unit"] = "count"
        handle.create_dataset(
            "matrix",
            data=np.arange(6, dtype="<i4").reshape(2, 3),
        )
        handle["alias"] = values
        handle["soft"] = h5py.SoftLink("/group/values")

    with h5py.File(destination, "r") as handle:
        assert sorted(handle) == ["alias", "group", "matrix", "soft"]
        np.testing.assert_array_equal(handle["alias"][:], np.arange(5))
        np.testing.assert_array_equal(handle["soft"][:], np.arange(5))
        assert handle.attrs["title"] == "old-style 32-bit offsets"


def main() -> None:
    directory = Path(__file__).resolve().parent.parent / "testdata"
    generate(directory / "symbol_table.h5", 8, 8)
    generate(directory / "symbol_table_32.h5", 4, 8)


if __name__ == "__main__":
    main()
