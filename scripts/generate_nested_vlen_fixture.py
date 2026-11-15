from pathlib import Path

import h5py
import numpy as np


def require_hdf5_1_14_6() -> None:
    if h5py.version.hdf5_version != "1.14.6":
        raise RuntimeError(
            "nested variable-length fixture requires HDF5 1.14.6, found "
            + h5py.version.hdf5_version
        )


def generate(destination: Path) -> None:
    require_hdf5_1_14_6()
    string_type = h5py.string_dtype(encoding="utf-8")
    nested_type = h5py.vlen_dtype(string_type)
    with h5py.File(destination, "w", libver="latest") as handle:
        contiguous = handle.create_dataset(
            "nested_sequences",
            shape=(3,),
            dtype=nested_type,
            track_times=False,
        )
        contiguous[0] = np.array([], dtype=object)
        contiguous[1] = np.array(["alpha", "", "数据"], dtype=object)
        contiguous[2] = np.array(["重复"], dtype=object)

        chunked = handle.create_dataset(
            "nested_sequences_chunked",
            shape=(4,),
            dtype=nested_type,
            chunks=(2,),
            track_times=False,
        )
        chunked[0] = np.array([], dtype=object)
        chunked[1] = np.array(["first", ""], dtype=object)
        chunked[2] = np.array(["旧", "值"], dtype=object)
        chunked[3] = np.array([], dtype=object)

        template = np.empty((1,), dtype=object)
        template[0] = np.array(["x", "y"], dtype=object)
        handle.attrs.create("nested_template", template, dtype=nested_type)


if __name__ == "__main__":
    generate(Path("testdata/nested_vlen.h5"))
