"""Verify committed 2/4/8-byte address and length-width fixtures."""

from __future__ import annotations

import h5py

from generate_size_matrix_fixtures import SIZES, verify_fixture


def main() -> None:
    if h5py.version.hdf5_version != "1.14.6":
        raise SystemExit(
            "size matrix fixtures require HDF5 1.14.6, found "
            + h5py.version.hdf5_version
        )
    for offset_size in SIZES:
        for length_size in SIZES:
            verify_fixture(offset_size, length_size)
    print(
        "Verified 2/4/8-byte address and length-width matrix with h5py "
        f"{h5py.__version__}/HDF5 {h5py.version.hdf5_version}"
    )


if __name__ == "__main__":
    main()
