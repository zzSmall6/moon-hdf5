"""Generate a four-byte-address compound-reference fixture with HDF5 1.14.6."""

from pathlib import Path

import h5py
import numpy as np


def nested_record_type():
    return np.dtype(
        [
            ("label", "S8"),
            ("obj", h5py.ref_dtype),
            ("region", h5py.regionref_dtype),
        ]
    )


def generate(destination: Path) -> None:
    if h5py.version.hdf5_version != "1.14.6":
        raise SystemExit(
            "nested_reference_32.h5 requires HDF5 1.14.6, found "
            + h5py.version.hdf5_version
        )

    fcpl = h5py.h5p.create(h5py.h5p.FILE_CREATE)
    fcpl.set_sizes(4, 8)
    fapl = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    fapl.set_libver_bounds(h5py.h5f.LIBVER_LATEST, h5py.h5f.LIBVER_LATEST)
    fid = h5py.h5f.create(
        bytes(destination), h5py.h5f.ACC_TRUNC, fcpl=fcpl, fapl=fapl
    )
    handle = h5py.File(fid)
    try:
        target = handle.create_dataset(
            "target", data=np.arange(4, dtype="<i4"), track_times=False
        )
        target2 = handle.create_dataset(
            "target2",
            data=np.arange(10, 14, dtype="<i4"),
            track_times=False,
        )
        group = handle.create_group("group")
        target.attrs.create("unit", np.array([7], dtype="<i4"))
        target2.attrs.create("unit", np.array([11], dtype="<i4"))

        datatype = nested_record_type()
        records = np.empty((2,), dtype=datatype)
        records["label"] = [b"original", b"empty"]
        records["obj"][0] = target.ref
        records["obj"][1] = h5py.Reference()
        records["region"][0] = target.regionref[1:3]
        records["region"][1] = h5py.RegionReference()
        records_dataset = handle.create_dataset(
            "records", data=records, track_times=False
        )
        records_dataset.attrs.create("prototype", records[:1])
    finally:
        handle.close()

    verify(destination)
    print(
        f"Generated {destination} with h5py {h5py.__version__} "
        f"and HDF5 {h5py.version.hdf5_version}"
    )


def verify(destination: Path) -> None:
    with h5py.File(destination, "r", libver="latest") as handle:
        assert handle.id.get_create_plist().get_sizes() == (4, 8)
        records = handle["records"]
        assert records.id.get_storage_size() == 2 * 20

        values = records[:]
        assert values["label"].tolist() == [b"original", b"empty"]
        assert handle[values["obj"][0]].name == "/target"
        assert not bool(values["obj"][1])
        region_target = handle[values["region"][0]]
        assert region_target.name == "/target"
        np.testing.assert_array_equal(
            region_target[values["region"][0]],
            np.array([1, 2], dtype="<i4"),
        )
        assert not bool(values["region"][1])

        prototype = records.attrs["prototype"]
        assert prototype["label"].tolist() == [b"original"]
        assert handle[prototype["obj"][0]].name == "/target"


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 2:
        generate(Path(sys.argv[1]))
    elif len(sys.argv) == 1:
        fixture_directory = Path(__file__).resolve().parents[1] / "testdata"
        generate(fixture_directory / "nested_reference_32.h5")
    else:
        raise SystemExit(
            "usage: generate_nested_reference_fixture.py [file.h5]"
        )
