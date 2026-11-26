"""Verify raw-space reuse and subsequent HDF5 allocations independently."""

import ctypes
from pathlib import Path
import subprocess
import tempfile

import h5py
import numpy as np

from verify_free_space import Section


ROOT = Path(__file__).resolve().parents[1]


def edit(operation: str, source: Path, dataset: str, destination: Path) -> None:
    subprocess.run(
        ["moon", "run", "src/cmd/free_space_demo", "--target", "native", "--",
         operation, str(source), dataset, str(destination)],
        cwd=ROOT, check=True,
    )


def sections(handle: h5py.File) -> list[tuple[int, int]]:
    library = ctypes.CDLL(h5py.h5f.__file__)
    query = library.H5Fget_free_sections
    query.argtypes = [ctypes.c_int64, ctypes.c_int, ctypes.c_size_t,
                      ctypes.POINTER(Section)]
    query.restype = ctypes.c_ssize_t
    count = query(handle.id.id, 0, 0, None)
    assert count >= 0
    output = (Section * max(count, 1))()
    assert query(handle.id.id, 0, max(count, 1), output) == count
    return [(item.addr, item.size) for item in output[:count]]


def create(path: Path, userblock: int, unallocated: bool) -> None:
    fcpl = h5py.h5p.create(h5py.h5p.FILE_CREATE)
    fcpl.set_file_space_strategy(h5py.h5f.FSPACE_STRATEGY_FSM_AGGR, True, 1)
    if userblock:
        fcpl.set_userblock(userblock)
    fapl = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    fapl.set_libver_bounds(h5py.h5f.LIBVER_LATEST, h5py.h5f.LIBVER_LATEST)
    fid = h5py.h5f.create(bytes(path), h5py.h5f.ACC_TRUNC, fcpl=fcpl, fapl=fapl)
    with h5py.File(fid) as handle:
        handle.create_dataset("hole", data=np.arange(512, dtype="<i4"))
        if unallocated:
            handle.create_dataset("target", shape=(512,), dtype="<i4", fillvalue=-1)
        else:
            target = handle.create_dataset("target", data=["alpha", "月亮", ""],
                                           dtype=h5py.string_dtype())
            target.attrs["unit"] = "names"
            handle["alias"] = target
            references = handle.create_dataset("references", (1,), dtype=h5py.ref_dtype)
            references[0] = target.ref
        handle.create_dataset("keep", data=np.arange(64, dtype="<i4"))
        del handle["hole"]


def verify() -> None:
    if h5py.version.hdf5_version != "1.14.6":
        raise SystemExit("space reuse verification requires HDF5 1.14.6")
    with tempfile.TemporaryDirectory(prefix="moon-hdf5-reuse-") as directory:
        temporary = Path(directory)
        for mode in ("plain", "userblock"):
            created = temporary / f"created-{mode}.h5"
            moved = temporary / f"moved-{mode}.h5"
            subprocess.run(
                ["moon", "run", "src/cmd/free_space_demo", "--target", "native",
                 "--", "create", mode, str(created)], cwd=ROOT, check=True,
            )
            with h5py.File(created, "r") as handle:
                assert handle.id.get_create_plist().get_file_space_strategy() == (0, 1, 1)
                assert handle.id.get_freespace() == 8192
                free_sections = sections(handle)
                assert len(free_sections) == 1 and free_sections[0][1] == 8192
                np.testing.assert_array_equal(handle["target"][:], np.full(512, 42))
                np.testing.assert_array_equal(handle["chunks"][:], [1, 2, 3, 4])
                assert handle["target"].attrs["unit"] == b"items"
                assert handle.userblock_size == (512 if mode == "userblock" else 0)
            edit("relocate", created, "/target", moved)
            with h5py.File(moved, "r") as handle:
                assert handle.id.get_freespace() == 8192
                assert sum(length for _, length in sections(handle)) == 8192
            for path in (created, moved):
                with h5py.File(path, "r") as handle:
                    available = sections(handle)
                with h5py.File(path, "r+") as handle:
                    dataset = handle.create_dataset("new", data=np.arange(1024, dtype="<i4"))
                    offset = dataset.id.get_offset() - handle.userblock_size
                    assert any(start <= offset and offset + dataset.nbytes <= start + length
                               for start, length in available)
                with h5py.File(path, "r") as handle:
                    np.testing.assert_array_equal(handle["new"][:], np.arange(1024))
                    np.testing.assert_array_equal(handle["target"][:], np.full(512, 42))
                    np.testing.assert_array_equal(handle["second"][:], np.full(512, 7))
                    np.testing.assert_array_equal(handle["chunks"][:], [1, 2, 3, 4])
        for name in ("fsm", "width4"):
            source = ROOT / "testdata" / f"free_space_{name}.h5"
            destination = temporary / f"{name}.h5"
            with h5py.File(source, "r") as original:
                old_address = original["data_0"].id.get_offset()
                size = original["data_0"].nbytes
                free = original.id.get_freespace()
                node = h5py.h5o.get_info(original["data_0"].id).addr
            edit("relocate", source, "/data_0", destination)
            for _iteration in range(3):
                with h5py.File(destination, "r") as handle:
                    assert handle.id.get_freespace() == free
                    assert h5py.h5o.get_info(handle["data_0"].id).addr == node
                    free_sections = sections(handle)
                    assert any(start <= old_address and old_address + size <= start + length
                               for start, length in free_sections)
                    for index in (0, 2, 4, 6, 7):
                        np.testing.assert_array_equal(
                            handle[f"data_{index}"][:], np.arange(600 + 17 * index)
                        )
                    old_address = handle["data_0"].id.get_offset()
                edit("relocate", destination, "/data_0", destination)
            with h5py.File(destination, "r+") as handle:
                handle.create_dataset("new", data=np.arange(500, dtype="<i4"))
            with h5py.File(destination, "r") as handle:
                np.testing.assert_array_equal(handle["new"][:], np.arange(500))
                np.testing.assert_array_equal(handle["data_0"][:], np.arange(600))
        for userblock in (0, 512):
            for unallocated in (False, True):
                source = temporary / f"source-{userblock}-{unallocated}.h5"
                destination = temporary / f"output-{userblock}-{unallocated}.h5"
                create(source, userblock, unallocated)
                with h5py.File(source, "r") as handle:
                    free = handle.id.get_freespace()
                edit("patch" if unallocated else "relocate", source, "/target", destination)
                with h5py.File(destination, "r+") as handle:
                    assert handle.userblock_size == userblock
                    assert handle.id.get_freespace() == free - (2048 if unallocated else 0)
                    sections(handle)
                    if unallocated:
                        np.testing.assert_array_equal(handle["target"][:], np.full(512, 42))
                    else:
                        assert handle["target"].asstr()[:].tolist() == ["alpha", "月亮", ""]
                        assert handle["alias"].id == handle["target"].id
                        assert handle[handle["references"][0]].id == handle["target"].id
                        assert handle["target"].attrs["unit"] == "names"
                    handle.create_dataset("new", data=np.arange(256, dtype="<i4"))
                with h5py.File(destination, "r") as handle:
                    np.testing.assert_array_equal(handle["keep"][:], np.arange(64))
                    np.testing.assert_array_equal(handle["new"][:], np.arange(256))
    print("Verified raw-space reuse, relocation and HDF5 1.14.6 continued writes")


if __name__ == "__main__":
    verify()
