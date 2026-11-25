"""Compare MoonBit's persisted sections with HDF5's public section API."""

import ctypes
from pathlib import Path
import subprocess

import h5py


ROOT = Path(__file__).resolve().parents[1]


class Section(ctypes.Structure):
    _fields_ = [("addr", ctypes.c_uint64), ("size", ctypes.c_uint64)]


def verify() -> None:
    if h5py.version.hdf5_version != "1.14.6":
        raise SystemExit("free-space verification requires HDF5 1.14.6")
    library = ctypes.CDLL(h5py.h5f.__file__)
    get_sections = library.H5Fget_free_sections
    get_sections.argtypes = [ctypes.c_int64, ctypes.c_int, ctypes.c_size_t,
                            ctypes.POINTER(Section)]
    get_sections.restype = ctypes.c_ssize_t
    paths = sorted((ROOT / "testdata").glob("free_space_*.h5"))
    assert len(paths) == 8
    result = subprocess.run(
        ["moon", "run", "src/cmd/free_space_demo", "--target", "native", "--",
         *map(str, paths)], cwd=ROOT, check=True, capture_output=True, text=True,
    )
    actual = {}
    for line in result.stdout.splitlines():
        if line.startswith("file "):
            _, name, total = line.split()
            current = (int(total), [])
            actual[name] = current
        else:
            current[1].append(tuple(map(int, line.split())))
    for path in paths:
        with h5py.File(path, "r") as handle:
            count = get_sections(handle.id.id, 0, 0, None)
            assert count >= 0
            sections = (Section * max(count, 1))()
            assert get_sections(handle.id.id, 0, max(count, 1), sections) == count
            expected = sorted((item.addr, item.size) for item in sections[:count])
            total, found = actual[str(path)]
            assert total == handle.id.get_freespace(), path.name
            assert sorted(found) == expected, path.name
    print("Verified persistent free-space sections against HDF5 1.14.6")


if __name__ == "__main__":
    verify()
