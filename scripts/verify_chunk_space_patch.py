"""Verify automatic chunk free-space accounting and continued HDF5 writes."""

from pathlib import Path
import subprocess
import tempfile

import h5py
import numpy as np

from verify_space_reuse import edit, sections


ROOT = Path(__file__).resolve().parents[1]


def noise(count: int) -> np.ndarray:
    state = 123456789
    values = []
    for _index in range(count):
        state = (state ^ (state << 13)) & 0xFFFFFFFF
        state ^= state >> 17
        state = (state ^ (state << 5)) & 0xFFFFFFFF
        values.append(state if state < 0x80000000 else state - 0x100000000)
    return np.array(values, dtype="<i4")


def stored(dataset: h5py.Dataset) -> int:
    return sum(dataset.id.get_chunk_info(index).size
               for index in range(dataset.id.get_num_chunks()))


def exercise(source: Path, output: Path, dataset: str) -> None:
    with h5py.File(source, "r") as handle:
        target = handle[dataset]
        initial_free = handle.id.get_freespace()
        initial_stored = stored(target)
        shape = target.shape
    current = source
    for operation in ("patch-noise", "patch", "patch-noise", "patch"):
        edit(operation, current, dataset, output)
        with h5py.File(output, "r") as handle:
            target = handle[dataset]
            expected = noise(target.size).reshape(shape) if operation == "patch-noise" else np.full(shape, 42)
            np.testing.assert_array_equal(target[:], expected)
            assert handle.id.get_freespace() == initial_free + initial_stored - stored(target)
            assert sum(size for _, size in sections(handle)) == handle.id.get_freespace()
        current = output
    with h5py.File(output, "r+") as handle:
        handle[dataset][...] = np.full(shape, 42)
        handle["edits"].create_dataset("new", data=np.arange(256))
    with h5py.File(output, "r") as handle:
        np.testing.assert_array_equal(handle[dataset][:], np.full(shape, 42))
        np.testing.assert_array_equal(handle["edits/new"][:], np.arange(256))


def verify() -> None:
    if h5py.version.hdf5_version != "1.14.6":
        raise SystemExit("chunk patch verification requires HDF5 1.14.6")
    with tempfile.TemporaryDirectory(prefix="moon-hdf5-chunk-patch-") as directory:
        temporary = Path(directory)
        for mode in ("plain", "userblock"):
            source = temporary / f"{mode}.h5"
            subprocess.run(["moon", "run", "src/cmd/free_space_demo", "--target", "native",
                            "--", "create-chunks", mode, str(source)], cwd=ROOT, check=True)
            for index in ("single", "fixed", "array", "tree"):
                dataset = f"{index}_filtered"
                exercise(source, temporary / f"{mode}-{dataset}.h5", dataset)
        for width in (4, 8):
            source = temporary / f"legacy-{width}.h5"
            fcpl = h5py.h5p.create(h5py.h5p.FILE_CREATE)
            fcpl.set_sizes(width, 8)
            fcpl.set_file_space_strategy(h5py.h5f.FSPACE_STRATEGY_FSM_AGGR, True, 1)
            fid = h5py.h5f.create(bytes(source), h5py.h5f.ACC_TRUNC, fcpl=fcpl)
            with h5py.File(fid) as handle:
                handle.create_group("edits")
                handle.create_dataset("hole", data=np.arange(65536, dtype="<i4"))
                handle.create_dataset("chunks", data=np.ones((32, 32), dtype="<i4"),
                                      chunks=(8, 8), compression="gzip")
                handle.create_dataset("keep", data=np.arange(16))
                del handle["hole"]
            exercise(source, temporary / f"legacy-{width}-patched.h5", "chunks")
    print("Verified automatic chunk reuse, shrink reclamation and HDF5 1.14.6 continued writes")


if __name__ == "__main__":
    verify()
