"""Generate persistent file-space fixtures with HDF5 1.14.6."""

from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("fsm", h5py.h5f.FSPACE_STRATEGY_FSM_AGGR, True, 8, 8, 0),
    ("page", h5py.h5f.FSPACE_STRATEGY_PAGE, True, 8, 8, 0),
    ("transient", h5py.h5f.FSPACE_STRATEGY_FSM_AGGR, False, 8, 8, 0),
    ("aggr", h5py.h5f.FSPACE_STRATEGY_AGGR, False, 8, 8, 0),
    ("none", h5py.h5f.FSPACE_STRATEGY_NONE, False, 8, 8, 0),
    ("width4", h5py.h5f.FSPACE_STRATEGY_FSM_AGGR, True, 4, 8, 0),
    ("mixed", h5py.h5f.FSPACE_STRATEGY_PAGE, True, 4, 8, 0),
    ("userblock", h5py.h5f.FSPACE_STRATEGY_PAGE, True, 8, 8, 4096),
)


def generate() -> None:
    if h5py.version.hdf5_version != "1.14.6":
        raise SystemExit("free-space fixtures require HDF5 1.14.6")
    for name, strategy, persist, offsets, lengths, userblock in CASES:
        path = ROOT / "testdata" / f"free_space_{name}.h5"
        fcpl = h5py.h5p.create(h5py.h5p.FILE_CREATE)
        fcpl.set_sizes(offsets, lengths)
        fcpl.set_file_space_strategy(strategy, persist, 1)
        if userblock:
            fcpl.set_userblock(userblock)
        fapl = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
        fapl.set_libver_bounds(h5py.h5f.LIBVER_LATEST, h5py.h5f.LIBVER_LATEST)
        fid = h5py.h5f.create(bytes(path), h5py.h5f.ACC_TRUNC, fcpl=fcpl, fapl=fapl)
        with h5py.File(fid) as handle:
            for index in range(8):
                handle.create_dataset(
                    f"data_{index}",
                    data=np.arange(600 + 17 * index, dtype="<i4"),
                    track_times=False,
                )
            for index in (1, 3, 5):
                del handle[f"data_{index}"]
        with h5py.File(path, "r") as handle:
            print(path.name, handle.id.get_freespace(), path.stat().st_size)


if __name__ == "__main__":
    generate()
