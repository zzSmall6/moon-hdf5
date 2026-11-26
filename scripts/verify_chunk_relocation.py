"""Check relocated chunks and subsequent writes with HDF5 1.14.6."""

from pathlib import Path
import subprocess
import tempfile

import h5py
import numpy as np

from verify_space_reuse import edit, sections


ROOT = Path(__file__).resolve().parents[1]


def verify_pair(source: Path, output: Path, dataset: str) -> None:
    with h5py.File(source, "r") as handle:
        original = handle[dataset]
        values = original[:]
        chunks = [original.id.get_chunk_info(index)
                  for index in range(original.id.get_num_chunks())]
        raw = [original.id.read_direct_chunk(chunk.chunk_offset) for chunk in chunks]
        free = handle.id.get_freespace()
        node = h5py.h5o.get_info(original.id).addr
        maxshape = original.maxshape
    edit("relocate", source, dataset, output)
    with h5py.File(output, "r") as handle:
        moved = handle[dataset]
        np.testing.assert_array_equal(moved[:], values)
        assert h5py.h5o.get_info(moved.id).addr == node
        assert moved.maxshape == maxshape
        assert moved.id.get_num_chunks() == len(chunks)
        assert handle.id.get_freespace() == free
        free_regions = sections(handle)
        base = handle.userblock_size
        for before, payload in zip(chunks, raw):
            after = moved.id.get_chunk_info_by_coord(before.chunk_offset)
            assert after.byte_offset != before.byte_offset
            assert after.byte_offset % 8 == 0
            assert after.size == before.size
            actual_mask, actual_bytes = moved.id.read_direct_chunk(before.chunk_offset)
            assert actual_bytes == payload[1], (dataset, "raw bytes")
            if len(chunks) > 1:
                assert actual_mask == payload[0], (dataset, "mask")
            assert any(start <= before.byte_offset - base and
                       before.byte_offset - base + before.size <= start + size
                       for start, size in free_regions)
    with h5py.File(output, "r+") as handle:
        moved = handle[dataset]
        moved[...] = values
        handle["edits"].create_dataset("new", data=np.arange(512, dtype="<i4"))
    with h5py.File(output, "r") as handle:
        np.testing.assert_array_equal(handle[dataset][:], values)
        np.testing.assert_array_equal(handle["edits/new"][:], np.arange(512))


def verify() -> None:
    if h5py.version.hdf5_version != "1.14.6":
        raise SystemExit("chunk relocation verification requires HDF5 1.14.6")
    with tempfile.TemporaryDirectory(prefix="moon-hdf5-chunks-") as directory:
        temporary = Path(directory)
        for mode in ("plain", "userblock"):
            source = temporary / f"{mode}.h5"
            subprocess.run(
                ["moon", "run", "src/cmd/free_space_demo", "--target", "native",
                 "--", "create-chunks", mode, str(source)], cwd=ROOT, check=True,
            )
            for index in ("single", "fixed", "array", "tree"):
                for filtering in ("raw", "filtered"):
                    dataset = f"{index}_{filtering}"
                    verify_pair(source, temporary / f"{mode}-{dataset}.h5", dataset)
            verify_pair(source, temporary / f"{mode}-names.h5", "names")
        for widths in ((8, 8), (4, 8)):
            source = temporary / f"legacy-{widths[0]}.h5"
            fcpl = h5py.h5p.create(h5py.h5p.FILE_CREATE)
            fcpl.set_sizes(*widths)
            fcpl.set_file_space_strategy(h5py.h5f.FSPACE_STRATEGY_FSM_AGGR, True, 1)
            fid = h5py.h5f.create(bytes(source), h5py.h5f.ACC_TRUNC, fcpl=fcpl)
            with h5py.File(fid) as handle:
                handle.create_group("edits")
                handle.create_dataset("hole", data=np.arange(16384, dtype="<i4"))
                handle.create_dataset("chunks", data=np.arange(1200).reshape(30, 40),
                                      chunks=(3, 4), compression="gzip")
                sparse = handle.create_dataset("sparse", (30, 40), dtype="<i4",
                                               chunks=(3, 4), fillvalue=-1)
                sparse[0:3, 0:4] = 42
                sparse[27:30, 36:40] = 7
                handle.create_dataset("keep", data=np.arange(16))
                del handle["hole"]
            for dataset in ("chunks", "sparse"):
                verify_pair(source, temporary / f"legacy-{widths[0]}-{dataset}.h5", dataset)
    print("Verified chunk relocation, raw bytes, masks and HDF5 1.14.6 continued writes")


if __name__ == "__main__":
    verify()
