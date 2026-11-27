"""Verify conservative dataset-deletion reclamation with HDF5 1.14.6."""

from pathlib import Path
import subprocess
import tempfile

import h5py
import numpy as np

from verify_space_reuse import edit, sections


ROOT = Path(__file__).resolve().parents[1]


def verify() -> None:
    if h5py.version.hdf5_version != "1.14.6":
        raise SystemExit("deletion verification requires HDF5 1.14.6")
    with tempfile.TemporaryDirectory(prefix="moon-hdf5-delete-") as directory:
        temporary = Path(directory)
        for width in (4, 8):
            for chunked in (False, True):
                for references in (False, True):
                    source = temporary / f"source-{width}-{chunked}-{references}.h5"
                    output = temporary / f"output-{width}-{chunked}-{references}.h5"
                    fcpl = h5py.h5p.create(h5py.h5p.FILE_CREATE)
                    fcpl.set_sizes(width, 8)
                    fcpl.set_file_space_strategy(h5py.h5f.FSPACE_STRATEGY_FSM_AGGR, True, 1)
                    fcpl.set_userblock(512)
                    fapl = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
                    fapl.set_libver_bounds(h5py.h5f.LIBVER_LATEST, h5py.h5f.LIBVER_LATEST)
                    fid = h5py.h5f.create(bytes(source), h5py.h5f.ACC_TRUNC, fcpl=fcpl, fapl=fapl)
                    with h5py.File(fid) as handle:
                        edits = handle.create_group("edits")
                        target = handle.create_dataset("target", data=np.arange(1024, dtype="<i4"),
                                                       **({"chunks": (128,), "compression": "gzip"} if chunked else {}))
                        edits["alias"] = target
                        handle.create_dataset("keep", data=np.arange(64))
                        handle.create_dataset("hole", data=np.arange(16384, dtype="<i4"))
                        handle.create_dataset("tail", data=np.arange(16))
                        del handle["hole"]
                        if references:
                            refs = handle.create_dataset("refs", (1,), dtype=h5py.ref_dtype)
                            refs[0] = target.ref
                    with h5py.File(source, "r") as handle:
                        free = handle.id.get_freespace()
                        target = handle["target"]
                        if chunked:
                            regions = [(target.id.get_chunk_info(index).byte_offset - 512,
                                        target.id.get_chunk_info(index).size)
                                       for index in range(target.id.get_num_chunks())]
                        else:
                            regions = [(target.id.get_offset() - 512, target.nbytes)]
                    edit("delete", source, "/target", output)
                    with h5py.File(output, "r") as handle:
                        assert "target" not in handle and "alias" not in handle["edits"]
                        expected = free if references else free + sum(size for _, size in regions)
                        assert handle.id.get_freespace() == expected
                        free_regions = sections(handle)
                        if not references:
                            assert all(any(start <= address and address + size <= start + length
                                           for start, length in free_regions) for address, size in regions)
                        else:
                            np.testing.assert_array_equal(handle[handle["refs"][0]][:], np.arange(1024))
                    with h5py.File(output, "r+") as handle:
                        handle["edits"].create_dataset("new", data=np.arange(512))
                    with h5py.File(output, "r") as handle:
                        np.testing.assert_array_equal(handle["edits/new"][:], np.arange(512))
                        np.testing.assert_array_equal(handle["keep"][:], np.arange(64))
        source = temporary / "writer.h5"
        output = temporary / "writer-deleted.h5"
        subprocess.run(["moon", "run", "src/cmd/free_space_demo", "--target", "native",
                        "--", "create", "userblock", str(source)], cwd=ROOT, check=True)
        with h5py.File(source, "r") as handle:
            free = handle.id.get_freespace()
            size = handle["target"].nbytes
        edit("delete", source, "/target", output)
        with h5py.File(output, "r+") as handle:
            assert handle.id.get_freespace() == free + size
            handle.create_dataset("replacement", data=np.arange(512, dtype="<i4"))
        with h5py.File(output, "r") as handle:
            np.testing.assert_array_equal(handle["replacement"][:], np.arange(512))
            np.testing.assert_array_equal(handle["second"][:], np.full(512, 7))
    print("Verified dataset deletion reclamation, hard links and reference protection with HDF5 1.14.6")


if __name__ == "__main__":
    verify()
