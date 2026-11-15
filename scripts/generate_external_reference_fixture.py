"""Generate version-4 references that point into an external HDF5 file."""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np

from generate_v4_reference_fixture import configure, load_hdf5, NativeReferences


def generate(destination: Path) -> None:
    if h5py.version.hdf5_version != "1.14.6":
        raise SystemExit(
            "external_references.h5 requires HDF5 1.14.6, found "
            + h5py.version.hdf5_version
        )

    destination = destination.resolve()
    previous = Path.cwd()
    os.chdir(destination.parent)
    target_path = Path("external_reference_target.h5")
    source_path = destination.name
    try:
        with h5py.File(target_path, "w", libver="latest") as target:
            target.create_dataset(
                "target", data=np.arange(6, dtype="<i4"), track_times=False
            )
            group = target.create_group("group")
            group.create_dataset(
                "child", data=np.array([4, 5], dtype="<i4"), track_times=False
            )
            target["target"].attrs.create("unit", np.array(7, dtype="<i4"))

        library = load_hdf5()
        configure(library)
        native = NativeReferences(library)
        with h5py.File(source_path, "w", libver="latest") as source:
            source.create_dataset(
                "local_target",
                data=np.array([1, 2], dtype="<i4"),
                track_times=False,
            )
            with h5py.File(target_path, "r") as target:
                target_dataset = target["target"]
                group = target["group"]
                child = group["child"]

                objects = [
                    native.object(target.id.id, b"/target"),
                    native.object(target.id.id, b"/group"),
                    native.object(target.id.id, b"/group/child"),
                    native.object(source.id.id, b"/local_target"),
                    native.null(),
                ]
                native.write_dataset(source.id.id, b"object_refs", objects)

                regions = []
                space = native.copy_space(target_dataset.id.id)
                native.select_hyperslab(space, (1,), (2,), (2,), (1,))
                regions.append(native.region(target_dataset.id.id, space))
                native.close_space(space)

                space = native.copy_space(target_dataset.id.id)
                native.select_points(space, ((4,), (0,)))
                regions.append(native.region(target_dataset.id.id, space))
                native.close_space(space)

                space = native.copy_space(target_dataset.id.id)
                native.select_all(space)
                regions.append(native.region(target_dataset.id.id, space))
                native.close_space(space)

                space = native.copy_space(target_dataset.id.id)
                native.select_none(space)
                regions.append(native.region(target_dataset.id.id, space))
                native.close_space(space)
                regions.append(native.null())
                native.write_dataset(source.id.id, b"region_refs", regions)

                attributes = [
                    native.attribute(target_dataset.id.id, b"unit"),
                    native.null(),
                ]
                native.write_dataset(source.id.id, b"attribute_refs", attributes)
                source.flush()
                native.destroy(objects + regions + attributes)
                _ = group, child
    finally:
        os.chdir(previous)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 2:
        generate(Path(sys.argv[1]))
    elif len(sys.argv) == 1:
        generate(
            Path(__file__).resolve().parents[1]
            / "testdata"
            / "external_references.h5"
        )
    else:
        raise SystemExit(
            "usage: generate_external_reference_fixture.py [file.h5]"
        )
