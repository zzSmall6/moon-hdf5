"""Generate virtual-dataset fixtures with HDF5 1.14.6."""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
from h5py import h5d, h5p, h5s, h5t


ROOT = Path(__file__).resolve().parents[1]
TESTDATA = ROOT / "testdata"


def create_four_byte_file(path: Path, mode: str) -> h5py.File:
    fcpl = h5py.h5p.create(h5py.h5p.FILE_CREATE)
    fcpl.set_sizes(4, 8)
    fapl = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    fapl.set_libver_bounds(h5py.h5f.LIBVER_LATEST, h5py.h5f.LIBVER_LATEST)
    fid = h5py.h5f.create(bytes(path), h5py.h5f.ACC_TRUNC, fcpl=fcpl, fapl=fapl)
    handle = h5py.File(fid)
    _ = mode
    return handle


def generate(destination: Path) -> None:
    if h5py.version.hdf5_version != "1.14.6":
        raise SystemExit(
            "virtual_dataset.h5 requires HDF5 1.14.6, found "
            + h5py.version.hdf5_version
        )

    destination = destination.resolve()
    previous = Path.cwd()
    os.chdir(destination.parent)
    main_name = destination.name
    nested_name = "virtual_dataset_nested.h5"
    source_name = "virtual_dataset_source.h5"
    source_32_name = "virtual_dataset_source_32.h5"
    main_32_name = "virtual_dataset_32.h5"
    printf_name = "virtual_dataset_printf.h5"
    printf_source_names = (
        "virtual_printf_source_0.h5",
        "virtual_printf_source_1.h5",
    )
    conversion_name = "virtual_dataset_conversion.h5"
    try:
        for name in (
            main_name,
            main_32_name,
            nested_name,
            printf_name,
            conversion_name,
            *printf_source_names,
            source_name,
            source_32_name,
        ):
            path = Path(name)
            if path.exists():
                path.unlink()

        matrix = np.arange(100, 136, dtype="<i4").reshape(6, 6)
        with h5py.File(source_name, "w", libver="latest") as source:
            source.create_dataset(
                "data",
                data=np.arange(20, dtype="<i4").reshape(4, 5),
                track_times=False,
            )
            source.create_dataset("matrix", data=matrix, track_times=False)
            source.create_dataset(
                "vector",
                data=np.arange(6, dtype="<i4"),
                track_times=False,
            )

        source_32_values = np.arange(200, 212, dtype="<i4").reshape(3, 4)
        with create_four_byte_file(Path(source_32_name), "w") as source:
            source.create_dataset(
                "data", data=source_32_values, track_times=False
            )

        with h5py.File(main_name, "w", libver="latest") as handle:
            handle.create_dataset(
                "same_source",
                data=np.arange(12, dtype="<i4").reshape(3, 4),
                track_times=False,
            )

            mapped = h5py.VirtualLayout(shape=(8, 10), dtype="<i4")
            external = h5py.VirtualSource(source_name, "/data", shape=(4, 5))
            mapped[0:4, 0:5] = external
            mapped[4:8, 5:10] = external
            handle.create_virtual_dataset("mapped", mapped, fillvalue=-1)

            same = h5py.VirtualLayout(shape=(3, 4), dtype="<i4")
            same_source = h5py.VirtualSource(".", "/same_source", shape=(3, 4))
            same[...] = same_source
            handle.create_virtual_dataset("same_file", same)

            strided = h5py.VirtualLayout(shape=(3, 3), dtype="<i4")
            strided_source = h5py.VirtualSource(
                source_name, "/matrix", shape=(6, 6)
            )
            strided[:, :] = strided_source[::2, ::2]
            handle.create_virtual_dataset("strided", strided, fillvalue=-7)

            multiple = h5py.VirtualLayout(shape=(6,), dtype="<i4")
            vector = h5py.VirtualSource(source_name, "/vector", shape=(6,))
            multiple[0:3] = vector[0:3]
            multiple[3:6] = vector[3:6]
            handle.create_virtual_dataset("multiple_mappings", multiple, fillvalue=-1)

        with create_four_byte_file(Path(main_32_name), "w") as handle:
            layout = h5py.VirtualLayout(shape=(4, 6), dtype="<i4")
            external = h5py.VirtualSource(
                source_32_name, "/data", shape=source_32_values.shape
            )
            layout[0:3, 0:4] = external
            handle.create_virtual_dataset("mapped", layout, fillvalue=-9)

        with h5py.File(nested_name, "w", libver="latest") as nested:
            layout = h5py.VirtualLayout(shape=(4, 5), dtype="<i4")
            source = h5py.VirtualSource(main_name, "/mapped", shape=(8, 10))
            layout[0:4, 0:5] = source[4:8, 5:10]
            nested.create_virtual_dataset("nested", layout, fillvalue=-3)

        create_printf_fixture(Path(printf_name), printf_source_names)
        create_conversion_fixture(Path(conversion_name))

        verify(
            Path(main_name),
            Path(main_32_name),
            Path(nested_name),
            Path(printf_name),
            Path(conversion_name),
        )
    finally:
        os.chdir(previous)

    print(
        "Generated virtual dataset fixtures with h5py "
        f"{h5py.__version__}/HDF5 {h5py.version.hdf5_version}"
    )


def create_printf_fixture(
    path: Path, source_names: tuple[str, str]
) -> None:
    """Create printf mappings for file names, dataset names and literal %."""
    source_values = (
        np.arange(8, dtype="<i2").reshape(2, 4),
        np.arange(8, dtype="<i2").reshape(2, 4) + 100,
    )
    for name, values in zip(source_names, source_values):
        with h5py.File(name, "w", libver="latest") as source:
            source.create_dataset("data", data=values, track_times=False)

    with h5py.File(path, "w", libver="latest") as handle:
        handle.create_dataset(
            "data%name", data=np.arange(4, dtype="<i2"), track_times=False
        )
        handle.create_dataset(
            "block_0", data=np.array([10, 11], dtype="<i4"), track_times=False
        )
        handle.create_dataset(
            "block_1", data=np.array([20, 21], dtype="<i4"), track_times=False
        )

        grid_space = h5s.create_simple((4, 4), (h5s.UNLIMITED, 4))
        grid_space.select_hyperslab(
            (0, 0), (h5s.UNLIMITED, 1), (2, 1), (2, 4)
        )
        source_space = h5s.create_simple((2, 4))
        create_list = h5p.create(h5p.DATASET_CREATE)
        create_list.set_virtual(
            grid_space,
            b"virtual_printf_source_%b.h5",
            b"/data",
            source_space,
        )
        h5d.create(
            handle.id,
            b"printf_grid",
            h5t.py_create(np.dtype("<i8")),
            grid_space,
            dcpl=create_list,
        )

        percent_layout = h5py.VirtualLayout(shape=(4,), dtype="<i2")
        percent_layout[...] = h5py.VirtualSource(
            ".", "/data%%name", shape=(4,), dtype="<i2"
        )
        handle.create_virtual_dataset("literal_percent", percent_layout)

        dataset_space = h5s.create_simple((4,), (h5s.UNLIMITED,))
        dataset_space.select_hyperslab(
            (0,), (h5s.UNLIMITED,), (2,), (2,)
        )
        dataset_source_space = h5s.create_simple((2,))
        dataset_create_list = h5p.create(h5p.DATASET_CREATE)
        dataset_create_list.set_virtual(
            dataset_space,
            b".",
            b"/block_%b",
            dataset_source_space,
        )
        h5d.create(
            handle.id,
            b"printf_dataset",
            h5t.py_create(np.dtype("<i4")),
            dataset_space,
            dcpl=dataset_create_list,
        )


def create_conversion_fixture(path: Path) -> None:
    """Create same-file mappings whose source and target datatypes differ."""
    with h5py.File(path, "w", libver="latest") as handle:
        int16_source = handle.create_dataset(
            "src_i16",
            data=np.array([[1, -2, 3], [4, 5, -6]], dtype="<i2"),
            track_times=False,
        )
        float32_source = handle.create_dataset(
            "src_f32",
            data=np.array(
                [[1.9, -2.9, 3.5], [-4.1, 5.9, -6.2]], dtype="<f4"
            ),
            track_times=False,
        )
        string_source = handle.create_dataset(
            "src_s3",
            data=np.array([b"ab", b"c", b"def"], dtype="S3"),
            track_times=False,
        )
        enum_source = handle.create_dataset(
            "src_enum",
            data=np.array(
                [1, 2],
                dtype=h5py.enum_dtype({"A": 1, "B": 2}, basetype="<i2"),
            ),
            track_times=False,
        )
        enum_integer_source = handle.create_dataset(
            "src_enum_integer",
            data=np.array([1, 2], dtype="<i2"),
            track_times=False,
        )
        target_enum = h5py.enum_dtype({"ONE": 1, "TWO": 2}, basetype="<i4")
        compound_source = handle.create_dataset(
            "src_compound",
            data=np.array(
                [(1, 2), (3, 4)],
                dtype=[("a", "<i2"), ("b", "<i2")],
            ),
            track_times=False,
        )
        compound_a_source = handle.create_dataset(
            "src_compound_a",
            data=np.array([(1,), (3,)], dtype=[("a", "<i2")]),
            track_times=False,
        )
        nested_source = handle.create_dataset(
            "src_nested_compound",
            data=np.array(
                [((1, 2.5),), ((3, 4.5),)],
                dtype=[("p", [("x", "<i2"), ("y", "<f4")])],
            ),
            track_times=False,
        )
        compound_string_source = handle.create_dataset(
            "src_compound_string",
            data=np.array(
                [(b"ab", 1), (b"c", 2)],
                dtype=[("s", "S3"), ("n", "<i2")],
            ),
            track_times=False,
        )
        compound_array_source = handle.create_dataset(
            "src_compound_array",
            data=np.array(
                [([1, 2],), ([3, 4],)],
                dtype=[("q", "<i2", (2,))],
            ),
            track_times=False,
        )
        string_grid_source = handle.create_dataset(
            "src_s2_grid",
            data=np.array(
                [[b"a", b"b", b"c"], [b"d", b"e", b"f"], [b"g", b"h", b"i"]],
                dtype="S2",
            ),
            track_times=False,
        )
        complex_source = handle.create_dataset(
            "src_complex64",
            data=np.array([1 + 2j, 3 + 4j], dtype="<c8"),
            track_times=False,
        )
        vlen_utf8 = h5py.string_dtype(encoding="utf-8")
        for name, source, dtype in (
            ("i16_i64", int16_source, "<i8"),
            ("i16_f64", int16_source, "<f8"),
            ("f32_i32", float32_source, "<i4"),
            ("s3_s8", string_source, "S8"),
            ("enum_to_int", enum_source, "<i4"),
            ("int_to_enum", enum_integer_source, target_enum),
            ("enum_to_enum", enum_source, target_enum),
            ("cmp_i16_i32", compound_source, [("a", "<i4"), ("b", "<i4")]),
            (
                "cmp_missing",
                compound_a_source,
                [("a", "<i4"), ("b", "<i4")],
            ),
            ("cmp_extra", compound_source, [("a", "<i4")]),
            ("cmp_reorder", compound_source, [("b", "<i4"), ("a", "<i4")]),
            (
                "cmp_nested",
                nested_source,
                [("p", [("x", "<i4"), ("y", "<f8")])],
            ),
            (
                "cmp_string",
                compound_string_source,
                [("s", "S8"), ("n", "<i4")],
            ),
            (
                "cmp_array",
                compound_array_source,
                [("q", "<i4", (2,))],
            ),
            ("s3_vlen_utf8", string_source, vlen_utf8),
            ("complex64_128", complex_source, "<c16"),
        ):
            layout = h5py.VirtualLayout(shape=source.shape, dtype=dtype)
            layout[...] = h5py.VirtualSource(source)
            handle.create_virtual_dataset(name, layout)
        strided_vlen = h5py.VirtualLayout(shape=(2, 2), dtype=vlen_utf8)
        strided_vlen[:, :] = h5py.VirtualSource(string_grid_source)[::2, ::2]
        handle.create_virtual_dataset("s3_vlen_strided", strided_vlen)
        filled_vlen = h5py.VirtualLayout(shape=(4,), dtype=vlen_utf8)
        filled_vlen[0:3] = h5py.VirtualSource(string_source)
        handle.create_virtual_dataset("s3_vlen_fill", filled_vlen)


def verify(
    main: Path,
    main_32: Path,
    nested: Path,
    printf: Path,
    conversion: Path,
) -> None:
    expected_mapped = np.full((8, 10), -1, dtype="<i4")
    values = np.arange(20, dtype="<i4").reshape(4, 5)
    expected_mapped[0:4, 0:5] = values
    expected_mapped[4:8, 5:10] = values
    with h5py.File(main, "r") as handle:
        np.testing.assert_array_equal(handle["mapped"][...], expected_mapped)
        np.testing.assert_array_equal(
            handle["same_file"][...], np.arange(12, dtype="<i4").reshape(3, 4)
        )
        np.testing.assert_array_equal(
            handle["strided"][...], np.arange(100, 136, dtype="<i4").reshape(6, 6)[
                ::2, ::2
            ]
        )
        np.testing.assert_array_equal(
            handle["multiple_mappings"][...], np.arange(6, dtype="<i4")
        )
        for name in ("mapped", "same_file", "strided", "multiple_mappings"):
            assert handle[name].id.get_create_plist().get_layout() == 3

    expected_32 = np.full((4, 6), -9, dtype="<i4")
    expected_32[0:3, 0:4] = np.arange(200, 212, dtype="<i4").reshape(3, 4)
    with h5py.File(main_32, "r") as handle:
        np.testing.assert_array_equal(handle["mapped"][...], expected_32)
        assert handle["mapped"].id.get_create_plist().get_layout() == 3

    with h5py.File(nested, "r") as handle:
        np.testing.assert_array_equal(
            handle["nested"][...], np.arange(20, dtype="<i4").reshape(4, 5)
        )
        assert handle["nested"].id.get_create_plist().get_layout() == 3

    expected_printf = np.array(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [100, 101, 102, 103],
            [104, 105, 106, 107],
        ],
        dtype="<i8",
    )
    with h5py.File(printf, "r") as handle:
        np.testing.assert_array_equal(handle["printf_grid"][...], expected_printf)
        np.testing.assert_array_equal(
            handle["literal_percent"][...], np.arange(4, dtype="<i2")
        )
        np.testing.assert_array_equal(
            handle["printf_dataset"][...], np.array([10, 11, 20, 21], dtype="<i4")
        )
        assert handle["printf_grid"].maxshape == (None, 4)
        assert handle["printf_dataset"].maxshape == (None,)
        for name in ("printf_grid", "literal_percent", "printf_dataset"):
            assert handle[name].is_virtual

    with h5py.File(conversion, "r") as handle:
        np.testing.assert_array_equal(
            handle["i16_i64"][...],
            np.array([[1, -2, 3], [4, 5, -6]], dtype="<i8"),
        )
        np.testing.assert_array_equal(
            handle["i16_f64"][...],
            np.array([[1, -2, 3], [4, 5, -6]], dtype="<f8"),
        )
        np.testing.assert_array_equal(
            handle["f32_i32"][...],
            np.array([[1, -2, 3], [-4, 5, -6]], dtype="<i4"),
        )
        np.testing.assert_array_equal(
            handle["s3_s8"][...], np.array([b"ab", b"c", b"def"], dtype="S8")
        )
        np.testing.assert_array_equal(
            handle["enum_to_int"][...], np.array([1, 2], dtype="<i4")
        )
        np.testing.assert_array_equal(
            handle["int_to_enum"][...], np.array([1, 2], dtype="<i4")
        )
        assert handle["int_to_enum"].dtype.metadata == {"enum": {"ONE": 1, "TWO": 2}}
        np.testing.assert_array_equal(
            handle["enum_to_enum"][...], np.array([1, 2], dtype="<i4")
        )
        assert handle["enum_to_enum"].dtype.metadata == {
            "enum": {"ONE": 1, "TWO": 2}
        }
        np.testing.assert_array_equal(
            handle["cmp_i16_i32"][...],
            np.array([(1, 2), (3, 4)], dtype=[("a", "<i4"), ("b", "<i4")]),
        )
        np.testing.assert_array_equal(
            handle["cmp_missing"][...],
            np.array([(1, 0), (3, 0)], dtype=[("a", "<i4"), ("b", "<i4")]),
        )
        np.testing.assert_array_equal(
            handle["cmp_extra"][...], np.array([(1,), (3,)], dtype=[("a", "<i4")])
        )
        np.testing.assert_array_equal(
            handle["cmp_reorder"][...],
            np.array([(2, 1), (4, 3)], dtype=[("b", "<i4"), ("a", "<i4")]),
        )
        np.testing.assert_array_equal(
            handle["cmp_nested"][...],
            np.array(
                [((1, 2.5),), ((3, 4.5),)],
                dtype=[("p", [("x", "<i4"), ("y", "<f8")])],
            ),
        )
        np.testing.assert_array_equal(
            handle["cmp_string"][...],
            np.array(
                [(b"ab", 1), (b"c", 2)],
                dtype=[("s", "S8"), ("n", "<i4")],
            ),
        )
        np.testing.assert_array_equal(
            handle["cmp_array"][...],
            np.array([([1, 2],), ([3, 4],)], dtype=[("q", "<i4", (2,))]),
        )
        np.testing.assert_array_equal(
            handle["s3_vlen_utf8"][...],
            np.array([b"ab", b"c", b"def"], dtype=object),
        )
        np.testing.assert_array_equal(
            handle["s3_vlen_strided"][...],
            np.array([[b"a", b"c"], [b"g", b"i"]], dtype=object),
        )
        np.testing.assert_array_equal(
            handle["s3_vlen_fill"][...],
            np.array([b"ab", b"c", b"def", b""], dtype=object),
        )
        np.testing.assert_array_equal(
            handle["complex64_128"][...], np.array([1 + 2j, 3 + 4j], dtype="<c16")
        )
        for name in (
            "i16_i64",
            "i16_f64",
            "f32_i32",
            "s3_s8",
            "enum_to_int",
            "int_to_enum",
            "enum_to_enum",
            "cmp_i16_i32",
            "cmp_missing",
            "cmp_extra",
            "cmp_reorder",
            "cmp_nested",
            "cmp_string",
            "cmp_array",
            "s3_vlen_utf8",
            "s3_vlen_strided",
            "s3_vlen_fill",
            "complex64_128",
        ):
            assert handle[name].is_virtual


if __name__ == "__main__":
    import sys

    target = (
        Path(sys.argv[1])
        if len(sys.argv) == 2
        else TESTDATA / "virtual_dataset.h5"
    )
    if len(sys.argv) > 2:
        raise SystemExit("usage: generate_virtual_dataset_fixture.py [file.h5]")
    generate(target)
