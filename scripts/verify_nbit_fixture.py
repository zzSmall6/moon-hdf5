from pathlib import Path
import sys

import h5py
import numpy as np


def require_hdf5_1_14_6() -> None:
    if h5py.version.hdf5_version != "1.14.6":
        raise RuntimeError(
            "N-bit fixture verification requires HDF5 1.14.6, found "
            + h5py.version.hdf5_version
        )


def filter_ids(dataset) -> list[int]:
    properties = dataset.id.get_create_plist()
    return [
        properties.get_filter(index)[0]
        for index in range(properties.get_nfilters())
    ]


def verify(path: Path) -> None:
    require_hdf5_1_14_6()
    int16_values = np.arange(-42, 22, dtype=">i2").reshape(4, 4, 4)
    int64_values = ((np.arange(35, dtype="<i8") - 17) * 1000000007).reshape(
        5, 7
    )
    float_values = np.array(
        [
            [1.5, -2.25, 0.0],
            [-9.5, 123.25, -0.125],
            [7.0, -8.5, 4.75],
        ]
    )
    uint8_values = np.arange(80, dtype="u1").reshape(8, 10)

    compound = np.dtype(
        {
            "names": ["left", "right"],
            "formats": ["<i4", "<i4"],
            "offsets": [0, 8],
            "itemsize": 16,
        }
    )
    compound_values = np.zeros((5, 3), dtype=compound)
    compound_values["left"] = np.arange(15, dtype="<i4").reshape(5, 3) - 7
    compound_values["right"] = np.arange(15, dtype="<i4").reshape(5, 3) * 3 + 100

    inner = np.dtype(
        {
            "names": ["small", "scale"],
            "formats": [">i2", ">f4"],
            "offsets": [0, 4],
            "itemsize": 8,
        }
    )
    nested = np.dtype(
        {
            "names": ["id", "inner", "tail"],
            "formats": [">i4", inner, ">i2"],
            "offsets": [0, 8, 16],
            "itemsize": 24,
        }
    )
    nested_values = np.zeros((4,), dtype=nested)
    nested_values["id"] = [1, -2, 3, -4]
    nested_values["inner"]["small"] = [10, -20, 30, -40]
    nested_values["inner"]["scale"] = [0.5, -1.5, 2.25, -3.75]
    nested_values["tail"] = [7, -8, 9, -10]

    padded_compound = np.dtype(
        {
            "names": ["left", "right"],
            "formats": ["<i4", "<i2"],
            "offsets": [0, 8],
            "itemsize": 16,
        }
    )
    padded_compound_values = np.zeros((5,), dtype=padded_compound)
    padded_compound_values["left"] = [0, 1, -1, 2047, -2048]
    padded_compound_values["right"] = [7, -8, 9, -10, 11]

    padded_bits = np.dtype(
        {
            "names": ["b1", "b2", "b3", "b4"],
            "formats": ["<i1", ">i2", "<i4", ">i8"],
            "offsets": [0, 1, 4, 8],
            "itemsize": 16,
        }
    )
    padded_bits_values = np.zeros((12,), dtype=padded_bits)
    padded_bits_values["b1"] = [0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1]
    padded_bits_values["b2"] = [
        -512,
        -511,
        -256,
        -1,
        0,
        1,
        255,
        511,
        -300,
        300,
        -2,
        2,
    ]
    padded_bits_values["b3"] = [
        -65536,
        -65535,
        -32768,
        -1,
        0,
        1,
        32767,
        65535,
        -12345,
        12345,
        -2,
        2,
    ]
    padded_bits_values["b4"] = [
        -(2**62) + 1,
        -123456789,
        -1,
        0,
        1,
        123456789,
        2**62 - 1,
        -7,
        7,
        -100,
        100,
        -9,
    ]

    expected = {
        "int16": int16_values,
        "int64_deflate": int64_values,
        "float32_shuffle": float_values.astype("<f4"),
        "float64_be": float_values.astype(">f8"),
        "uint8_combined": uint8_values,
        "compound_noop": compound_values,
        "compound_nested_be": nested_values,
        "padded_int32": np.array([0, 1, -1, 2047, -2048, 300, -300], dtype="<i4"),
        "padded_int16_be": np.array(
            [0, 1, -1, 127, -128, 255, -256], dtype=">i2"
        ),
        "padded_compound": padded_compound_values,
        "padded_array": np.arange(24, dtype="<i4").reshape(4, 2, 3) - 5,
        "padded_bits": padded_bits_values,
        "enum_noop": np.array([[1, 2, 1], [2, 1, 2]], dtype="<i2"),
        "string_noop": np.array([b"alpha", b"", b"omega"], dtype="S8"),
    }
    partial_fill = np.full((5, 4), -7, dtype="<i4")
    partial_fill[0, 0] = 42
    partial_fill[2, 2] = 99
    expected["partial_fill"] = partial_fill

    pipelines = {
        "int16": [h5py.h5z.FILTER_NBIT],
        "int64_deflate": [h5py.h5z.FILTER_NBIT, h5py.h5z.FILTER_DEFLATE],
        "float32_shuffle": [h5py.h5z.FILTER_SHUFFLE, h5py.h5z.FILTER_NBIT],
        "float64_be": [h5py.h5z.FILTER_NBIT],
        "uint8_combined": [
            h5py.h5z.FILTER_SHUFFLE,
            h5py.h5z.FILTER_NBIT,
            h5py.h5z.FILTER_DEFLATE,
        ],
        "compound_noop": [h5py.h5z.FILTER_NBIT],
        "compound_nested_be": [h5py.h5z.FILTER_NBIT],
        "padded_int32": [h5py.h5z.FILTER_NBIT],
        "padded_int16_be": [h5py.h5z.FILTER_NBIT],
        "padded_compound": [h5py.h5z.FILTER_NBIT],
        "padded_array": [h5py.h5z.FILTER_NBIT],
        "padded_bits": [h5py.h5z.FILTER_NBIT],
        "enum_noop": [h5py.h5z.FILTER_NBIT],
        "string_noop": [h5py.h5z.FILTER_NBIT],
        "partial_fill": [h5py.h5z.FILTER_NBIT, h5py.h5z.FILTER_DEFLATE],
    }

    compressed_padded = (
        "padded_int32",
        "padded_int16_be",
        "padded_compound",
        "padded_array",
        "padded_bits",
    )
    identity_nbit = ("int16", "enum_noop", "string_noop")

    with h5py.File(path, "r") as handle:
        for name, values in expected.items():
            dataset = handle[name]
            np.testing.assert_array_equal(dataset[:], values)
            assert filter_ids(dataset) == pipelines[name]

        assert handle["padded_int32"].id.get_type().get_precision() == 12
        assert handle["padded_int32"].id.get_type().get_offset() == 4
        assert handle["padded_int16_be"].id.get_type().get_precision() == 9
        assert handle["padded_int16_be"].id.get_type().get_offset() == 3

        for name in compressed_padded:
            dataset = handle[name]
            info = dataset.id.get_chunk_info(0)
            raw_size = len(dataset.id.read_direct_chunk(info.chunk_offset)[1])
            uncompressed_size = int(np.prod(dataset.chunks)) * dataset.dtype.itemsize
            assert raw_size < uncompressed_size, (
                f"{name} N-bit chunk did not remove datatype padding"
            )

        for name in identity_nbit:
            dataset = handle[name]
            info = dataset.id.get_chunk_info(0)
            raw_size = len(dataset.id.read_direct_chunk(info.chunk_offset)[1])
            uncompressed_size = int(np.prod(dataset.chunks)) * dataset.dtype.itemsize
            assert raw_size == uncompressed_size, (
                f"{name} full-precision N-bit chunk should be unchanged"
            )

        enum_type = handle["enum_noop"].id.get_type()
        assert enum_type.get_nmembers() == 2
        assert enum_type.get_member_name(0) == b"LOW"
        assert enum_type.get_member_name(1) == b"HIGH"

    print(
        f"Verified {path.name} with h5py {h5py.__version__} "
        f"and HDF5 {h5py.version.hdf5_version}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_nbit_fixture.py <file.h5>")
    verify(Path(sys.argv[1]))
