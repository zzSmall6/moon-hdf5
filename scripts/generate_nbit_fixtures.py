from pathlib import Path

import h5py
import numpy as np


def create_nbit_dataset(
    handle,
    name,
    values,
    *,
    chunks,
    filters=(),
    fill_value=None,
):
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_chunk(tuple(chunks))
    if fill_value is not None:
        dcpl.set_fill_value(np.asarray(fill_value, dtype=values.dtype))
    for filter_name in filters:
        if filter_name == "nbit":
            dcpl.set_filter(h5py.h5z.FILTER_NBIT, 0, ())
        elif filter_name == "shuffle":
            dcpl.set_shuffle()
        elif filter_name == "deflate":
            dcpl.set_deflate(6)
        else:
            raise ValueError(f"unknown filter {filter_name}")
    space = h5py.h5s.create_simple(values.shape)
    datatype = h5py.h5t.py_create(values.dtype)
    dataset_id = h5py.h5d.create(
        handle.id, name.encode(), datatype, space, dcpl=dcpl
    )
    dataset = h5py.Dataset(dataset_id)
    dataset[...] = values
    return dataset


def padded_integer(size, precision, offset, order):
    dtype = np.dtype((">i" if order == "be" else "<i") + str(size))
    datatype = h5py.h5t.py_create(dtype)
    datatype.set_precision(precision)
    datatype.set_offset(offset)
    return datatype


def create_typed_dataset(
    handle, name, values, datatype, memory_dtype, *, chunks, shape=None
):
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_chunk(tuple(chunks))
    dcpl.set_filter(h5py.h5z.FILTER_NBIT, 0, ())
    space = h5py.h5s.create_simple(values.shape if shape is None else tuple(shape))
    dataset_id = h5py.h5d.create(handle.id, name.encode(), datatype, space, dcpl=dcpl)
    dataset = h5py.Dataset(dataset_id)
    dataset.id.write(
        h5py.h5s.ALL,
        h5py.h5s.ALL,
        values,
        mtype=h5py.h5t.py_create(memory_dtype),
    )
    return dataset


def generate(destination: Path) -> None:
    expected = {}
    with h5py.File(destination, "w", libver=("earliest", "v108")) as handle:
        values = np.arange(-42, 22, dtype=">i2").reshape(4, 4, 4)
        create_nbit_dataset(
            handle, "int16", values, chunks=(2, 3, 2), filters=("nbit",)
        )
        expected["int16"] = values

        values = (np.arange(35, dtype="<i8") - 17) * 1000000007
        values = values.reshape(5, 7)
        create_nbit_dataset(
            handle,
            "int64_deflate",
            values,
            chunks=(3, 4),
            filters=("nbit", "deflate"),
        )
        expected["int64_deflate"] = values

        values = np.array(
            [
                [1.5, -2.25, 0.0],
                [-9.5, 123.25, -0.125],
                [7.0, -8.5, 4.75],
            ],
            dtype="<f4",
        )
        create_nbit_dataset(
            handle,
            "float32_shuffle",
            values,
            chunks=(2, 2),
            filters=("shuffle", "nbit"),
        )
        expected["float32_shuffle"] = values

        values = np.array(
            [
                [1.5, -2.25, 0.0],
                [-9.5, 123.25, -0.125],
                [7.0, -8.5, 4.75],
            ],
            dtype=">f8",
        )
        create_nbit_dataset(
            handle, "float64_be", values, chunks=(2, 3), filters=("nbit",)
        )
        expected["float64_be"] = values

        values = np.arange(80, dtype="u1").reshape(8, 10)
        create_nbit_dataset(
            handle,
            "uint8_combined",
            values,
            chunks=(3, 4),
            filters=("shuffle", "nbit", "deflate"),
        )
        expected["uint8_combined"] = values

        compound = np.dtype(
            {
                "names": ["left", "right"],
                "formats": ["<i4", "<i4"],
                "offsets": [0, 8],
                "itemsize": 16,
            }
        )
        values = np.zeros((5, 3), dtype=compound)
        values["left"] = np.arange(15, dtype="<i4").reshape(5, 3) - 7
        values["right"] = (np.arange(15, dtype="<i4").reshape(5, 3) * 3) + 100
        create_nbit_dataset(
            handle, "compound_noop", values, chunks=(2, 2), filters=("nbit",)
        )
        expected["compound_noop"] = values

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
        values = np.zeros((4,), dtype=nested)
        values["id"] = [1, -2, 3, -4]
        values["inner"]["small"] = [10, -20, 30, -40]
        values["inner"]["scale"] = [0.5, -1.5, 2.25, -3.75]
        values["tail"] = [7, -8, 9, -10]
        create_nbit_dataset(
            handle, "compound_nested_be", values, chunks=(3,), filters=("nbit",)
        )
        expected["compound_nested_be"] = values

        padded32 = padded_integer(4, 12, 4, "le")
        values = np.array([0, 1, -1, 2047, -2048, 300, -300], dtype="<i4")
        create_typed_dataset(
            handle,
            "padded_int32",
            values,
            padded32,
            np.dtype("<i4"),
            chunks=(3,),
        )
        expected["padded_int32"] = values

        padded16 = padded_integer(2, 9, 3, "be")
        values = np.array([0, 1, -1, 127, -128, 255, -256], dtype=">i2")
        create_typed_dataset(
            handle,
            "padded_int16_be",
            values,
            padded16,
            np.dtype(">i2"),
            chunks=(4,),
        )
        expected["padded_int16_be"] = values

        right = h5py.h5t.py_create(np.dtype("<i2"))
        compound_type = h5py.h5t.create(h5py.h5t.COMPOUND, 16)
        compound_type.insert(b"left", 0, padded32)
        compound_type.insert(b"right", 8, right)
        compound_memory = np.dtype(
            {
                "names": ["left", "right"],
                "formats": ["<i4", "<i2"],
                "offsets": [0, 8],
                "itemsize": 16,
            }
        )
        values = np.zeros((5,), dtype=compound_memory)
        values["left"] = [0, 1, -1, 2047, -2048]
        values["right"] = [7, -8, 9, -10, 11]
        create_typed_dataset(
            handle,
            "padded_compound",
            values,
            compound_type,
            compound_memory,
            chunks=(3,),
        )
        expected["padded_compound"] = values

        array_type = h5py.h5t.array_create(padded32, (2, 3))
        array_memory = np.dtype(("<i4", (2, 3)))
        values = np.arange(24, dtype="<i4").reshape(4, 2, 3) - 5
        create_typed_dataset(
            handle,
            "padded_array",
            values,
            array_type,
            array_memory,
            chunks=(3,),
            shape=(4,),
        )
        expected["padded_array"] = values

        padded_bits = h5py.h5t.create(h5py.h5t.COMPOUND, 16)
        padded_bits.insert(b"b1", 0, padded_integer(1, 1, 0, "le"))
        padded_bits.insert(b"b2", 1, padded_integer(2, 10, 5, "be"))
        padded_bits.insert(b"b3", 4, padded_integer(4, 17, 7, "le"))
        padded_bits.insert(b"b4", 8, padded_integer(8, 63, 1, "be"))
        padded_bits_memory = np.dtype(
            {
                "names": ["b1", "b2", "b3", "b4"],
                "formats": ["<i1", ">i2", "<i4", ">i8"],
                "offsets": [0, 1, 4, 8],
                "itemsize": 16,
            }
        )
        values = np.zeros((12,), dtype=padded_bits_memory)
        values["b1"] = [0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1]
        values["b2"] = [-512, -511, -256, -1, 0, 1, 255, 511, -300, 300, -2, 2]
        values["b3"] = [
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
        values["b4"] = [
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
        create_typed_dataset(
            handle,
            "padded_bits",
            values,
            padded_bits,
            padded_bits_memory,
            chunks=(5,),
        )
        expected["padded_bits"] = values

        enum_base = h5py.h5t.py_create(np.dtype("<i2"))
        enum_type = h5py.h5t.enum_create(enum_base)
        enum_type.enum_insert(b"LOW", 1)
        enum_type.enum_insert(b"HIGH", 2)
        values = np.array([[1, 2, 1], [2, 1, 2]], dtype="<i2")
        create_typed_dataset(
            handle,
            "enum_noop",
            values,
            enum_type,
            np.dtype("<i2"),
            chunks=(1, 2),
        )
        expected["enum_noop"] = values

        values = np.array([b"alpha", b"", b"omega"], dtype="S8")
        create_nbit_dataset(
            handle, "string_noop", values, chunks=(2,), filters=("nbit",)
        )
        expected["string_noop"] = values

        values = np.full((5, 4), -7, dtype="<i4")
        values[0, 0] = 42
        values[2, 2] = 99
        create_nbit_dataset(
            handle,
            "partial_fill",
            values,
            chunks=(2, 2),
            filters=("nbit", "deflate"),
            fill_value=-7,
        )
        expected["partial_fill"] = values

    with h5py.File(destination, "r") as handle:
        for name, values in expected.items():
            np.testing.assert_array_equal(handle[name][:], values)
            create = handle[name].id.get_create_plist()
            assert create.get_nfilters() >= 1
            assert any(create.get_filter(i)[0] == h5py.h5z.FILTER_NBIT
                       for i in range(create.get_nfilters()))
        for name in (
            "padded_int32",
            "padded_int16_be",
            "padded_compound",
            "padded_array",
            "padded_bits",
        ):
            dataset = handle[name]
            _, raw = dataset.id.read_direct_chunk((0,))
            element_size = dataset.dtype.itemsize
            chunk_elements = int(np.prod(dataset.chunks))
            assert len(raw) < chunk_elements * element_size, (
                f"{name} N-bit did not remove datatype padding"
            )
    print(f"Verified {destination} with h5py {h5py.__version__}")


if __name__ == "__main__":
    directory = Path(__file__).resolve().parents[1] / "testdata"
    generate(directory / "nbit_filters.h5")
