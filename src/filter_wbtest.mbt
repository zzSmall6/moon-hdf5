///|
test "filter pipelines decode combinations edge chunks and skipped filters" {
  for path in ["testdata/filters_v1.h5", "testdata/filters_v2.h5"] {
    let file = Hdf5File::parse(@fs.read_file_to_bytes(path))
    for name in ["gzip", "shuffle", "fletcher", "combined"] {
      let values = integer_values(file.read_dataset("/" + name))
      assert_eq(values.length(), 35)
      for index in 0..<35 {
        assert_eq(values[index], index.to_int64())
      }
    }
    assert_eq(unsigned_values(file.read_dataset("/special/odd")), [
      0UL, 1UL, 2UL, 3UL, 4UL, 5UL, 6UL, 7UL, 8UL,
    ])
    assert_eq(integer_values(file.read_dataset("/special/partial")), [
      42L, -7L, -7L, -7L, -7L, -7L,
    ])
    assert_eq(integer_values(file.read_dataset("/special/skipped")), [
      9L, 8L, 7L, 6L,
    ])
    assert_eq(integer_values(file.read_dataset("/special/skip_deflate")), [
      9L, 8L, 7L, 6L,
    ])
    assert_eq(float_values(file.read_dataset("/special/floats")), [
      1.5, -2.25, 0.0, 123.5, 7.25,
    ])
    let error = try {
      file.read_dataset("/special/corrupt") |> ignore
      None
    } catch {
      error => Some(error)
    }
    assert_eq(error, Some(InvalidField("Fletcher32 checksum mismatch")))
  }
}

///|
test "unknown active filters are rejected but skipped filters are allowed" {
  let filters : Array[Filter] = [{ id: 32000, parameters: [], decoder: None, }]
  assert_eq(decode_filters(b"abcd", filters, 1U, 1, 4), b"abcd")
  let error = try {
    decode_filters(b"abcd", filters, 0U, 1, 4) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_true(error is Some(UnsupportedFeature(_)))
}

///|
test "deflate decoding bounds output and validates checksums" {
  let filters : Array[Filter] = [{ id: 1, parameters: [6U], decoder: None, }]
  let compressed = @zlib.zlib(b"abcdefgh"[:])
  assert_eq(decode_filters(compressed, filters, 0U, 1, 8), b"abcdefgh")
  let error = try {
    decode_filters(compressed, filters, 0U, 1, 1) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_true(error is Some(InvalidField(_)))
  let corrupted = compressed.to_array()
  corrupted[corrupted.length() - 1] = corrupted[corrupted.length() - 1] ^
    b'\x01'
  let checksum_error = try {
    decode_filters(Bytes::from_array(corrupted), filters, 0U, 1, 8) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_true(checksum_error is Some(InvalidField(_)))
}

///|
test "filter pipeline framing and masks reject invalid input" {
  for
    data in [
      b"\x01", b"\x02\x21", b"\x02\x01\x01\x00", b"\x02\x01\x01\x00\x04\x00\x00\x00",
    ] {
    let error = try {
      parse_filters({
        msg_type: msg_filter_pipeline,
        flags: b'\x00',
        data,
        data_offset: -1,
        chunk_start: -1,
        checksum_offset: -1,
      })
      |> ignore
      None
    } catch {
      error => Some(error)
    }
    assert_true(
      error is Some(InvalidField(_)) || error is Some(UnexpectedEof(_)),
    )
  }
  let error = try {
    decode_filters(b"abcd", [], 1U, 1, 4) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(error, Some(InvalidField("chunk filter mask exceeds pipeline")))
  let size_error = try {
    decode_filters(b"abc", [], 0U, 1, 4) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    size_error,
    Some(InvalidField("decoded chunk size does not match declared shape")),
  )
}

///|
test "shuffle can derive its element size from the filter pipeline" {
  let filters : Array[Filter] = [{ id: 2, parameters: [4U], decoder: None, }]
  let shuffled = b"aebfcgdh"
  assert_eq(decode_filters(shuffled, filters, 0U, 0, 8), b"abcdefgh")
  assert_eq(decode_filters(shuffled, filters, 0U, 4, 8), b"abcdefgh")
}

///|
test "N-bit chunks decode padded, array and compound datatypes" {
  let bytes = @fs.read_file_to_bytes("testdata/nbit_filters.h5")
  let file = Hdf5File::parse(bytes)
  assert_eq(integer_values(file.read_dataset("/padded_int32")), [
    0L, 1L, -1L, 2047L, -2048L, 300L, -300L,
  ])
  assert_eq(integer_values(file.read_dataset("/padded_int16_be")), [
    0L, 1L, -1L, 127L, -128L, 255L, -256L,
  ])
  let selected = file.read_dataset_selection(
    "/padded_int32",
    Hyperslab([2UL], [3UL], [2UL]),
  )
  match selected.values {
    Integers(values) => assert_eq(values, [-1L, -2048L, -300L])
    _ => abort("expected N-bit selection values")
  }
  let sources : Array[&Hdf5Source] = [bytes]
  let random = Hdf5File::parse_source(sources[0])
  assert_eq(integer_values(random.read_dataset("/padded_int32")), [
    0L, 1L, -1L, 2047L, -2048L, 300L, -300L,
  ])

  match file.read_dataset("/padded_compound").values {
    Structured(values) =>
      assert_eq(values, [
        Compound([
          { name: "left", value: SignedInteger(0L), },
          { name: "right", value: SignedInteger(7L), },
        ]),
        Compound([
          { name: "left", value: SignedInteger(1L), },
          { name: "right", value: SignedInteger(-8L), },
        ]),
        Compound([
          { name: "left", value: SignedInteger(-1L), },
          { name: "right", value: SignedInteger(9L), },
        ]),
        Compound([
          { name: "left", value: SignedInteger(2047L), },
          { name: "right", value: SignedInteger(-10L), },
        ]),
        Compound([
          { name: "left", value: SignedInteger(-2048L), },
          { name: "right", value: SignedInteger(11L), },
        ]),
      ])
    _ => abort("expected N-bit compound values")
  }
  match file.read_dataset("/padded_array").values {
    Structured(values) => {
      assert_eq(values.length(), 4)
      for element in 0..<4 {
        match values[element] {
          Elements(items) => {
            assert_eq(items.length(), 6)
            for item in 0..<6 {
              assert_eq(
                items[item],
                SignedInteger((element * 6 + item - 5).to_int64()),
              )
            }
          }
          _ => abort("expected N-bit array elements")
        }
      }
    }
    _ => abort("expected N-bit array dataset")
  }
}

///|
test "N-bit decodes non-byte-aligned bit fields across widths" {
  let file = Hdf5File::parse(@fs.read_file_to_bytes("testdata/nbit_filters.h5"))
  let expected = [
    [0L, -512L, -65536L, -4611686018427387903L],
    [-1L, -511L, -65535L, -123456789L],
    [0L, -256L, -32768L, -1L],
    [-1L, -1L, -1L, 0L],
    [0L, 0L, 0L, 1L],
    [-1L, 1L, 1L, 123456789L],
    [0L, 255L, 32767L, 4611686018427387903L],
    [-1L, 511L, 65535L, -7L],
    [0L, -300L, -12345L, 7L],
    [-1L, 300L, 12345L, -100L],
    [0L, -2L, -2L, 100L],
    [-1L, 2L, 2L, -9L],
  ]
  match file.read_dataset("/padded_bits").values {
    Structured(values) => {
      assert_eq(values.length(), 12)
      for row in 0..<12 {
        match values[row] {
          Compound(fields) => {
            assert_eq(fields.length(), 4)
            for column in 0..<4 {
              match fields[column].value {
                SignedInteger(value) => assert_eq(value, expected[row][column])
                _ => abort("expected signed N-bit field")
              }
            }
          }
          _ => abort("expected N-bit compound")
        }
      }
    }
    _ => abort("expected N-bit structured values")
  }
}

///|
test "N-bit pipelines preserve standard types and fill chunks" {
  let file = Hdf5File::parse(@fs.read_file_to_bytes("testdata/nbit_filters.h5"))
  let int16_expected = Array::make(64, 0L)
  for index in 0..<64 {
    int16_expected[index] = (-42 + index).to_int64()
  }
  assert_eq(integer_values(file.read_dataset("/int16")), int16_expected)
  let int64_expected = Array::make(35, 0L)
  for index in 0..<35 {
    int64_expected[index] = (index - 17).to_int64() * 1000000007L
  }
  assert_eq(integer_values(file.read_dataset("/int64_deflate")), int64_expected)
  let float32_expected = [
    1.5, -2.25, 0.0, -9.5, 123.25, -0.125, 7.0, -8.5, 4.75,
  ]
  assert_eq(
    float_values(file.read_dataset("/float32_shuffle")),
    float32_expected,
  )
  assert_eq(float_values(file.read_dataset("/float64_be")), float32_expected)
  let uint8_expected = Array::make(80, 0UL)
  for index in 0..<80 {
    uint8_expected[index] = index.to_uint64()
  }
  assert_eq(
    unsigned_values(file.read_dataset("/uint8_combined")),
    uint8_expected,
  )
  assert_eq(string_values(file.read_dataset("/string_noop")), [
    "alpha", "", "omega",
  ])
  let enum_values = match file.read_dataset("/enum_noop").values {
    Enums(values) => values
    _ => abort("expected N-bit enum values")
  }
  assert_eq(enum_values, ["LOW", "HIGH", "LOW", "HIGH", "LOW", "HIGH"])
  let partial_expected = Array::make(20, -7L)
  partial_expected[0] = 42L
  partial_expected[10] = 99L
  assert_eq(
    integer_values(file.read_dataset("/partial_fill")),
    partial_expected,
  )
}

///|
test "integer scale-offset chunks decode widths signs and byte orders" {
  let file = Hdf5File::parse(@fs.read_file_to_bytes("testdata/scaleoffset.h5"))
  let int32_auto = Array::make(20, -7L)
  for index in 0..<10 {
    int32_auto[index] = (-50 + index).to_int64()
  }
  assert_eq(integer_values(file.read_dataset("/int32_auto")), int32_auto)
  let selected = file.read_dataset_selection(
    "/int32_auto",
    Hyperslab([8UL], [5UL], [1UL]),
  )
  match selected.values {
    Integers(values) => assert_eq(values, [-42L, -41L, -7L, -7L, -7L])
    _ => abort("expected scale-offset integer selection")
  }
  let fixed : Array[Int64] = []
  for index in 0..<16 {
    fixed.push((100 + index).to_int64())
  }
  assert_eq(integer_values(file.read_dataset("/int32_fixed")), fixed)
  assert_eq(integer_values(file.read_dataset("/int32_mixed_fill")), [
    10L, -7L, 12L, 13L,
  ])
  assert_eq(integer_values(file.read_dataset("/int32_all_fill")), [
    -7L, -7L, -7L, -7L,
  ])
  assert_eq(integer_values(file.read_dataset("/int32_full")), [
    -5L, -4L, -3L, -2L,
  ])
  assert_eq(integer_values(file.read_dataset("/int16_be")), [
    -300L, -299L, -250L, -200L, -199L, -100L,
  ])
  assert_eq(integer_values(file.read_dataset("/int8")), [-50L, -7L, 0L, 50L])
  assert_eq(unsigned_values(file.read_dataset("/uint16_be")), [
    10000UL, 10001UL, 15000UL, 20000UL,
  ])
  assert_eq(unsigned_values(file.read_dataset("/uint8")), [
    0UL, 1UL, 2UL, 3UL, 4UL, 5UL, 6UL,
  ])
  assert_eq(unsigned_values(file.read_dataset("/uint32")), [
    1000000UL, 1000001UL, 2000000UL, 3000000UL,
  ])
  assert_eq(integer_values(file.read_dataset("/int64")), [
    -(1L << 40),
    -(1L << 40) + 1L,
    0L,
    1L << 40,
  ])
  assert_eq(unsigned_values(file.read_dataset("/uint64")), [
    1UL << 40,
    (1UL << 40) + 5UL,
    (1UL << 40) + 9UL,
    (1UL << 40) + 12UL,
  ])
}

///|
test "floating scale-offset chunks decode f32 f64 byte orders and pipelines" {
  let file = Hdf5File::parse(@fs.read_file_to_bytes("testdata/scaleoffset.h5"))
  let f32_values = [1.25, 2.5, -1.75, 9.5, -9.5, 0.0]
  assert_eq(float_values(file.read_dataset("/float32_dscale")), f32_values)
  let selected = file.read_dataset_selection(
    "/float32_dscale",
    Hyperslab([1UL], [3UL], [1UL]),
  )
  match selected.values {
    Floats(values) => assert_eq(values, [2.5, -1.75, 9.5])
    _ => abort("expected floating scale-offset selection")
  }
  assert_eq(float_values(file.read_dataset("/float64_dscale")), [
    1.5, -2.5, 3.5, -9.5, 0.0,
  ])
  assert_eq(float_values(file.read_dataset("/float32_be")), f32_values)
  assert_eq(float_values(file.read_dataset("/float32_pipeline")), [
    10.25, 11.5, 12.75, -9.5, 0.0,
  ])
  assert_eq(float_values(file.read_dataset("/float32_all_fill")), [
    -9.5, -9.5, -9.5, -9.5,
  ])
}

///|
test "malformed scale-offset headers are rejected" {
  let parameters : Array[UInt32] = [
    2U, 0U, 1U, 0U, 4U, 1U, 0U, 1U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U,
  ]
  let filter : Filter = { id: 6, parameters, decoder: None, }
  for data in [b"", b"\x00\x00\x00\x00\x08"] {
    let error = try {
      decode_scaleoffset(data, filter, 4, 4) |> ignore
      None
    } catch {
      error => Some(error)
    }
    assert_true(error is Some(InvalidField(_)))
  }
  let float_parameters : Array[UInt32] = [
    1U, 0U, 1U, 1U, 4U, 0U, 0U, 1U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U,
  ]
  let float_error = try {
    decode_scaleoffset(
      b"\x00\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
      { id: 6, parameters: float_parameters, decoder: None, },
      4,
      4,
    )
    |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_true(float_error is Some(UnsupportedFeature(_)))
}
