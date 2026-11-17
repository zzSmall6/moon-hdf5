///|
priv struct Filter {
  id : Int32
  parameters : Array[UInt32]
  decoder : &Hdf5CustomFilter?
}

///|
fn parse_filters(
  message : HeaderMessage,
  custom_filters~ : Array[&Hdf5CustomFilter] = [],
) -> Array[Filter] raise Hdf5Error {
  guard (message.flags.to_int() & 2) == 0 else {
    raise InvalidField("shared filter pipeline message was not resolved")
  }
  let reader = Reader::new(message.data)
  let version = reader.read_u8().to_int()
  guard version == 1 || version == 2 else {
    raise UnsupportedFeature("unsupported filter pipeline version")
  }
  let count = reader.read_u8().to_int()
  guard count <= 32 else { raise InvalidField("too many filters") }
  if version == 1 {
    reader.skip(6)
  }
  let filters : Array[Filter] = []
  for _ in 0..<count {
    let id = reader.read_u16_le().to_int()
    let name_size = if version == 1 || id >= 256 {
      reader.read_u16_le().to_int()
    } else {
      0
    }
    let flags = reader.read_u16_le().to_int()
    guard (flags & 0xFFFE) == 0 else {
      raise InvalidField("invalid filter flags")
    }
    let parameter_count = reader.read_u16_le().to_int()
    reader.skip(name_size)
    let parameters : Array[UInt32] = []
    for _ in 0..<parameter_count {
      parameters.push(reader.read_u32_le())
    }
    if version == 1 && parameter_count % 2 != 0 {
      reader.skip(4)
    }
    let decoder = match custom_filters.search_by(filter => filter.id() == id) {
      Some(index) => Some(custom_filters[index])
      None => None
    }
    filters.push({ id, parameters, decoder, })
  }
  filters
}

///|
fn strip_fletcher32(data : Bytes) -> Bytes raise Hdf5Error {
  guard data.length() >= 4 else {
    raise InvalidField("truncated Fletcher32 chunk")
  }
  let size = data.length() - 4
  let mut first = 0U
  let mut second = 0U
  let mut offset = 0
  while offset < size {
    let high = data[offset].to_uint() << 8
    let low = if offset + 1 < size { data[offset + 1].to_uint() } else { 0U }
    first = first + high + low
    first = (first & 0xFFFFU) + (first >> 16)
    second = second + first
    second = (second & 0xFFFFU) + (second >> 16)
    offset = offset + 2
  }
  first = (first & 0xFFFFU) + (first >> 16)
  second = (second & 0xFFFFU) + (second >> 16)
  let checksum = (second << 16) | first
  let legacy = ((checksum & 0x00FF00FFU) << 8) | ((checksum & 0xFF00FF00U) >> 8)
  let stored = Reader::at(data, size).read_u32_le()
  guard stored == checksum || stored == legacy else {
    raise InvalidField("Fletcher32 checksum mismatch")
  }
  data.get_view(start=0, end=size).unwrap().to_owned()
}

///|
priv struct NbitField {
  size : Int32
  order : Int32
  precision : Int32
  offset : Int32
}

///|
priv enum NbitNode {
  NbitAtomic(NbitField)
  NbitArray(Int32, Int32)
  NbitCompound(Int32, Array[(Int32, Int32)])
  NbitNoop(Int32)
}

///|
priv struct NbitCursor {
  input : Bytes
  mut bit : Int64
}

///|
fn nbit_read_parameter(
  parameters : Array[UInt32],
  cursor : FixedArray[Int32],
) -> Int32 raise Hdf5Error {
  guard cursor[0] >= 0 && cursor[0] < parameters.length() else {
    raise InvalidField("truncated N-bit filter parameters")
  }
  let value = parameters[cursor[0]]
  cursor[0] = cursor[0] + 1
  guard value <= 0x7FFFFFFFU else {
    raise UnsupportedFeature("N-bit filter parameter is too large")
  }
  value.reinterpret_as_int()
}

///|
fn nbit_node_size(
  nodes : Array[NbitNode],
  index : Int32,
) -> Int32 raise Hdf5Error {
  guard index >= 0 && index < nodes.length() else {
    raise InvalidField("invalid N-bit datatype node")
  }
  match nodes[index] {
    NbitAtomic(field) => field.size
    NbitArray(size, _) => size
    NbitCompound(size, _) => size
    NbitNoop(size) => size
  }
}

///|
fn nbit_parse_node(
  parameters : Array[UInt32],
  cursor : FixedArray[Int32],
  nodes : Array[NbitNode],
  datatype_class : Int32,
  depth : Int32,
) -> Int32 raise Hdf5Error {
  guard depth <= 16 else {
    raise UnsupportedFeature("N-bit datatype nesting is too deep")
  }
  match datatype_class {
    1 => {
      let size = nbit_read_parameter(parameters, cursor)
      let order = nbit_read_parameter(parameters, cursor)
      let encoded_precision = nbit_read_parameter(parameters, cursor)
      let offset = nbit_read_parameter(parameters, cursor)
      let precision = if encoded_precision == 0 {
        size * 8
      } else {
        encoded_precision
      }
      guard size > 0 &&
        size <= 0x0FFFFFFF &&
        (order == 0 || order == 1) &&
        precision > 0 &&
        precision.to_int64() <= size.to_int64() * 8L &&
        offset >= 0 &&
        offset.to_int64() + precision.to_int64() <= size.to_int64() * 8L else {
        raise InvalidField("invalid N-bit atomic datatype")
      }
      nodes.push(NbitAtomic({ size, order, precision, offset, }))
    }
    2 => {
      let size = nbit_read_parameter(parameters, cursor)
      let base_class = nbit_read_parameter(parameters, cursor)
      let base = nbit_parse_node(
        parameters,
        cursor,
        nodes,
        base_class,
        depth + 1,
      )
      let base_size = nbit_node_size(nodes, base)
      guard size > 0 && base_size > 0 && size % base_size == 0 else {
        raise InvalidField("invalid N-bit array datatype")
      }
      nodes.push(NbitArray(size, base))
    }
    3 => {
      let size = nbit_read_parameter(parameters, cursor)
      let member_count = nbit_read_parameter(parameters, cursor)
      guard size > 0 && member_count >= 0 && member_count <= 0xFFF else {
        raise InvalidField("invalid N-bit compound datatype")
      }
      let members : Array[(Int32, Int32)] = []
      let mut used_size = 0
      for _ in 0..<member_count {
        let member_offset = nbit_read_parameter(parameters, cursor)
        let member_class = nbit_read_parameter(parameters, cursor)
        let child = nbit_parse_node(
          parameters,
          cursor,
          nodes,
          member_class,
          depth + 1,
        )
        let member_size = nbit_node_size(nodes, child)
        guard member_offset >= 0 &&
          member_size > 0 &&
          used_size <= size - member_size &&
          member_offset <= size - member_size else {
          raise InvalidField("invalid N-bit compound member")
        }
        used_size = used_size + member_size
        members.push((member_offset, child))
      }
      nodes.push(NbitCompound(size, members))
    }
    4 => {
      let size = nbit_read_parameter(parameters, cursor)
      guard size > 0 else { raise InvalidField("invalid N-bit no-op datatype") }
      nodes.push(NbitNoop(size))
    }
    _ => raise InvalidField("invalid N-bit datatype class")
  }
  nodes.length() - 1
}

///|
fn nbit_bit_mask(count : Int32) -> Int32 {
  if count == 0 {
    0
  } else {
    (1 << count) - 1
  }
}

///|
fn NbitCursor::read_bits(
  self : NbitCursor,
  count : Int32,
) -> Int32 raise Hdf5Error {
  guard count >= 0 &&
    count <= 8 &&
    self.bit >= 0L &&
    self.bit + count.to_int64() <= self.input.length().to_int64() * 8L else {
    raise InvalidField("truncated N-bit chunk")
  }
  let byte_index = (self.bit / 8L).to_int()
  let bit_offset = (self.bit % 8L).to_int()
  let available = 8 - bit_offset
  let value = if count <= available {
    (self.input[byte_index].to_int() >> (available - count)) &
    nbit_bit_mask(count)
  } else {
    let first = available
    let second = count - first
    ((self.input[byte_index].to_int() & nbit_bit_mask(first)) << second) |
    (
      (self.input[byte_index + 1].to_int() >> (8 - second)) &
      nbit_bit_mask(second)
    )
  }
  self.bit = self.bit + count.to_int64()
  value
}

///|
fn nbit_decompress_atomic(
  output : FixedArray[Byte],
  data_offset : Int32,
  cursor : NbitCursor,
  field : NbitField,
) -> Unit raise Hdf5Error {
  guard data_offset >= 0 &&
    data_offset <= output.length() - field.size &&
    field.size > 0 else {
    raise InvalidField("N-bit atomic value exceeds its chunk")
  }
  let datatype_len = field.size.to_int64() * 8L
  let begin_i : Int32 = if field.order == 0 {
    let sum = field.precision.to_int64() + field.offset.to_int64()
    if sum % 8L != 0L {
      (sum / 8L).to_int()
    } else {
      (sum / 8L - 1L).to_int()
    }
  } else {
    ((datatype_len - field.precision.to_int64() - field.offset.to_int64()) / 8L).to_int()
  }
  let end_i : Int32 = if field.order == 0 {
    field.offset / 8
  } else if field.offset % 8 != 0 {
    ((datatype_len - field.offset.to_int64()) / 8L).to_int()
  } else {
    ((datatype_len - field.offset.to_int64()) / 8L - 1L).to_int()
  }
  guard begin_i >= 0 && end_i >= 0 && begin_i < field.size && end_i < field.size else {
    raise InvalidField("invalid N-bit atomic bit range")
  }
  let mut k = begin_i
  let mut finished = false
  while !finished {
    let (dat_len, dat_offset) = if begin_i != end_i {
      if k == begin_i {
        let tail = (
            datatype_len - field.precision.to_int64() - field.offset.to_int64()
          ) %
          8L
        (8 - tail.to_int(), 0)
      } else if k == end_i {
        let length = 8 - field.offset % 8
        (length, 8 - length)
      } else {
        (8, 0)
      }
    } else {
      (field.precision, field.offset % 8)
    }
    guard dat_len >= 0 &&
      dat_len <= 8 &&
      dat_offset >= 0 &&
      dat_offset + dat_len <= 8 else {
      raise InvalidField("invalid N-bit atomic byte width")
    }
    let value = cursor.read_bits(dat_len)
    output[data_offset + k] = (value << dat_offset).to_byte()
    if field.order == 0 {
      if k == end_i {
        finished = true
      } else {
        k = k - 1
      }
    } else if k == end_i {
      finished = true
    } else {
      k = k + 1
    }
  }
}

///|
fn nbit_decompress_node(
  nodes : Array[NbitNode],
  index : Int32,
  output : FixedArray[Byte],
  data_offset : Int32,
  cursor : NbitCursor,
) -> Unit raise Hdf5Error {
  guard index >= 0 && index < nodes.length() else {
    raise InvalidField("invalid N-bit datatype node")
  }
  match nodes[index] {
    NbitAtomic(field) =>
      nbit_decompress_atomic(output, data_offset, cursor, field)
    NbitArray(size, base) => {
      let base_size = nbit_node_size(nodes, base)
      guard base_size > 0 && size > 0 && size % base_size == 0 else {
        raise InvalidField("invalid N-bit array datatype")
      }
      for element in 0..<(size / base_size) {
        nbit_decompress_node(
          nodes,
          base,
          output,
          data_offset + element * base_size,
          cursor,
        )
      }
    }
    NbitCompound(_, members) =>
      for (member_offset, child) in members {
        nbit_decompress_node(
          nodes,
          child,
          output,
          data_offset + member_offset,
          cursor,
        )
      }
    NbitNoop(size) => {
      guard data_offset >= 0 && data_offset <= output.length() - size else {
        raise InvalidField("N-bit no-op value exceeds its chunk")
      }
      for byte_index in 0..<size {
        output[data_offset + byte_index] = cursor.read_bits(8).to_byte()
      }
    }
  }
}

///|
fn decode_nbit(
  data : Bytes,
  parameters : Array[UInt32],
  expected_size : Int32,
) -> Bytes raise Hdf5Error {
  guard parameters.length() >= 3 &&
    parameters[0].reinterpret_as_int() == parameters.length() else {
    raise InvalidField("invalid N-bit filter parameter count")
  }
  guard parameters[1] <= 1U else {
    raise InvalidField("invalid N-bit compression flag")
  }
  if parameters[1] == 1U {
    guard data.length() == expected_size else {
      raise InvalidField("uncompressed N-bit chunk size mismatch")
    }
    return data
  }
  let cursor = FixedArray::make(1, 0)
  cursor[0] = 2
  let element_count = nbit_read_parameter(parameters, cursor)
  let datatype_class = nbit_read_parameter(parameters, cursor)
  let nodes : Array[NbitNode] = []
  let root = nbit_parse_node(parameters, cursor, nodes, datatype_class, 0)
  guard cursor[0] == parameters.length() else {
    raise InvalidField("N-bit filter parameter has trailing values")
  }
  let element_size = nbit_node_size(nodes, root)
  guard element_count > 0 &&
    element_size > 0 &&
    element_count <= 0x7FFFFFFF / element_size &&
    element_count * element_size == expected_size else {
    raise InvalidField("N-bit output size does not match declared shape")
  }
  let output = FixedArray::make(expected_size, b'\x00')
  let bit_cursor : NbitCursor = { input: data, bit: 0L, }
  for element in 0..<element_count {
    nbit_decompress_node(
      nodes,
      root,
      output,
      element * element_size,
      bit_cursor,
    )
  }
  guard bit_cursor.bit <= data.length().to_int64() * 8L &&
    data.length().to_int64() * 8L - bit_cursor.bit <= 8L else {
    raise InvalidField("N-bit chunk has an invalid payload length")
  }
  output.unsafe_reinterpret_as_bytes()
}

///|
priv struct NbitWriteCursor {
  output : FixedArray[Byte]
  mut bit : Int64
}

///|
fn NbitWriteCursor::write_bits(
  self : NbitWriteCursor,
  value : Int32,
  count : Int32,
) -> Unit raise Hdf5Error {
  guard count >= 0 &&
    count <= 8 &&
    value >= 0 &&
    self.bit >= 0L &&
    self.bit + count.to_int64() <= self.output.length().to_int64() * 8L else {
    raise InvalidField("invalid N-bit output bit range")
  }
  let mut consumed = 0
  while consumed < count {
    let byte_index = (self.bit / 8L).to_int()
    let bit_offset = (self.bit % 8L).to_int()
    let available = 8 - bit_offset
    let take = if count - consumed < available {
      count - consumed
    } else {
      available
    }
    let shift = count - consumed - take
    let part = (value >> shift) & nbit_bit_mask(take)
    self.output[byte_index] = (self.output[byte_index].to_int() |
    (part << (available - take))).to_byte()
    self.bit = self.bit + take.to_int64()
    consumed = consumed + take
  }
}

///|
fn nbit_node_bit_count(
  nodes : Array[NbitNode],
  index : Int32,
) -> Int64 raise Hdf5Error {
  guard index >= 0 && index < nodes.length() else {
    raise InvalidField("invalid N-bit datatype node")
  }
  match nodes[index] {
    NbitAtomic(field) => field.precision.to_int64()
    NbitArray(size, base) => {
      let base_size = nbit_node_size(nodes, base)
      guard base_size > 0 && size > 0 && size % base_size == 0 else {
        raise InvalidField("invalid N-bit array datatype")
      }
      let count = (size / base_size).to_int64()
      let base_bits = nbit_node_bit_count(nodes, base)
      guard count <= 0x7FFFFFFFFFFFFFFFL / base_bits else {
        raise InvalidField("N-bit datatype is too large")
      }
      count * base_bits
    }
    NbitCompound(_, members) => {
      let mut bits = 0L
      for (_, child) in members {
        let child_bits = nbit_node_bit_count(nodes, child)
        guard bits <= 0x7FFFFFFFFFFFFFFFL - child_bits else {
          raise InvalidField("N-bit datatype is too large")
        }
        bits = bits + child_bits
      }
      bits
    }
    NbitNoop(size) => size.to_int64() * 8L
  }
}

///|
fn nbit_compress_atomic(
  data : Bytes,
  data_offset : Int32,
  cursor : NbitWriteCursor,
  field : NbitField,
) -> Unit raise Hdf5Error {
  guard data_offset >= 0 &&
    data_offset <= data.length() - field.size &&
    field.size > 0 else {
    raise InvalidField("N-bit atomic value exceeds its chunk")
  }
  let datatype_len = field.size.to_int64() * 8L
  let begin_i : Int32 = if field.order == 0 {
    let sum = field.precision.to_int64() + field.offset.to_int64()
    if sum % 8L != 0L {
      (sum / 8L).to_int()
    } else {
      (sum / 8L - 1L).to_int()
    }
  } else {
    ((datatype_len - field.precision.to_int64() - field.offset.to_int64()) / 8L).to_int()
  }
  let end_i : Int32 = if field.order == 0 {
    field.offset / 8
  } else if field.offset % 8 != 0 {
    ((datatype_len - field.offset.to_int64()) / 8L).to_int()
  } else {
    ((datatype_len - field.offset.to_int64()) / 8L - 1L).to_int()
  }
  guard begin_i >= 0 && end_i >= 0 && begin_i < field.size && end_i < field.size else {
    raise InvalidField("invalid N-bit atomic bit range")
  }
  let mut k = begin_i
  let mut finished = false
  while !finished {
    let (dat_len, value) = if begin_i != end_i {
      if k == begin_i {
        let tail = (
            datatype_len - field.precision.to_int64() - field.offset.to_int64()
          ) %
          8L
        let width = (8L - tail).to_int()
        (width, data[data_offset + k].to_int() & nbit_bit_mask(width))
      } else if k == end_i {
        let width = 8 - field.offset % 8
        (width, data[data_offset + k].to_int() >> (8 - width))
      } else {
        (8, data[data_offset + k].to_int())
      }
    } else {
      (
        field.precision,
        (data[data_offset + k].to_int() >> (field.offset % 8)) &
        nbit_bit_mask(field.precision),
      )
    }
    guard dat_len >= 0 && dat_len <= 8 else {
      raise InvalidField("invalid N-bit atomic byte width")
    }
    cursor.write_bits(value, dat_len)
    if field.order == 0 {
      if k == end_i {
        finished = true
      } else {
        k = k - 1
      }
    } else if k == end_i {
      finished = true
    } else {
      k = k + 1
    }
  }
}

///|
fn nbit_compress_node(
  nodes : Array[NbitNode],
  index : Int32,
  data : Bytes,
  data_offset : Int32,
  cursor : NbitWriteCursor,
) -> Unit raise Hdf5Error {
  guard index >= 0 && index < nodes.length() else {
    raise InvalidField("invalid N-bit datatype node")
  }
  match nodes[index] {
    NbitAtomic(field) => nbit_compress_atomic(data, data_offset, cursor, field)
    NbitArray(size, base) => {
      let base_size = nbit_node_size(nodes, base)
      guard base_size > 0 && size > 0 && size % base_size == 0 else {
        raise InvalidField("invalid N-bit array datatype")
      }
      for element in 0..<(size / base_size) {
        nbit_compress_node(
          nodes,
          base,
          data,
          data_offset + element * base_size,
          cursor,
        )
      }
    }
    NbitCompound(_, members) =>
      for (member_offset, child) in members {
        nbit_compress_node(
          nodes,
          child,
          data,
          data_offset + member_offset,
          cursor,
        )
      }
    NbitNoop(size) => {
      guard data_offset >= 0 && data_offset <= data.length() - size else {
        raise InvalidField("N-bit no-op value exceeds its chunk")
      }
      for byte_index in 0..<size {
        cursor.write_bits(data[data_offset + byte_index].to_int(), 8)
      }
    }
  }
}

///|
fn encode_nbit(
  data : Bytes,
  parameters : Array[UInt32],
  expected_size : Int32,
) -> Bytes raise Hdf5Error {
  guard data.length() == expected_size &&
    parameters.length() >= 3 &&
    parameters[0].reinterpret_as_int() == parameters.length() else {
    raise InvalidField("invalid N-bit filter parameter count")
  }
  guard parameters[1] <= 1U else {
    raise InvalidField("invalid N-bit compression flag")
  }
  if parameters[1] == 1U {
    return data
  }
  let cursor = FixedArray::make(1, 0)
  cursor[0] = 2
  let element_count = nbit_read_parameter(parameters, cursor)
  let datatype_class = nbit_read_parameter(parameters, cursor)
  let nodes : Array[NbitNode] = []
  let root = nbit_parse_node(parameters, cursor, nodes, datatype_class, 0)
  guard cursor[0] == parameters.length() else {
    raise InvalidField("N-bit filter parameter has trailing values")
  }
  let element_size = nbit_node_size(nodes, root)
  guard element_count > 0 &&
    element_size > 0 &&
    element_count <= 0x7FFFFFFF / element_size &&
    element_count * element_size == expected_size else {
    raise InvalidField("N-bit input size does not match declared shape")
  }
  let element_bits = nbit_node_bit_count(nodes, root)
  guard element_count.to_int64() <= 0x7FFFFFFFFFFFFFFFL / element_bits else {
    raise InvalidField("N-bit chunk is too large")
  }
  let total_bits = element_count.to_int64() * element_bits
  let output_size = total_bits / 8L + 1L
  guard output_size > 0L && output_size <= 0x7FFFFFFFL else {
    raise InvalidField("N-bit chunk is too large")
  }
  let output = FixedArray::make(output_size.to_int(), b'\x00')
  let bit_cursor : NbitWriteCursor = { output, bit: 0L, }
  for element in 0..<element_count {
    nbit_compress_node(nodes, root, data, element * element_size, bit_cursor)
  }
  guard bit_cursor.bit == total_bits else {
    raise InvalidField("N-bit encoder produced an unexpected bit length")
  }
  output.unsafe_reinterpret_as_bytes()
}

///|
fn scaleoffset_parameter_value(
  parameters : Array[UInt32],
  size : Int32,
) -> UInt64 {
  let mut value = 0UL
  for index in 0..<size {
    let word = parameters[8 + index / 4].to_uint64()
    let byte = (word >> (index % 4 * 8)) & 0xFFUL
    value = value | (byte << (index * 8))
  }
  value
}

///|
fn scaleoffset_write_scalar(
  output : FixedArray[Byte],
  offset : Int32,
  size : Int32,
  order : Int32,
  value : UInt64,
) -> Unit {
  for byte_index in 0..<size {
    let shift = if order == 0 {
      byte_index * 8
    } else {
      (size - 1 - byte_index) * 8
    }
    output[offset + byte_index] = ((value >> shift) & 0xFFUL).to_byte()
  }
}

///|
fn scaleoffset_float_factor32(digits : Int32) -> Float {
  @math.powf(Float(10.0), Float::from_int(digits))
}

///|
fn scaleoffset_float_factor64(digits : Int32) -> Double {
  @math.pow(10.0, Double::from_int(digits))
}

///|
fn decode_scaleoffset(
  data : Bytes,
  filter : Filter,
  element_size : Int32,
  expected_size : Int32,
) -> Bytes raise Hdf5Error {
  guard element_size > 0 &&
    expected_size >= 0 &&
    expected_size % element_size == 0 else {
    raise InvalidField("invalid scale-offset chunk size")
  }
  guard filter.parameters.length() == 20 else {
    raise UnsupportedFeature("unsupported scale-offset parameters")
  }
  let scale_type = filter.parameters[0].reinterpret_as_int()
  let datatype_class = filter.parameters[3].reinterpret_as_int()
  let datatype_size = filter.parameters[4].reinterpret_as_int()
  let sign = filter.parameters[5].reinterpret_as_int()
  let order = filter.parameters[6].reinterpret_as_int()
  let fill_available = filter.parameters[7].reinterpret_as_int()
  let filter_value = filter.parameters[1].reinterpret_as_int()
  guard datatype_size == element_size &&
    (sign == 0 || sign == 1) &&
    (order == 0 || order == 1) &&
    (fill_available == 0 || fill_available == 1) else {
    raise InvalidField("invalid scale-offset filter parameters")
  }
  let is_integer = datatype_class == 0 && scale_type == 2
  let is_float = datatype_class == 1 &&
    scale_type == 0 &&
    (element_size == 4 || element_size == 8)
  guard is_integer || is_float else {
    raise UnsupportedFeature("scale-offset datatype class is not supported")
  }
  if is_integer && filter_value == element_size * 8 {
    guard data.length() == expected_size else {
      raise InvalidField("unfiltered scale-offset chunk has the wrong size")
    }
    return data
  }
  guard data.length() >= 21 else {
    raise InvalidField("truncated scale-offset chunk")
  }
  let reader = Reader::new(data)
  let minbits = reader.read_u32_le().reinterpret_as_int()
  let minval_size = data[4].to_int()
  guard minbits >= 0 &&
    minbits < element_size * 8 &&
    minval_size >= 0 &&
    minval_size <= 8 &&
    (!is_float || minval_size >= element_size) else {
    raise InvalidField("invalid scale-offset chunk header")
  }
  let mut minval = 0UL
  for index in 0..<minval_size {
    minval = minval | (data[5 + index].to_uint64() << (index * 8))
  }
  let count = expected_size / element_size
  let element_bits = minbits.to_int64()
  let total_bits = count.to_int64() * element_bits
  guard total_bits >= 0 &&
    total_bits <= 0x7FFFFFFF &&
    (total_bits + 7) / 8 <= (data.length() - 21).to_int64() else {
    raise InvalidField("truncated scale-offset bit stream")
  }
  let fill = if fill_available == 1 {
    scaleoffset_parameter_value(filter.parameters, element_size)
  } else {
    0UL
  }
  let maximum_digits = if element_size == 4 { 38 } else { 308 }
  guard !is_float || (filter_value >= 0 && filter_value <= maximum_digits) else {
    raise UnsupportedFeature("unsupported floating-point scale factor")
  }
  let minimum = if is_float {
    read_float_at(data, 5, element_size, true)
  } else {
    0.0
  }
  let factor32 = if is_float && element_size == 4 {
    scaleoffset_float_factor32(filter_value)
  } else {
    Float(1.0)
  }
  let factor64 = if is_float && element_size == 8 {
    scaleoffset_float_factor64(filter_value)
  } else {
    1.0
  }
  let marker = if minbits == 0 { 0UL } else { (1UL << minbits) - 1UL }
  let mask = if element_size == 8 {
    0xFFFFFFFFFFFFFFFFUL
  } else {
    (1UL << (element_size * 8)) - 1UL
  }
  let output = FixedArray::make(expected_size, b'\x00')
  let mut bit_index = 0
  for element in 0..<count {
    let mut packed = 0UL
    let mut remaining = minbits
    while remaining > 0 {
      let byte_index = bit_index / 8
      let bit_offset = bit_index % 8
      let available = 8 - bit_offset
      let take = if remaining < available { remaining } else { available }
      let part = (data[21 + byte_index].to_uint() >> (available - take)) &
        ((1U << take) - 1U)
      packed = (packed << take) | part.to_uint64()
      bit_index = bit_index + take
      remaining = remaining - take
    }
    let decoded = if fill_available == 1 && packed == marker {
      Some(fill)
    } else {
      None
    }
    let offset = element * element_size
    match decoded {
      Some(value) =>
        scaleoffset_write_scalar(output, offset, element_size, order, value)
      None =>
        if is_float {
          if element_size == 4 {
            let value = Float::from_double(packed.to_double()) / factor32 +
              Float::from_double(minimum)
            scaleoffset_write_scalar(
              output,
              offset,
              element_size,
              order,
              value.reinterpret_as_uint().to_uint64(),
            )
          } else {
            let value = packed.to_double() / factor64 + minimum
            scaleoffset_write_scalar(
              output,
              offset,
              element_size,
              order,
              value.reinterpret_as_uint64(),
            )
          }
        } else {
          scaleoffset_write_scalar(
            output,
            offset,
            element_size,
            order,
            (packed + minval) & mask,
          )
        }
    }
  }
  output.unsafe_reinterpret_as_bytes()
}

///|
fn decode_filters(
  data : Bytes,
  filters : Array[Filter],
  mask : UInt32,
  element_size : Int32,
  expected_size : Int32,
) -> Bytes raise Hdf5Error {
  guard filters.length() == 32 || mask >> filters.length() == 0U else {
    raise InvalidField("chunk filter mask exceeds pipeline")
  }
  guard expected_size >= 0 &&
    expected_size <= 0x7FFFFFFF - filters.length() * 32 else {
    raise InvalidField("filtered chunk is too large")
  }
  let limit = expected_size + filters.length() * 32
  let mut output = data
  let mut index = filters.length()
  while index > 0 {
    index = index - 1
    if (mask & (1U << index)) != 0U {
      continue
    }
    let filter = filters[index]
    output = match filter.id {
      1 => {
        guard filter.parameters.length() == 1 && filter.parameters[0] <= 9U else {
          raise InvalidField("invalid deflate parameters")
        }
        guard output.length() >= 2 && output[0].to_int() >> 4 <= 7 else {
          raise InvalidField("invalid zlib window size")
        }
        @zlib.unzlib(output[:], checksum=true, limit~) catch {
          _ => raise InvalidField("invalid or oversized deflate chunk")
        }
      }
      2 => {
        guard filter.parameters.length() == 1 else {
          raise InvalidField("invalid shuffle element size")
        }
        let shuffle_size = if element_size > 0 {
          element_size
        } else {
          let parameter = filter.parameters[0]
          guard parameter > 0U && parameter <= 0x7FFFFFFFU else {
            raise InvalidField("invalid shuffle element size")
          }
          parameter.reinterpret_as_int()
        }
        guard filter.parameters[0] == shuffle_size.reinterpret_as_uint() else {
          raise InvalidField("invalid shuffle element size")
        }
        guard output.length() <= limit else {
          raise InvalidField("oversized shuffle chunk")
        }
        let elements = output.length() / shuffle_size
        let restored = FixedArray::make(output.length(), b'\x00')
        for byte_index in 0..<shuffle_size {
          for element in 0..<elements {
            restored[element * shuffle_size + byte_index] = output[byte_index *
              elements +
              element]
          }
        }
        for tail in (elements * shuffle_size)..<output.length() {
          restored[tail] = output[tail]
        }
        restored.unsafe_reinterpret_as_bytes()
      }
      3 => {
        guard filter.parameters.length() == 0 else {
          raise InvalidField("invalid Fletcher32 parameters")
        }
        strip_fletcher32(output)
      }
      5 => decode_nbit(output, filter.parameters, expected_size)
      6 => decode_scaleoffset(output, filter, element_size, expected_size)
      _ =>
        match filter.decoder {
          Some(decoder) => {
            let decoded = decoder.decode(
              output,
              filter.parameters,
              element_size,
              expected_size,
            )
            guard decoded.length() <= limit else {
              raise InvalidField("oversized custom filter chunk")
            }
            decoded
          }
          None =>
            raise UnsupportedFeature(
              "filter " + filter.id.to_string() + " is not supported",
            )
        }
    }
  }
  guard output.length() == expected_size else {
    raise InvalidField("decoded chunk size does not match declared shape")
  }
  output
}
