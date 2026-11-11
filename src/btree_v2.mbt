///|
fn validate_btree_v2_config(config : BTreeV2Config) -> Unit raise Hdf5Error {
  guard config.node_size > 10 &&
    config.split_percent > 0 &&
    config.split_percent <= 100 &&
    config.merge_percent > 0 &&
    config.merge_percent <= 100 &&
    config.merge_percent * 2 < config.split_percent else {
    raise InvalidField("invalid v2 B-tree configuration")
  }
}

///|
fn btree_integer_width(value : UInt64) -> Int32 {
  let mut width = 1
  let mut remaining = value
  while remaining > 255UL {
    remaining = remaining >> 8
    width = width + 1
  }
  width
}

///|
priv struct ChunkBTreeV2 {
  source : &Hdf5Source
  layout : DataLayout
  dataspace : Dataspace
  datatype : Datatype
  offset_size : Int32
  base_address : UInt64
  undefined : UInt64
  filters : Array[Filter]
  output : FixedArray[Byte]
  selection : Array[UInt64]?
  records : Array[RawChunkRecord]?
  config : BTreeV2Config
  client : Int32
  record_size : Int32
  size_width : Int32
  chunk_size : Int32
  count_width : Int32
  capacities : Array[Int32]
  total_widths : Array[Int32]
  mut previous : Array[UInt64]?
  mut remaining_records : UInt64
}

///|
fn ChunkBTreeV2::record(
  self : ChunkBTreeV2,
  reader : Reader,
  node_start : Int32,
  metadata_checksum_offset : Int32,
) -> Unit raise Hdf5Error {
  guard self.remaining_records > 0UL else {
    raise InvalidField("v2 B-tree exceeds declared record count")
  }
  self.remaining_records = self.remaining_records - 1UL
  let address_offset = node_start + reader.position()
  let address = reader.read_uint_le(self.offset_size)
  guard address != self.undefined else {
    raise InvalidField("undefined v2 B-tree chunk address")
  }
  let stored_size = if self.client == 11 {
    reader.read_var_uint_le(self.size_width)
  } else {
    self.chunk_size.to_uint64()
  }
  let stored_size_offset = node_start + reader.position() - self.size_width
  let mask = if self.client == 11 { reader.read_u32_le() } else { 0U }
  let coordinates : Array[UInt64] = []
  let offsets : Array[UInt64] = []
  for dimension in 0..<self.layout.chunk_dimensions.length() {
    let coordinate = reader.read_u64_le()
    let width = self.layout.chunk_dimensions[dimension].to_uint64()
    guard self.dataspace.dimensions[dimension] > 0UL &&
      coordinate <= (self.dataspace.dimensions[dimension] - 1UL) / width else {
      raise InvalidField("v2 B-tree chunk coordinate is outside dataspace")
    }
    coordinates.push(coordinate)
    offsets.push(coordinate * width)
  }
  if self.previous is Some(previous) {
    let mut greater = false
    for dimension in 0..<coordinates.length() {
      if coordinates[dimension] != previous[dimension] {
        greater = coordinates[dimension] > previous[dimension]
        break
      }
    }
    guard greater else {
      raise InvalidField("v2 B-tree chunk records are not strictly ordered")
    }
  }
  self.previous = Some(coordinates)
  push_located_raw_chunk_record(
    self.records,
    offsets,
    address,
    stored_size,
    mask,
    address_offset,
    self.offset_size,
    if self.client == 11 {
      stored_size_offset
    } else {
      -1
    },
    if self.client == 11 {
      self.size_width
    } else {
      0
    },
    if self.client == 11 {
      node_start
    } else {
      -1
    },
    if self.client == 11 {
      metadata_checksum_offset
    } else {
      -1
    },
  )
  if !chunk_contains_selection(
      offsets,
      self.layout.chunk_dimensions,
      self.dataspace.dimensions,
      self.selection,
    ) {
    return
  }
  let decoded = read_source_filtered_chunk(
    self.source,
    address,
    self.base_address,
    stored_size,
    self.filters,
    mask,
    self.datatype.size,
    self.chunk_size,
  )
  copy_chunk(
    decoded,
    decoded.length(),
    offsets,
    self.layout.chunk_dimensions,
    self.dataspace.dimensions,
    self.datatype.size,
    self.output,
    self.selection,
  )
}

///|
fn ChunkBTreeV2::node(
  self : ChunkBTreeV2,
  address : UInt64,
  depth : Int32,
  count : Int32,
  total : UInt64,
) -> Unit raise Hdf5Error {
  guard address != self.undefined &&
    count > 0 &&
    count <= self.capacities[depth] &&
    count.to_uint64() <= total &&
    total <= self.remaining_records else {
    raise InvalidField("invalid v2 B-tree node record count")
  }
  let start = source_address_to_offset(address, self.base_address)
  guard start <= 0x7FFFFFFFUL else {
    raise InvalidField("v2 B-tree node offset is out of range")
  }
  let node_start = start.to_int()
  let node = self.source.read_at(start, self.config.node_size)
  let reader = Reader::new(node)
  guard reader.read_bytes(4) == (if depth == 0 { b"BTLF" } else { b"BTIN" }) &&
    reader.read_u8() == b'\x00' &&
    reader.read_u8().to_int() == self.client else {
    raise InvalidField("invalid v2 B-tree node")
  }
  let records_start = reader.position()
  reader.skip(count * self.record_size)
  let children : Array[(UInt64, Int32, UInt64)] = []
  let mut computed_total = count.to_uint64()
  if depth > 0 {
    for _ in 0..<(count + 1) {
      let child_address = reader.read_uint_le(self.offset_size)
      let child_count = reader.read_var_uint_le(self.count_width)
      guard child_count <= self.capacities[depth - 1].to_uint64() else {
        raise InvalidField("v2 B-tree child count exceeds capacity")
      }
      let child_total = if depth > 1 {
        reader.read_var_uint_le(self.total_widths[depth - 1])
      } else {
        child_count
      }
      guard computed_total <= total && child_total <= total - computed_total else {
        raise InvalidField("invalid v2 B-tree subtree count")
      }
      computed_total = computed_total + child_total
      children.push((child_address, child_count.to_int(), child_total))
    }
  }
  let checksum_offset = node_start + reader.position()
  verify_metadata_checksum(node, 0, reader.position(), "v2 B-tree node")
  reader.skip(4)
  guard computed_total == total else {
    raise InvalidField("v2 B-tree subtree count mismatch")
  }
  for record_index in 0..<count {
    if depth > 0 {
      let (child_address, child_count, child_total) = children[record_index]
      self.node(child_address, depth - 1, child_count, child_total)
    }
    self.record(
      Reader::at(node, records_start + record_index * self.record_size),
      node_start,
      checksum_offset,
    )
  }
  if depth > 0 {
    let (child_address, child_count, child_total) = children[count]
    self.node(child_address, depth - 1, child_count, child_total)
  }
}

///|
fn read_btree_v2_chunks(
  bytes : Bytes,
  layout : DataLayout,
  dataspace : Dataspace,
  datatype : Datatype,
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
  filters : Array[Filter],
  output : FixedArray[Byte],
  undefined : UInt64,
  config : BTreeV2Config,
  selection : Array[UInt64]?,
  records~ : Array[RawChunkRecord]? = None,
) -> Unit raise Hdf5Error {
  let sources : Array[&Hdf5Source] = [bytes]
  read_source_btree_v2_chunks(
    sources[0],
    layout,
    dataspace,
    datatype,
    offset_size,
    length_size,
    base_address,
    filters,
    output,
    undefined,
    config,
    selection,
    records~,
  )
}

///|
fn read_source_btree_v2_chunks(
  source : &Hdf5Source,
  layout : DataLayout,
  dataspace : Dataspace,
  datatype : Datatype,
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
  filters : Array[Filter],
  output : FixedArray[Byte],
  undefined : UInt64,
  config : BTreeV2Config,
  selection : Array[UInt64]?,
  records~ : Array[RawChunkRecord]? = None,
) -> Unit raise Hdf5Error {
  validate_btree_v2_config(config)
  let header_offset = source_address_to_offset(
    layout.data_address,
    base_address,
  )
  let header_content = 18 + offset_size + length_size
  let header = Reader::new(
    source_read_checksummed(
      source, header_offset, header_content, "v2 B-tree header",
    ),
  )
  guard header.read_bytes(4) == b"BTHD" && header.read_u8() == b'\x00' else {
    raise InvalidField("invalid v2 B-tree header")
  }
  let client = header.read_u8().to_int()
  guard client == (if filters.length() == 0 { 10 } else { 11 }) &&
    header.read_u32_le() == config.node_size.reinterpret_as_uint() else {
    raise InvalidField("invalid v2 B-tree type or node size")
  }
  let record_size = header.read_u16_le().to_int()
  let size_width = record_size -
    offset_size -
    8 * layout.chunk_dimensions.length() -
    4
  guard (client == 10 && size_width == -4) ||
    (client == 11 && size_width >= 1 && size_width <= 8) else {
    raise InvalidField("invalid v2 B-tree record size")
  }
  let depth = header.read_u16_le().to_int()
  guard depth <= 32 &&
    header.read_u8().to_int() == config.split_percent &&
    header.read_u8().to_int() == config.merge_percent else {
    raise InvalidField("invalid v2 B-tree depth or thresholds")
  }
  let root = header.read_uint_le(offset_size)
  let count = header.read_u16_le().to_int()
  let total = header.read_uint_le(length_size)
  guard header.position() == header_content else {
    raise InvalidField("v2 B-tree header size mismatch")
  }
  header.skip(4)
  guard total <= source.length() / record_size.to_uint64() else {
    raise InvalidField("v2 B-tree record count exceeds input")
  }
  if root == undefined {
    guard total == 0UL && count == 0 && depth == 0 else {
      raise InvalidField("invalid empty v2 B-tree")
    }
    return
  }
  let leaf_capacity = (config.node_size - 10) / record_size
  guard leaf_capacity > 0 && leaf_capacity <= 65535 else {
    raise InvalidField("invalid v2 B-tree leaf capacity")
  }
  let count_width = btree_integer_width(leaf_capacity.to_uint64())
  let capacities = [leaf_capacity]
  let total_widths = [0]
  let mut cumulative = leaf_capacity.to_uint64()
  for level in 1..<(depth + 1) {
    let pointer_size = offset_size + count_width + total_widths[level - 1]
    let capacity = (config.node_size - 10 - pointer_size) /
      (record_size + pointer_size)
    guard capacity > 0 &&
      cumulative <=
      (0xFFFFFFFFFFFFFFFFUL - capacity.to_uint64()) / (capacity + 1).to_uint64() else {
      raise InvalidField("invalid v2 B-tree internal capacity")
    }
    cumulative = cumulative * (capacity + 1).to_uint64() + capacity.to_uint64()
    capacities.push(capacity)
    total_widths.push(btree_integer_width(cumulative))
  }
  let context : ChunkBTreeV2 = {
    source,
    layout,
    dataspace,
    datatype,
    offset_size,
    base_address,
    undefined,
    filters,
    output,
    selection,
    records,
    config,
    client,
    record_size,
    size_width,
    chunk_size: chunk_byte_size(layout.chunk_dimensions, datatype.size),
    count_width,
    capacities,
    total_widths,
    previous: None,
    remaining_records: total,
  }
  context.node(root, depth, count, total)
  guard context.remaining_records == 0UL else {
    raise InvalidField("v2 B-tree record count mismatch")
  }
}

///|
fn btree_v2_stored_size_width(value : UInt64) -> Int32 {
  let width = btree_integer_width(value)
  if width > 8 {
    8
  } else {
    width
  }
}

///|
fn btree_v2_coordinates(
  index : Int32,
  grid : Array[Int64],
) -> Array[UInt64] raise Hdf5Error {
  let coordinates : Array[UInt64] = Array::make(grid.length(), 0UL)
  let mut remaining = index.to_int64()
  let mut axis = grid.length()
  while axis > 0 {
    axis = axis - 1
    let width = grid[axis]
    guard width > 0L else { raise InvalidField("invalid v2 B-tree chunk grid") }
    coordinates[axis] = (remaining % width).reinterpret_as_uint64()
    remaining = remaining / width
  }
  guard remaining == 0L else {
    raise InvalidField("v2 B-tree chunk index is outside the grid")
  }
  coordinates
}

///|
fn encode_btree_v2_chunk_record(
  address : UInt64,
  stored_size : UInt64,
  filter_mask : UInt32,
  coordinates : Array[UInt64],
  filtered : Bool,
  size_width : Int32,
  offset_size : Int32,
) -> Bytes raise Hdf5Error {
  guard offset_size >= 1 && offset_size <= 8 else {
    raise UnsupportedAddressSize(offset_size)
  }
  if filtered {
    guard size_width >= 1 && size_width <= 8 else {
      raise InvalidField("invalid v2 B-tree chunk size width")
    }
    guard btree_integer_width(stored_size) <= size_width else {
      raise InvalidField("v2 B-tree chunk size exceeds its record field")
    }
  }
  let buffer = @buffer.Buffer::Buffer(
    size_hint=offset_size +
      (if filtered { size_width + 4 } else { 0 }) +
      8 * coordinates.length(),
  )
  writer_write_uint_width(buffer, address, offset_size)
  if filtered {
    writer_write_uint_width(buffer, stored_size, size_width)
    writer_write_uint_width(buffer, filter_mask.to_uint64(), 4)
  }
  for coordinate in coordinates {
    buffer.write_uint64_le(coordinate)
  }
  buffer.to_bytes()
}

///|
fn encode_btree_v2_header(
  root : UInt64,
  root_count : Int32,
  depth : Int32,
  total : UInt64,
  node_size : Int32,
  record_size : Int32,
  split_percent : Int32,
  merge_percent : Int32,
  client : Int32,
  offset_size : Int32,
  length_size : Int32,
) -> Bytes raise Hdf5Error {
  guard root_count >= 0 &&
    root_count <= 65535 &&
    depth >= 0 &&
    depth <= 65535 &&
    split_percent >= 1 &&
    split_percent <= 100 &&
    merge_percent >= 1 &&
    merge_percent <= 100 &&
    offset_size >= 1 &&
    offset_size <= 8 &&
    length_size >= 1 &&
    length_size <= 8 else {
    raise InvalidField("invalid v2 B-tree header fields")
  }
  let buffer = @buffer.Buffer::Buffer(size_hint=22 + offset_size + length_size)
  writer_write_bytes(buffer, b"BTHD")
  buffer.write_byte(b'\x00')
  buffer.write_byte(client.to_byte())
  buffer.write_uint_le(node_size.reinterpret_as_uint())
  buffer.write_uint16_le(record_size.to_uint16())
  buffer.write_uint16_le(depth.to_uint16())
  buffer.write_byte(split_percent.to_byte())
  buffer.write_byte(merge_percent.to_byte())
  writer_write_uint_width(buffer, root, offset_size)
  buffer.write_uint16_le(root_count.to_uint16())
  writer_write_uint_width(buffer, total, length_size)
  buffer.write_uint_le(metadata_checksum(buffer.to_bytes(), 0, buffer.length()))
  buffer.to_bytes()
}

///|
fn encode_btree_v2_node(
  node : WriterBTreeV2Node,
  nodes : Array[WriterBTreeV2Node],
  node_size : Int32,
  record_size : Int32,
  offset_size : Int32,
  count_width : Int32,
  total_widths : Array[Int32],
  client : Int32,
) -> Bytes raise Hdf5Error {
  let buffer = @buffer.Buffer::Buffer(size_hint=node_size)
  writer_write_bytes(buffer, if node.depth == 0 { b"BTLF" } else { b"BTIN" })
  buffer.write_byte(b'\x00')
  buffer.write_byte(client.to_byte())
  for record in node.records {
    guard record.length() == record_size else {
      raise InvalidField("v2 B-tree record size changed")
    }
    writer_write_bytes(buffer, record)
  }
  if node.depth > 0 {
    for child_index in node.children {
      let child = nodes[child_index]
      writer_write_uint_width(buffer, child.address, offset_size)
      writer_write_uint_width(
        buffer,
        child.records.length().to_uint64(),
        count_width,
      )
      if node.depth > 1 {
        writer_write_uint_width(
          buffer,
          child.total,
          total_widths[node.depth - 1],
        )
      }
    }
  }
  guard buffer.length() + 4 <= node_size else {
    raise InvalidField("v2 B-tree node exceeds its configured size")
  }
  buffer.write_uint_le(metadata_checksum(buffer.to_bytes(), 0, buffer.length()))
  writer_write_zeros(buffer, node_size - buffer.length())
  buffer.to_bytes()
}

///|
fn encode_btree_v2_chunk_tree(
  start : UInt64,
  records : Array[Bytes],
  config : BTreeV2Config,
  offset_size : Int32,
  length_size : Int32,
  record_size : Int32,
  client : Int32,
) -> Bytes raise Hdf5Error {
  validate_btree_v2_config(config)
  guard record_size > 0 && record_size <= 65535 else {
    raise InvalidField("invalid v2 B-tree record size")
  }
  let header_length = 22 + offset_size + length_size
  let (capacities, total_widths, count_width) = writer_btree_capacities(
    config.node_size,
    record_size,
    offset_size,
  )
  let nodes : Array[WriterBTreeV2Node] = []
  let mut root = -1
  let mut depth = 0
  if records.length() > 0 {
    depth = writer_btree_depth(records.length().to_uint64(), capacities)
    root = writer_btree_build_node(records, depth, capacities, nodes)
  }
  let node_start = writer_align(header_length)
  for index in 0..<nodes.length() {
    nodes[index].address = start +
      node_start.to_uint64() +
      config.node_size.to_uint64() * index.to_uint64()
  }
  let root_address = if root >= 0 {
    nodes[root].address
  } else {
    0xFFFFFFFFFFFFFFFFUL
  }
  let root_count = if root >= 0 { nodes[root].records.length() } else { 0 }
  let header = encode_btree_v2_header(
    root_address,
    root_count,
    depth,
    records.length().to_uint64(),
    config.node_size,
    record_size,
    config.split_percent,
    config.merge_percent,
    client,
    offset_size,
    length_size,
  )
  let tree_length = if nodes.length() == 0 {
    header_length
  } else {
    node_start + config.node_size * nodes.length()
  }
  let output = @buffer.Buffer::Buffer(size_hint=tree_length)
  writer_write_bytes(output, header)
  if nodes.length() > 0 {
    writer_write_zeros(output, node_start - output.length())
    for node in nodes {
      writer_write_bytes(
        output,
        encode_btree_v2_node(
          node,
          nodes,
          config.node_size,
          record_size,
          offset_size,
          count_width,
          total_widths,
          client,
        ),
      )
    }
  }
  guard output.length() == tree_length else {
    raise InvalidField("v2 B-tree tree size changed during encoding")
  }
  output.to_bytes()
}
