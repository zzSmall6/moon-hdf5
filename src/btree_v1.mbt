///|
priv struct BTreeV1ChunkRecord {
  address : UInt64
  size : UInt64
  mask : UInt32
  offsets : Array[UInt64]
}

///|
priv struct WriterBTreeV1Node {
  level : Int32
  keys : Array[Bytes]
  child_addresses : Array[UInt64]
  child_nodes : Array[Int32]
  first_key : Bytes
  final_key : Bytes
  mut address : UInt64
  mut left_sibling : UInt64
  mut right_sibling : UInt64
}

///|
fn btree_v1_undefined_address(offset_size : Int32) -> UInt64 {
  if offset_size == 8 {
    0xFFFFFFFFFFFFFFFFUL
  } else {
    (1UL << (offset_size * 8)) - 1UL
  }
}

///|
fn encode_btree_v1_chunk_key(
  size : UInt64,
  mask : UInt32,
  offsets : Array[UInt64],
  storage_offset : UInt64,
) -> Bytes raise Hdf5Error {
  guard offsets.length() > 0 && size <= 0xFFFFFFFFUL else {
    raise InvalidField("invalid v1 B-tree chunk key")
  }
  let buffer = @buffer.Buffer::Buffer(size_hint=16 + 8 * offsets.length())
  writer_write_uint_width(buffer, size, 4)
  writer_write_uint_width(buffer, mask.to_uint64(), 4)
  for offset in offsets {
    buffer.write_uint64_le(offset)
  }
  buffer.write_uint64_le(storage_offset)
  buffer.to_bytes()
}

///|
fn btree_v1_final_key(
  record : BTreeV1ChunkRecord,
  chunk_widths : Array[Int64],
  element_size : Int32,
) -> Bytes raise Hdf5Error {
  guard record.offsets.length() == chunk_widths.length() && element_size > 0 else {
    raise InvalidField("v1 B-tree chunk key rank does not match the grid")
  }
  let offsets : Array[UInt64] = Array::make(record.offsets.length(), 0UL)
  for axis in 0..<record.offsets.length() {
    let width = chunk_widths[axis]
    guard width > 0L &&
      record.offsets[axis] <=
      0xFFFFFFFFFFFFFFFFUL - width.reinterpret_as_uint64() else {
      raise InvalidField("v1 B-tree final chunk key overflows")
    }
    offsets[axis] = record.offsets[axis] + width.reinterpret_as_uint64()
  }
  encode_btree_v1_chunk_key(
    0UL,
    0U,
    offsets,
    element_size.to_int64().reinterpret_as_uint64(),
  )
}

///|
fn btree_v1_chunk_offsets(
  index : Int32,
  grid : Array[Int64],
  chunk_widths : Array[Int64],
) -> Array[UInt64] raise Hdf5Error {
  guard grid.length() == chunk_widths.length() else {
    raise InvalidField("v1 B-tree chunk grid rank mismatch")
  }
  let offsets : Array[UInt64] = Array::make(grid.length(), 0UL)
  let mut remaining = index.to_int64()
  let mut axis = grid.length()
  while axis > 0 {
    axis = axis - 1
    let width = grid[axis]
    guard width > 0L && chunk_widths[axis] > 0L else {
      raise InvalidField("invalid v1 B-tree chunk grid")
    }
    offsets[axis] = (remaining % width * chunk_widths[axis]).reinterpret_as_uint64()
    remaining = remaining / width
  }
  guard remaining == 0L else {
    raise InvalidField("v1 B-tree chunk index is outside the grid")
  }
  offsets
}

///|
fn btree_v1_node_size(
  rank : Int32,
  offset_size : Int32,
  capacity : Int32,
) -> Int32 raise Hdf5Error {
  guard capacity > 0 &&
    capacity <= 65535 &&
    offset_size >= 1 &&
    offset_size <= 8 &&
    rank > 0 else {
    raise InvalidField("invalid v1 B-tree geometry")
  }
  let key_size = 8L + 8L * (rank.to_int64() + 1L)
  let header_size = 8L + 2L * offset_size.to_int64()
  let node_size = header_size +
    capacity.to_int64() * offset_size.to_int64() +
    (capacity.to_int64() + 1L) * key_size
  guard node_size <= 0x7FFFFFFFL else {
    raise UnsupportedFeature("v1 B-tree node is too large")
  }
  node_size.to_int()
}

///|
fn btree_v1_group_sizes(
  count : Int32,
  capacity : Int32,
) -> Array[Int32] raise Hdf5Error {
  guard count > 0 && capacity > 0 else {
    raise InvalidField("invalid v1 B-tree group geometry")
  }
  let count64 = count.to_int64()
  let capacity64 = capacity.to_int64()
  let group_count = (count64 + capacity64 - 1L) / capacity64
  guard group_count > 0 && group_count <= 0x7FFFFFFFL else {
    raise InvalidField("v1 B-tree group count is out of range")
  }
  let base = count64 / group_count
  let remainder = (count64 % group_count).to_int()
  let sizes : Array[Int32] = []
  for group in 0..<group_count.to_int() {
    let extra = if group < remainder { 1L } else { 0L }
    sizes.push((base + extra).to_int())
  }
  sizes
}

///|
fn encode_btree_v1_node(
  node : WriterBTreeV1Node,
  nodes : Array[WriterBTreeV1Node],
  node_size : Int32,
  key_size : Int32,
  offset_size : Int32,
) -> Bytes raise Hdf5Error {
  let undefined = btree_v1_undefined_address(offset_size)
  guard node.keys.length() <= 65535 else {
    raise InvalidField("v1 B-tree node has too many entries")
  }
  if node.level == 0 {
    guard node.child_addresses.length() == node.keys.length() &&
      node.child_nodes.length() == 0 else {
      raise InvalidField("invalid v1 B-tree leaf children")
    }
  } else {
    guard node.child_nodes.length() == node.keys.length() &&
      node.child_addresses.length() == 0 else {
      raise InvalidField("invalid v1 B-tree internal children")
    }
  }
  guard node.final_key.length() == key_size else {
    raise InvalidField("v1 B-tree final key size changed")
  }
  let buffer = @buffer.Buffer::Buffer(size_hint=node_size)
  writer_write_bytes(buffer, b"TREE")
  buffer.write_byte(b'\x01')
  buffer.write_byte(node.level.to_byte())
  buffer.write_uint16_le(node.keys.length().to_uint16())
  writer_write_uint_width(
    buffer,
    if node.left_sibling == 0UL {
      undefined
    } else {
      node.left_sibling
    },
    offset_size,
  )
  writer_write_uint_width(
    buffer,
    if node.right_sibling == 0UL {
      undefined
    } else {
      node.right_sibling
    },
    offset_size,
  )
  for entry in 0..<node.keys.length() {
    let key = node.keys[entry]
    guard key.length() == key_size else {
      raise InvalidField("v1 B-tree key size changed")
    }
    writer_write_bytes(buffer, key)
    let child = if node.level == 0 {
      node.child_addresses[entry]
    } else {
      nodes[node.child_nodes[entry]].address
    }
    writer_write_uint_width(buffer, child, offset_size)
  }
  writer_write_bytes(buffer, node.final_key)
  guard buffer.length() <= node_size else {
    raise InvalidField("v1 B-tree node exceeds its computed size")
  }
  writer_write_zeros(buffer, node_size - buffer.length())
  buffer.to_bytes()
}

///|
fn encode_btree_v1_chunk_tree(
  start : UInt64,
  records : Array[BTreeV1ChunkRecord],
  chunk_widths : Array[Int64],
  element_size : Int32,
  offset_size : Int32,
  capacity : Int32,
) -> Bytes raise Hdf5Error {
  guard records.length() > 0 else {
    raise InvalidField("cannot encode an empty v1 B-tree chunk tree")
  }
  let rank = chunk_widths.length()
  guard element_size > 0 else {
    raise InvalidField("v1 B-tree element size must be positive")
  }
  let node_size = btree_v1_node_size(rank, offset_size, capacity)
  let key_size = 8 + 8 * (rank + 1)
  let nodes : Array[WriterBTreeV1Node] = []
  let leaf_sizes = btree_v1_group_sizes(records.length(), capacity)
  let mut cursor = 0
  for group_size in leaf_sizes {
    let keys : Array[Bytes] = []
    let addresses : Array[UInt64] = []
    let end = cursor + group_size
    guard end <= records.length() else {
      raise InvalidField("v1 B-tree leaf grouping exceeds the record count")
    }
    for record_index in cursor..<end {
      let record = records[record_index]
      guard record.offsets.length() == rank else {
        raise InvalidField("v1 B-tree record rank does not match the grid")
      }
      keys.push(
        encode_btree_v1_chunk_key(record.size, record.mask, record.offsets, 0UL),
      )
      addresses.push(record.address)
    }
    let final_key = btree_v1_final_key(
      records[end - 1],
      chunk_widths,
      element_size,
    )
    nodes.push({
      level: 0,
      keys,
      child_addresses: addresses,
      child_nodes: [],
      first_key: keys[0],
      final_key,
      address: 0UL,
      left_sibling: 0UL,
      right_sibling: 0UL,
    })
    cursor = end
  }
  guard cursor == records.length() else {
    raise InvalidField("v1 B-tree leaf grouping consumed the wrong records")
  }
  let mut current : Array[Int32] = []
  for index in 0..<nodes.length() {
    current.push(index)
  }
  let mut level = 0
  while current.length() > 1 {
    let parent_sizes = btree_v1_group_sizes(current.length(), capacity)
    let parents : Array[Int32] = []
    let mut child_cursor = 0
    for group_size in parent_sizes {
      let keys : Array[Bytes] = []
      let children : Array[Int32] = []
      let end = child_cursor + group_size
      guard end <= current.length() else {
        raise InvalidField("v1 B-tree parent grouping exceeds the node count")
      }
      for child_offset in child_cursor..<end {
        let child = current[child_offset]
        keys.push(nodes[child].first_key)
        children.push(child)
      }
      let parent_index = nodes.length()
      nodes.push({
        level: level + 1,
        keys,
        child_addresses: [],
        child_nodes: children,
        first_key: nodes[children[0]].first_key,
        final_key: nodes[children[children.length() - 1]].final_key,
        address: 0UL,
        left_sibling: 0UL,
        right_sibling: 0UL,
      })
      parents.push(parent_index)
      child_cursor = end
    }
    guard child_cursor == current.length() else {
      raise InvalidField("v1 B-tree parent grouping consumed the wrong nodes")
    }
    current = parents
    level = level + 1
  }
  let node_count = nodes.length()
  guard node_count.to_int64() * node_size.to_int64() <= 0x7FFFFFFFL else {
    raise UnsupportedFeature("v1 B-tree image is too large")
  }
  for index in 0..<node_count {
    nodes[index].address = start + node_size.to_uint64() * index.to_uint64()
  }
  let previous_at_level : Array[Int32] = Array::make(level + 1, -1)
  for index in 0..<node_count {
    let node_level = nodes[index].level
    let previous = previous_at_level[node_level]
    if previous >= 0 {
      nodes[index].left_sibling = nodes[previous].address
      nodes[previous].right_sibling = nodes[index].address
    }
    previous_at_level[node_level] = index
  }
  let output = @buffer.Buffer::Buffer(size_hint=node_count * node_size)
  for node in nodes {
    writer_write_bytes(
      output,
      encode_btree_v1_node(node, nodes, node_size, key_size, offset_size),
    )
  }
  guard output.length() == node_count * node_size else {
    raise InvalidField("v1 B-tree image size changed during encoding")
  }
  output.to_bytes()
}
