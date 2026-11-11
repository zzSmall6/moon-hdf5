///|
test "v1 B-tree encoder builds multi-level trees and correct final keys" {
  let capacity = 4
  let rank = 2
  let chunk_widths : Array[Int64] = [1L, 1L]
  let records : Array[BTreeV1ChunkRecord] = []
  for index in 0..<100 {
    let row = index / 16
    let column = index % 16
    records.push({
      address: 0x1000UL + index.to_int64().reinterpret_as_uint64() * 8UL,
      size: 4UL,
      mask: 0U,
      offsets: [
        row.to_int64().reinterpret_as_uint64(),
        column.to_int64().reinterpret_as_uint64(),
      ],
    })
  }
  let tree = encode_btree_v1_chunk_tree(
    0UL, records, chunk_widths, 4, 8, capacity,
  )
  let node_size = btree_v1_node_size(rank, 8, capacity)
  let leaf_sizes = btree_v1_group_sizes(records.length(), capacity)
  let parent_sizes = btree_v1_group_sizes(leaf_sizes.length(), capacity)
  let grandparent_sizes = btree_v1_group_sizes(parent_sizes.length(), capacity)
  assert_eq(leaf_sizes.length(), 25)
  assert_eq(parent_sizes.length(), 7)
  assert_eq(grandparent_sizes.length(), 2)
  let root_offset = (
      leaf_sizes.length() + parent_sizes.length() + grandparent_sizes.length()
    ) *
    node_size
  let root = Reader::at(tree, root_offset)
  assert_eq(root.read_bytes(4), b"TREE")
  assert_eq(root.read_u8(), b'\x01')
  assert_eq(root.read_u8().to_int(), 3)
  assert_eq(root.read_u16_le().to_int(), 2)

  let first_leaf_final_key = Reader::at(tree, 24 + 4 * (32 + 8))
  assert_eq(first_leaf_final_key.read_u32_le(), 0U)
  assert_eq(first_leaf_final_key.read_u32_le(), 0U)
  assert_eq(first_leaf_final_key.read_u64_le(), 1UL)
  assert_eq(first_leaf_final_key.read_u64_le(), 4UL)
  assert_eq(first_leaf_final_key.read_u64_le(), 4UL)

  let parsed : Array[RawChunkRecord] = []
  read_chunk_btree(
    tree,
    root_offset.to_int64().reinterpret_as_uint64(),
    -1,
    [1U, 1U],
    [16UL, 16UL],
    4,
    8,
    0UL,
    FixedArray::make(0, b'\x00'),
    [],
    Some([]),
    records=Some(parsed),
  )
  assert_eq(parsed.length(), records.length())
  assert_eq(parsed[0].address, 0x1000UL)
  assert_eq(parsed[0].offsets, [0UL, 0UL])
  assert_eq(parsed[99].address, 0x1000UL + 99UL * 8UL)
  assert_eq(parsed[99].offsets, [6UL, 3UL])
}
