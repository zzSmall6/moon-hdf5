///|
priv struct EditableLink {
  link : GroupLink
  creation_order : UInt64
}

///|
fn validate_editable_link_name(name : String) -> Unit raise Hdf5Error {
  let encoded = @utf8.encode(name)
  guard encoded.length() > 0 && encoded.length() <= 65535 else {
    raise InvalidArgument("link name is empty or too long")
  }
  guard name != "." && name != ".." && !name.contains("/") else {
    raise InvalidArgument("link name is invalid")
  }
  for byte in encoded {
    guard byte != b'\x00' else {
      raise InvalidArgument("link names cannot contain NUL")
    }
  }
}

///|
fn validate_editable_soft_target(target : String) -> Unit raise Hdf5Error {
  let encoded = @utf8.encode(target)
  guard encoded.length() > 0 && encoded.length() <= 65535 else {
    raise InvalidArgument("soft link target is empty or too long")
  }
  for byte in encoded {
    guard byte != b'\x00' else {
      raise InvalidArgument("soft link targets cannot contain NUL")
    }
  }
}

///|
fn validate_editable_external_target(
  target_file : String,
  target_path : String,
) -> Unit raise Hdf5Error {
  let file = @utf8.encode(target_file)
  let path = @utf8.encode(target_path)
  guard file.length() > 0 && file.length() <= 65535 else {
    raise InvalidArgument("external link target file is empty or too long")
  }
  guard path.length() > 0 && path.length() <= 65535 else {
    raise InvalidArgument("external link target path is empty or too long")
  }
  for byte in file {
    guard byte != b'\x00' else {
      raise InvalidArgument("external link target files cannot contain NUL")
    }
  }
  for byte in path {
    guard byte != b'\x00' else {
      raise InvalidArgument("external link target paths cannot contain NUL")
    }
  }
}

///|
fn editable_link_index(links : Array[EditableLink], name : String) -> Int32? {
  links.search_by(link => link.link.name == name)
}

///|
fn editable_link_creation_order(data : Bytes) -> UInt64? raise Hdf5Error {
  let reader = Reader::new(data)
  guard reader.read_u8() == b'\x01' else {
    raise InvalidField("unsupported link message version")
  }
  let flags = reader.read_u8().to_int()
  if (flags & 0x08) != 0 {
    ignore(reader.read_u8())
  }
  if (flags & 0x04) != 0 {
    Some(reader.read_u64_le())
  } else {
    None
  }
}

///|
fn editable_link_storage(
  header : ObjectHeader,
  offset_size : Int32,
) -> (Bool, UInt64, UInt64, UInt64) raise Hdf5Error {
  let undefined = heap_undefined_address(offset_size)
  match header.first(msg_link_info) {
    Some(message) => {
      let reader = Reader::new(message.data)
      guard reader.read_u8() == b'\x00' else {
        raise UnsupportedFeature("unsupported link info message version")
      }
      let flags = reader.read_u8().to_int()
      guard (flags & 0xFC) == 0 && (flags & 0x03) != 2 else {
        raise InvalidField("invalid link info flags")
      }
      let tracking = (flags & 0x01) != 0
      if tracking {
        reader.skip(8)
      }
      let heap_address = reader.read_uint_le(offset_size)
      let name_index_address = reader.read_uint_le(offset_size)
      let order_index_address = if (flags & 0x02) != 0 {
        reader.read_uint_le(offset_size)
      } else {
        undefined
      }
      (tracking, heap_address, name_index_address, order_index_address)
    }
    None => (false, undefined, undefined, undefined)
  }
}

///|
fn Hdf5File::editable_dense_link_filters(
  self : Hdf5File,
  header : ObjectHeader,
) -> MetadataHeapFilters raise Hdf5Error {
  if header.first(msg_symbol_table) is Some(_) {
    return MetadataHeapFilters::empty()
  }
  let offset_size = self.superblock.offset_size
  let (_, heap_address, _, _) = editable_link_storage(header, offset_size)
  if heap_address == heap_undefined_address(offset_size) {
    return MetadataHeapFilters::empty()
  }
  let heap = parse_fractal_heap(
    self.bytes,
    heap_address,
    offset_size,
    self.superblock.length_size,
    self.superblock.base_address,
    custom_filters=self.custom_filters,
  )
  MetadataHeapFilters::from_heap(heap)
}

///|
fn editable_dense_links(
  bytes : Bytes,
  heap_address : UInt64,
  index_address : UInt64,
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
  tracking : Bool,
  custom_filters~ : Array[&Hdf5CustomFilter] = [],
) -> Array[EditableLink] raise Hdf5Error {
  let heap = parse_fractal_heap(
    bytes,
    heap_address,
    offset_size,
    length_size,
    base_address,
    custom_filters~,
  )
  let contents = read_v2_btree_records(
    bytes,
    index_address,
    offset_size,
    length_size,
    base_address,
    [5, 6],
    "editable link index",
  )
  let links : Array[EditableLink] = []
  for record_index in 0..<contents.records.length() {
    let reader = Reader::new(contents.records[record_index])
    let record_order = if contents.node_type == 6 {
      reader.read_u64_le()
    } else {
      0UL
    }
    if contents.node_type == 5 {
      reader.skip(4)
    }
    let heap_id = reader.read_bytes(
      contents.record_size - (if contents.node_type == 5 { 4 } else { 8 }),
    )
    let data = heap.get_data(heap_id)
    let link = parse_dense_link(data, b'\x00', offset_size)
    let creation_order = match editable_link_creation_order(data) {
      Some(order) => order
      None =>
        if contents.node_type == 6 {
          record_order
        } else if tracking {
          record_order
        } else {
          record_index.to_uint64()
        }
    }
    links.push({ link, creation_order, })
  }
  if tracking {
    links.sort_by((left, right) => {
      if left.creation_order < right.creation_order {
        -1
      } else if left.creation_order > right.creation_order {
        1
      } else if left.link.name < right.link.name {
        -1
      } else if left.link.name > right.link.name {
        1
      } else {
        0
      }
    })
  } else {
    links.sort_by((left, right) => {
      if left.link.name < right.link.name {
        -1
      } else if left.link.name > right.link.name {
        1
      } else {
        0
      }
    })
  }
  links
}

///|
fn Hdf5File::editable_links(
  self : Hdf5File,
  header : ObjectHeader,
) -> Array[EditableLink] raise Hdf5Error {
  match header.first(msg_symbol_table) {
    Some(message) => {
      let offset_size = self.superblock.offset_size
      let length_size = self.superblock.length_size
      let symbol_table = parse_symbol_table_message(
        message, offset_size, length_size,
      )
      let links = read_group_links_v1(
        self.bytes,
        symbol_table.btree_address,
        symbol_table.heap_address,
        offset_size,
        length_size,
        self.superblock.base_address,
      )
      let editable : Array[EditableLink] = []
      for index in 0..<links.length() {
        let link = links[index]
        let normalized = if link.cache_type == 2U {
          link
        } else {
          { ..link, cache_type: 0U, link_target: None, external_path: None, }
        }
        editable.push({ link: normalized, creation_order: index.to_uint64(), })
      }
      return editable
    }
    None => ()
  }
  let offset_size = self.superblock.offset_size
  let length_size = self.superblock.length_size
  let (tracking, heap_address, name_index_address, order_index_address) = editable_link_storage(
    header, offset_size,
  )
  if heap_address != heap_undefined_address(offset_size) {
    let index_address = if (
        tracking ||
        !header.messages.any(message => message.msg_type == msg_link)
      ) &&
      order_index_address != heap_undefined_address(offset_size) {
      order_index_address
    } else if name_index_address != heap_undefined_address(offset_size) {
      name_index_address
    } else if order_index_address != heap_undefined_address(offset_size) {
      order_index_address
    } else {
      raise InvalidField("dense group has no name or order index")
    }
    return editable_dense_links(
      self.bytes,
      heap_address,
      index_address,
      offset_size,
      length_size,
      self.superblock.base_address,
      tracking,
      custom_filters=self.custom_filters,
    )
  }
  guard name_index_address == heap_undefined_address(offset_size) &&
    order_index_address == heap_undefined_address(offset_size) else {
    raise InvalidField("group index is present without a fractal heap")
  }
  let links : Array[EditableLink] = []
  let messages = header.find(msg_link)
  for index in 0..<messages.length() {
    let link = parse_link_message(messages[index], offset_size)
    let creation_order = match
      editable_link_creation_order(messages[index].data) {
      Some(order) => order
      None =>
        if tracking {
          raise InvalidField("tracked compact link has no creation order")
        } else {
          index.to_uint64()
        }
    }
    links.push({ link, creation_order, })
  }
  links
}

///|
fn encode_editable_link(
  link : EditableLink,
  name : String,
  track_order : Bool,
  offset_size : Int32,
) -> Bytes raise Hdf5Error {
  let item = link.link
  if item.cache_type == 0U {
    encode_hard_link(
      name,
      item.object_header_address,
      link.creation_order,
      track_order,
      address_size=offset_size,
    )
  } else if item.cache_type == 1U || item.cache_type == 2U {
    match item.link_target {
      Some(target) =>
        encode_soft_link(name, target, link.creation_order, track_order)
      None => raise InvalidField("soft link has no target")
    }
  } else if item.cache_type == 64U {
    match (item.link_target, item.external_path) {
      (Some(target), Some(path)) =>
        encode_external_link(
          name,
          target,
          path,
          link.creation_order,
          track_order,
        )
      _ => raise InvalidField("external link has incomplete target information")
    }
  } else {
    raise InvalidField(
      "unsupported link type for editing: " + item.cache_type.to_string(),
    )
  }
}

///|
fn encode_editable_compact_link_info(
  offset_size : Int32,
  track_order : Bool,
  max_creation_order : UInt64,
) -> Bytes {
  let undefined = heap_undefined_address(offset_size)
  let buffer = @buffer.Buffer::Buffer(size_hint=2 + 8 + 2 * offset_size)
  buffer.write_byte(b'\x00')
  buffer.write_byte(if track_order { b'\x01' } else { b'\x00' })
  if track_order {
    buffer.write_uint64_le(max_creation_order)
  }
  write_uint_width(buffer, undefined, offset_size)
  write_uint_width(buffer, undefined, offset_size)
  buffer.to_bytes()
}

///|
fn encode_editable_dense_links(
  entries : Array[WritableDenseHeapEntry],
  heap_address : UInt64,
  offset_size : Int32,
  length_size : Int32,
  metadata_filters~ : Array[Filter] = [],
  encoded_filters~ : Bytes = b"",
) -> (Bytes, Bytes, Bytes) raise Hdf5Error {
  guard writer_dense_entries_fit(entries) else {
    raise InvalidField("dense link metadata is not available")
  }
  writer_sort_dense_entries(entries)
  let (heap_bytes, ids) = encode_writer_dense_heap_storage(
    entries,
    heap_address,
    7,
    512,
    32,
    offset_size~,
    length_size~,
    metadata_filters~,
    encoded_filters~,
  )
  let index_address = heap_address + heap_bytes.length().to_uint64()
  let records : Array[Bytes] = []
  for index in 0..<entries.length() {
    let record = @buffer.Buffer::Buffer(size_hint=11)
    record.write_uint_le(entries[index].hash)
    writer_write_bytes(record, ids[index])
    records.push(record.to_bytes())
  }
  let order_indices : Array[Int32] = []
  for index in 0..<entries.length() {
    order_indices.push(index)
  }
  order_indices.sort_by((left, right) => {
    let left_order = entries[left].creation_order
    let right_order = entries[right].creation_order
    if left_order < right_order {
      -1
    } else if left_order > right_order {
      1
    } else if entries[left].name < entries[right].name {
      -1
    } else if entries[left].name > entries[right].name {
      1
    } else {
      0
    }
  })
  let order_records : Array[Bytes] = []
  for index in order_indices {
    let record = @buffer.Buffer::Buffer(size_hint=15)
    record.write_uint64_le(entries[index].creation_order)
    writer_write_bytes(record, ids[index])
    order_records.push(record.to_bytes())
  }
  let index_bytes = encode_writer_metadata_btree(
    5,
    writer_dense_btree_node_size,
    11,
    records,
    index_address,
    offset_size~,
    length_size~,
  )
  let order_index_address = index_address + index_bytes.length().to_uint64()
  let order_bytes = encode_writer_metadata_btree(
    6,
    writer_dense_btree_node_size,
    15,
    order_records,
    order_index_address,
    offset_size~,
    length_size~,
  )
  (heap_bytes, index_bytes, order_bytes)
}

///|
fn Hdf5File::write_rebuilt_links(
  self : Hdf5File,
  address : UInt64,
  header : ObjectHeader,
  links : Array[EditableLink],
  tail : @buffer.Buffer,
) -> Bytes raise Hdf5Error {
  guard links.length() <= 65535 else {
    raise InvalidArgument("too many links for one group")
  }
  let offset_size = self.superblock.offset_size
  let length_size = self.superblock.length_size
  let (existing_tracking, _, _, _) = editable_link_storage(header, offset_size)
  let metadata_filters = self.editable_dense_link_filters(header)
  let mut max_creation_order = 0UL
  for link in links {
    if link.creation_order > max_creation_order {
      max_creation_order = link.creation_order
    }
  }
  let dense = links.length() > writer_dense_threshold
  let symbol_table = header.first(msg_symbol_table) is Some(_)
  let tracking = existing_tracking || dense || symbol_table
  let new_messages : Array[RebuiltObjectMessage] = []
  if dense {
    let entries : Array[WritableDenseHeapEntry] = []
    for link in links {
      entries.push({
        name: link.link.name,
        hash: writer_name_hash(link.link.name),
        data: encode_editable_link(link, link.link.name, true, offset_size),
        creation_order: link.creation_order,
        flags: b'\x00',
        shared_id: None,
      })
    }
    let heap_offset = next_aligned_tail_start(
      self.bytes.length(),
      tail.length(),
      "append dense link metadata",
    )
    let heap_address = file_offset_to_stored_address(
      heap_offset,
      self.superblock.base_address,
    )
    let (heap_bytes, index_bytes, order_bytes) = encode_editable_dense_links(
      entries,
      heap_address,
      offset_size,
      length_size,
      metadata_filters=metadata_filters.filters,
      encoded_filters=metadata_filters.encoded,
    )
    let start = append_contiguous_dense_metadata(
      tail,
      self.bytes.length(),
      heap_bytes,
      index_bytes,
      order_bytes,
    )
    guard start == heap_offset else {
      raise InvalidField("dense link metadata address changed")
    }
    let index_address = heap_address + heap_bytes.length().to_uint64()
    let order_address = index_address + index_bytes.length().to_uint64()
    new_messages.push({
      msg_type: msg_link_info,
      flags: b'\x00',
      data: encode_dense_storage_info(
        true,
        max_creation_order,
        heap_address,
        index_address,
        order_address,
        offset_size~,
      ),
      creation_order: 0UL,
    })
  } else {
    new_messages.push({
      msg_type: msg_link_info,
      flags: b'\x00',
      data: encode_editable_compact_link_info(
        offset_size, tracking, max_creation_order,
      ),
      creation_order: 0UL,
    })
    for link in links {
      new_messages.push({
        msg_type: msg_link,
        flags: b'\x00',
        data: encode_editable_link(link, link.link.name, tracking, offset_size),
        creation_order: link.creation_order,
      })
    }
  }
  self.write_rebuilt_object_messages(
    address,
    header,
    if symbol_table {
      [
        _msg_nil, msg_object_continuation, msg_symbol_table, msg_link_info, msg_link,
      ]
    } else {
      [_msg_nil, msg_object_continuation, msg_link_info, msg_link]
    },
    new_messages,
    tracking,
    tail,
  )
}

///|
fn Hdf5File::editable_group(
  self : Hdf5File,
  path : String,
) -> (UInt64, ObjectHeader, Array[EditableLink]) raise Hdf5Error {
  guard self.source is None else {
    raise InvalidArgument("link editing requires a bytes-backed Hdf5File")
  }
  let node = self.resolve(path)
  let header = self.object_header_at(node.address)
  guard object_kind(header) == Group else {
    raise InvalidArgument("link parent is not a group: " + path)
  }
  (node.address, header, self.editable_links(header))
}

///|
/// Adds a hard link from a group to an existing object in the same file.
pub fn Hdf5File::add_hard_link(
  self : Hdf5File,
  parent_path : String,
  name : String,
  target_path : String,
) -> Bytes raise Hdf5Error {
  validate_editable_link_name(name)
  let (address, header, links) = self.editable_group(parent_path)
  guard editable_link_index(links, name) is None else {
    raise InvalidArgument("link name already exists: " + name)
  }
  let target = self.resolve(target_path)
  let mut creation_order = 0UL
  for link in links {
    if link.creation_order >= creation_order {
      creation_order = link.creation_order + 1UL
    }
  }
  links.push({
    link: {
      name,
      object_header_address: target.address,
      cache_type: 0U,
      link_target: None,
      external_path: None,
    },
    creation_order,
  })
  self.write_rebuilt_links(address, header, links, @buffer.Buffer::Buffer())
}

///|
/// Adds a soft link whose target is interpreted relative to the parent group.
pub fn Hdf5File::add_soft_link(
  self : Hdf5File,
  parent_path : String,
  name : String,
  target : String,
) -> Bytes raise Hdf5Error {
  validate_editable_link_name(name)
  validate_editable_soft_target(target)
  let (address, header, links) = self.editable_group(parent_path)
  guard editable_link_index(links, name) is None else {
    raise InvalidArgument("link name already exists: " + name)
  }
  let mut creation_order = 0UL
  for link in links {
    if link.creation_order >= creation_order {
      creation_order = link.creation_order + 1UL
    }
  }
  links.push({
    link: {
      name,
      object_header_address: 0UL,
      cache_type: 1U,
      link_target: Some(target),
      external_path: None,
    },
    creation_order,
  })
  self.write_rebuilt_links(address, header, links, @buffer.Buffer::Buffer())
}

///|
/// Adds an external link to another HDF5 file.
pub fn Hdf5File::add_external_link(
  self : Hdf5File,
  parent_path : String,
  name : String,
  target_file : String,
  target_path : String,
) -> Bytes raise Hdf5Error {
  validate_editable_link_name(name)
  validate_editable_external_target(target_file, target_path)
  let (address, header, links) = self.editable_group(parent_path)
  guard editable_link_index(links, name) is None else {
    raise InvalidArgument("link name already exists: " + name)
  }
  let mut creation_order = 0UL
  for link in links {
    if link.creation_order >= creation_order {
      creation_order = link.creation_order + 1UL
    }
  }
  links.push({
    link: {
      name,
      object_header_address: 0UL,
      cache_type: 64U,
      link_target: Some(target_file),
      external_path: Some(target_path),
    },
    creation_order,
  })
  self.write_rebuilt_links(address, header, links, @buffer.Buffer::Buffer())
}

///|
/// Renames an existing hard, soft or external link while preserving its target
/// and creation order.
pub fn Hdf5File::rename_link(
  self : Hdf5File,
  parent_path : String,
  old_name : String,
  new_name : String,
) -> Bytes raise Hdf5Error {
  validate_editable_link_name(new_name)
  let (address, header, links) = self.editable_group(parent_path)
  if old_name == new_name {
    return self.bytes
  }
  let index = match editable_link_index(links, old_name) {
    Some(index) => index
    None => raise InvalidArgument("link does not exist: " + old_name)
  }
  guard editable_link_index(links, new_name) is None else {
    raise InvalidArgument("link name already exists: " + new_name)
  }
  links[index] = {
    link: { ..links[index].link, name: new_name, },
    creation_order: links[index].creation_order,
  }
  self.write_rebuilt_links(address, header, links, @buffer.Buffer::Buffer())
}

///|
/// Deletes a link and rebuilds the group's compact or dense link storage.
pub fn Hdf5File::delete_link(
  self : Hdf5File,
  parent_path : String,
  name : String,
) -> Bytes raise Hdf5Error {
  let (address, header, links) = self.editable_group(parent_path)
  let index = match editable_link_index(links, name) {
    Some(index) => index
    None => raise InvalidArgument("link does not exist: " + name)
  }
  let rebuilt : Array[EditableLink] = []
  for position in 0..<links.length() {
    if position != index {
      rebuilt.push(links[position])
    }
  }
  self.write_rebuilt_links(address, header, rebuilt, @buffer.Buffer::Buffer())
}

///|
/// Deletes an object by removing every hard link that names it.
///
/// HDF5 has no separate object-deletion operation: an object becomes
/// unreachable when its final hard link is removed. This function also removes
/// hard links in other groups so aliases cannot keep the object reachable.
/// Soft and external links are left untouched and may subsequently dangle. The
/// root group cannot be deleted. Metadata and raw data from the deleted object
/// remain in the append-only file until free-space reclamation is available.
pub fn Hdf5File::delete_object(
  self : Hdf5File,
  path : String,
) -> Bytes raise Hdf5Error {
  guard self.source is None else {
    raise InvalidArgument("object deletion requires a bytes-backed Hdf5File")
  }
  let target = self.resolve(path)
  guard target.address != self.root_address() else {
    raise InvalidArgument("the root group cannot be deleted")
  }
  let group_addresses = [self.root_address()]
  for node in self.walk() {
    if node.kind == Group && !group_addresses.contains(node.address) {
      group_addresses.push(node.address)
    }
  }
  let mut bytes = self.bytes
  let mut removed = false
  for address in group_addresses {
    let file = Hdf5File::parse(bytes)
    let header = file.object_header_at(address)
    guard object_kind(header) == Group else { continue }
    let links = file.editable_links(header)
    let rebuilt : Array[EditableLink] = []
    for link in links {
      if link.link.cache_type == 0U &&
        link.link.object_header_address == target.address {
        removed = true
      } else {
        rebuilt.push(link)
      }
    }
    if rebuilt.length() != links.length() {
      bytes = file.write_rebuilt_links(
        address,
        header,
        rebuilt,
        @buffer.Buffer::Buffer(),
      )
    }
  }
  guard removed else {
    raise InvalidArgument("object has no removable hard link: " + path)
  }
  bytes
}
