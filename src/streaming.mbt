///|
/// One decoded tile from a dataset. `offset` is the tile origin in dataset
/// coordinates and `dimensions` is the edge-clipped tile shape.
#derive(@debug.Debug)
pub(all) struct DatasetChunk {
  offset : Array[UInt64]
  dimensions : Array[UInt64]
  values : DatasetValues
}

///|
pub extend DatasetChunk with @debug.Debug::{to_repr}

///|
/// Iterates over a dataset in caller-selected tiles without materializing the
/// complete dataset. Storage chunks that intersect a tile are decoded at most
/// once per `next` call.
pub struct DatasetChunkIterator {
  priv file : Hdf5File
  priv header : ObjectHeader
  priv metadata : DatasetMetadata
  priv tile_dimensions : Array[UInt64]
  priv mut next_index : Int64
  priv total_tiles : Int64
  priv resolver : Hdf5Resolver?
  priv referring_file : String
}

///|
fn dataset_tile_count(
  dataspace : Dataspace,
  tile_dimensions : Array[UInt64],
) -> Int64 raise Hdf5Error {
  guard dataspace.kind != Null || tile_dimensions.length() == 0 else {
    raise InvalidArgument("null datasets do not accept tile dimensions")
  }
  if dataspace.kind == Null {
    return 0L
  }
  let rank = dataspace.dimensions.length()
  guard tile_dimensions.length() == rank else {
    raise InvalidArgument("tile rank does not match dataset rank")
  }
  let mut tile_elements = 1L
  for axis in 0..<rank {
    let tile = tile_dimensions[axis]
    guard tile > 0UL else {
      raise InvalidArgument("tile dimensions must be positive")
    }
    guard tile <= 0x7FFFFFFFUL else {
      raise InvalidArgument("tile dimension is too large")
    }
    guard tile_elements <= 0x7FFFFFFFFFFFFFFFL / tile.reinterpret_as_int64() else {
      raise InvalidArgument("tile element count overflow")
    }
    tile_elements = tile_elements * tile.reinterpret_as_int64()
  }
  guard tile_elements <= 0x7FFFFFFFL else {
    raise InvalidArgument("tile contains too many elements")
  }
  if rank == 0 || dataspace.dimensions.contains(0UL) {
    return if rank == 0 { 1L } else { 0L }
  }
  let mut tiles = 1L
  for axis in 0..<rank {
    let dimension = dataspace.dimensions[axis]
    let tile = tile_dimensions[axis]
    let mut grid = dimension / tile
    if dimension % tile != 0UL {
      grid = grid + 1UL
    }
    guard grid <= 0x7FFFFFFFFFFFFFFFUL else {
      raise InvalidArgument("tile count overflow")
    }
    let grid_count = grid.reinterpret_as_int64()
    guard tiles <= 0x7FFFFFFFFFFFFFFFL / grid_count else {
      raise InvalidArgument("tile count overflow")
    }
    tiles = tiles * grid_count
  }
  tiles
}

///|
fn make_dataset_chunk_iterator(
  file : Hdf5File,
  header : ObjectHeader,
  metadata : DatasetMetadata,
  tile_dimensions : Array[UInt64],
  resolver : Hdf5Resolver?,
  referring_file : String,
) -> DatasetChunkIterator raise Hdf5Error {
  let total_tiles = dataset_tile_count(metadata.dataspace, tile_dimensions)
  {
    file,
    header,
    metadata,
    tile_dimensions: tile_dimensions.copy(),
    next_index: 0L,
    total_tiles,
    resolver,
    referring_file,
  }
}

///|
/// Creates a bounded-memory iterator over a dataset.
pub fn Hdf5File::chunk_iterator(
  self : Hdf5File,
  path : String,
  tile_dimensions : Array[UInt64],
) -> DatasetChunkIterator raise Hdf5Error {
  let node = self.resolve(path)
  guard node.kind == Dataset else {
    raise InvalidField("object is not a dataset: " + path)
  }
  let header = self.object_header_at(node.address)
  let metadata = read_dataset_metadata(
    header,
    self.superblock.offset_size,
    self.superblock.length_size,
  )
  make_dataset_chunk_iterator(self, header, metadata, tile_dimensions, None, "")
}

///|
fn DatasetChunkIterator::read_selection(
  self : DatasetChunkIterator,
  selection : DatasetSelection,
) -> DatasetSelectionResult raise Hdf5Error {
  if self.header.first(msg_external_data_files) is Some(_) {
    let resolver = match self.resolver {
      Some(resolver) => resolver
      None =>
        raise UnsupportedFeature(
          "external dataset storage requires Hdf5Resolver",
        )
    }
    return resolver.read_external_dataset_selection(
      self.referring_file,
      self.file,
      self.header,
      self.metadata,
      selection,
    )
  }
  let file = self.file
  match file.source {
    Some(source) =>
      read_source_dataset_selection(
        source,
        self.header,
        self.metadata,
        selection,
        file.superblock.offset_size,
        file.superblock.length_size,
        file.superblock.base_address,
      )
    None =>
      read_dataset_selection(
        file.bytes,
        self.header,
        self.metadata,
        selection,
        file.superblock.offset_size,
        file.superblock.length_size,
        file.superblock.base_address,
      )
  }
}

///|
/// Returns the next tile, or `None` after the dataset has been consumed.
/// Callers can stop early by no longer calling `next`.
pub fn DatasetChunkIterator::next(
  self : DatasetChunkIterator,
) -> DatasetChunk? raise Hdf5Error {
  if self.next_index >= self.total_tiles {
    return None
  }
  let tile_index = self.next_index
  self.next_index = self.next_index + 1L
  let rank = self.metadata.dataspace.dimensions.length()
  if rank == 0 {
    let selected = self.read_selection(Hyperslab([], [], []))
    return Some({ offset: [], dimensions: [], values: selected.values, })
  }
  let offset : Array[UInt64] = Array::make(rank, 0UL)
  let dimensions : Array[UInt64] = Array::make(rank, 0UL)
  let mut remaining = tile_index
  let mut axis = rank - 1
  while axis >= 0 {
    let dimension = self.metadata.dataspace.dimensions[axis]
    let tile = self.tile_dimensions[axis]
    let mut grid = dimension / tile
    if dimension % tile != 0UL {
      grid = grid + 1UL
    }
    let grid_count = grid.reinterpret_as_int64()
    let coordinate = remaining % grid_count
    remaining = remaining / grid_count
    let start = coordinate.reinterpret_as_uint64() * tile
    let available = dimension - start
    offset[axis] = start
    dimensions[axis] = if available < tile { available } else { tile }
    axis = axis - 1
  }
  let stride : Array[UInt64] = Array::make(rank, 1UL)
  let selected = self.read_selection(
    Hyperslab(offset.copy(), dimensions.copy(), stride),
  )
  Some({ offset, dimensions, values: selected.values, })
}

///|
/// One page of a dataset selection. `offset` is the zero-based ordinal of the
/// first value inside the original selection; values are returned in selection
/// order as a flat page.
#derive(@debug.Debug)
pub(all) struct DatasetSelectionPage {
  offset : UInt64
  datatype : Datatype
  values : DatasetValues
}

///|
pub extend DatasetSelectionPage with @debug.Debug::{to_repr}

///|
/// Iterates over a dataset selection in bounded-memory pages. Hyperslab
/// ordinals are generated lazily, so creating an iterator for a very large
/// selection does not allocate an index array for the complete result.
pub struct DatasetSelectionIterator {
  priv file : Hdf5File
  priv header : ObjectHeader
  priv metadata : DatasetMetadata
  priv selection : DatasetSelection
  priv page_size : Int32
  priv total : Int64
  priv mut next_offset : Int64
  priv resolver : Hdf5Resolver?
  priv referring_file : String
}

///|
fn copy_dataset_selection(selection : DatasetSelection) -> DatasetSelection {
  match selection {
    Hyperslab(start, count, stride) =>
      Hyperslab(start.copy(), count.copy(), stride.copy())
    Points(points) => {
      let copied : Array[Array[UInt64]] = []
      for point in points {
        copied.push(point.copy())
      }
      Points(copied)
    }
  }
}

///|
fn make_dataset_selection_iterator(
  file : Hdf5File,
  header : ObjectHeader,
  metadata : DatasetMetadata,
  selection : DatasetSelection,
  page_size : Int32,
  resolver : Hdf5Resolver?,
  referring_file : String,
) -> DatasetSelectionIterator raise Hdf5Error {
  guard page_size > 0 else {
    raise InvalidArgument("selection page size must be positive")
  }
  let copied = copy_dataset_selection(selection)
  let total = selection_element_total(metadata.dataspace, copied)
  {
    file,
    header,
    metadata,
    selection: copied,
    page_size,
    total,
    next_offset: 0L,
    resolver,
    referring_file,
  }
}

///|
/// Creates a bounded-memory iterator over an arbitrary dataset selection.
/// Use `next` to retrieve pages of at most `page_size` values.
pub fn Hdf5File::selection_iterator(
  self : Hdf5File,
  path : String,
  selection : DatasetSelection,
  page_size : Int32,
) -> DatasetSelectionIterator raise Hdf5Error {
  let node = self.resolve(path)
  guard node.kind == Dataset else {
    raise InvalidField("object is not a dataset: " + path)
  }
  let header = self.object_header_at(node.address)
  let metadata = read_dataset_metadata(
    header,
    self.superblock.offset_size,
    self.superblock.length_size,
  )
  make_dataset_selection_iterator(
    self,
    header,
    metadata,
    selection,
    page_size,
    None,
    "",
  )
}

///|
/// Creates a bounded-memory selection iterator through the resolver, including
/// contiguous external raw-data storage.
pub fn Hdf5Resolver::selection_iterator(
  self : Hdf5Resolver,
  path : String,
  selection : DatasetSelection,
  page_size : Int32,
) -> DatasetSelectionIterator raise Hdf5Error {
  let node = self.resolve(path)
  guard node.kind == Dataset else {
    raise InvalidField("object is not a dataset: " + path)
  }
  let file = self.open(node.file)
  let header = file.object_header_at(node.address)
  let metadata = read_dataset_metadata(
    header,
    file.superblock.offset_size,
    file.superblock.length_size,
  )
  make_dataset_selection_iterator(
    file,
    header,
    metadata,
    selection,
    page_size,
    Some(self),
    node.file,
  )
}

///|
fn DatasetSelectionIterator::read_page(
  self : DatasetSelectionIterator,
  indices : Array[UInt64],
) -> DatasetSelectionResult raise Hdf5Error {
  let dimensions = [indices.length().to_uint64()]
  if self.header.first(msg_external_data_files) is Some(_) {
    let resolver = match self.resolver {
      Some(resolver) => resolver
      None =>
        raise UnsupportedFeature(
          "external dataset storage requires Hdf5Resolver",
        )
    }
    return resolver.read_external_dataset_indices(
      self.referring_file,
      self.file,
      self.header,
      self.metadata,
      dimensions,
      indices,
    )
  }
  let file = self.file
  match file.source {
    Some(source) =>
      read_source_dataset_indices(
        source,
        self.header,
        self.metadata,
        dimensions,
        indices,
        file.superblock.offset_size,
        file.superblock.length_size,
        file.superblock.base_address,
      )
    None =>
      read_dataset_indices(
        file.bytes,
        self.header,
        self.metadata,
        dimensions,
        indices,
        file.superblock.offset_size,
        file.superblock.length_size,
        file.superblock.base_address,
      )
  }
}

///|
/// Returns the next selection page, or `None` after the selection has been
/// consumed. Callers can stop early without reading the remaining pages.
pub fn DatasetSelectionIterator::next(
  self : DatasetSelectionIterator,
) -> DatasetSelectionPage? raise Hdf5Error {
  if self.next_offset >= self.total {
    return None
  }
  let offset = self.next_offset
  let remaining = self.total - offset
  let page_size = if remaining < self.page_size.to_int64() {
    remaining.to_int()
  } else {
    self.page_size
  }
  let indices : Array[UInt64] = []
  for ordinal in 0..<page_size {
    indices.push(
      selection_index_at(
        self.metadata.dataspace,
        self.selection,
        offset + ordinal.to_int64(),
      ),
    )
  }
  let result = self.read_page(indices)
  self.next_offset = offset + page_size.to_int64()
  Some({
    offset: offset.reinterpret_as_uint64(),
    datatype: result.datatype,
    values: result.values,
  })
}
