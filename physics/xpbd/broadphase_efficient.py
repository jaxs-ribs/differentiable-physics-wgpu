from tinygrad import Tensor, dtypes
from .broadphase_consts import *


def compute_aabb_from_shape(x: Tensor, shape_type: Tensor, shape_params: Tensor) -> tuple[Tensor, Tensor]:
    """Compute axis-aligned bounding boxes for each body.
    
    Returns:
        aabb_min: (N, 3) minimum corners
        aabb_max: (N, 3) maximum corners
    """
    N = x.shape[0]
    
    # For now, use shape_params[0] as radius for all shapes
    # TODO: Implement per-shape-type AABB computation
    radius = shape_params[:, 0].unsqueeze(-1)
    
    aabb_min = x - radius
    aabb_max = x + radius
    
    return aabb_min, aabb_max


def compute_adaptive_cell_size(aabb_min: Tensor, aabb_max: Tensor) -> float:
    """Compute cell size based on largest AABB dimension."""
    aabb_sizes = (aabb_max - aabb_min).max(axis=-1)
    max_aabb_size = aabb_sizes.max().numpy()
    
    cell_size = max_aabb_size * CELL_SIZE_MULTIPLIER
    cell_size = max(MIN_CELL_SIZE, min(MAX_CELL_SIZE, cell_size))
    
    return cell_size


def spatial_hash(cell_ids: Tensor, table_size: int) -> Tensor:
    """Compute spatial hash from cell coordinates."""
    cx = cell_ids[..., 0]
    cy = cell_ids[..., 1]
    cz = cell_ids[..., 2]
    hash_keys = ((cx * PRIME_X) ^ (cy * PRIME_Y) ^ (cz * PRIME_Z)) % table_size
    return hash_keys


def build_count_table(hash_keys: Tensor, table_size: int) -> Tensor:
    """Count bodies per hash bucket using scatter_add."""
    N = hash_keys.shape[0]
    
    # Create count array
    counts = Tensor.zeros(table_size, dtype=dtypes.int32)
    ones = Tensor.ones(N, dtype=dtypes.int32)
    
    # Scatter add to count bodies per bucket
    # This simulates atomic addition
    counts = counts.scatter_add(0, hash_keys, ones)
    
    return counts


def compute_prefix_sum(counts: Tensor) -> Tensor:
    """Compute exclusive prefix sum for bucket offsets."""
    # Cumsum gives inclusive prefix sum, shift for exclusive
    prefix = counts.cumsum(axis=0)
    # Shift right and prepend 0
    prefix_exclusive = Tensor.cat([Tensor.zeros(1, dtype=dtypes.int32), prefix[:-1]], dim=0)
    return prefix_exclusive


def sort_by_hash(body_indices: Tensor, hash_keys: Tensor) -> tuple[Tensor, Tensor]:
    """Sort body indices by their hash keys."""
    # Get sort indices
    sort_indices = hash_keys.argsort()
    
    sorted_bodies = body_indices.gather(0, sort_indices)
    sorted_hashes = hash_keys.gather(0, sort_indices)
    
    return sorted_bodies, sorted_hashes


def generate_neighbor_pairs_tensor(cell_ids: Tensor, hash_keys: Tensor, 
                                 table_size: int, max_pairs_per_body: int = 64) -> Tensor:
    """Generate candidate pairs using pure tensor operations.
    
    This version avoids loops and uses masking for efficiency.
    """
    N = cell_ids.shape[0]
    
    # Create all possible neighbor offsets as a tensor
    neighbor_offsets = Tensor(NEIGHBOR_OFFSETS, dtype=dtypes.int32)  # (27, 3)
    
    # Expand cell_ids to check all neighbors for each body
    # cell_ids: (N, 3) -> (N, 27, 3)
    expanded_cells = cell_ids.unsqueeze(1) + neighbor_offsets.unsqueeze(0)
    
    # Compute hash for all neighbor cells: (N, 27)
    neighbor_hashes = spatial_hash(expanded_cells.reshape(-1, 3), table_size).reshape(N, 27)
    
    # Create masks for valid pairs
    # Compare each body's neighbor hashes with all bodies' hashes
    # hash_keys: (N,) -> (N, 1, 1)
    # neighbor_hashes: (N, 27) -> (1, N, 27)
    hash_match = hash_keys.unsqueeze(1).unsqueeze(1) == neighbor_hashes.unsqueeze(0)
    
    # Get indices where hashes match
    i_indices = Tensor.arange(N).unsqueeze(1).unsqueeze(2).expand(N, N, 27)
    j_indices = Tensor.arange(N).unsqueeze(0).unsqueeze(2).expand(N, N, 27)
    
    # Flatten and filter
    valid_mask = hash_match & (i_indices < j_indices)  # Only keep i < j to avoid duplicates
    
    # Extract valid pairs
    valid_indices = valid_mask.nonzero()
    if valid_indices.shape[0] > 0:
        pairs = Tensor.stack([
            i_indices[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]],
            j_indices[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]]
        ], dim=-1)
    else:
        pairs = Tensor.zeros((0, 2), dtype=dtypes.int32)
    
    return pairs


def efficient_spatial_hash(x: Tensor, shape_type: Tensor, shape_params: Tensor,
                          cell_size: float = None, table_size: int = HASH_TABLE_SIZE) -> Tensor:
    """O(N) broad-phase collision detection using spatial hashing."""
    N = x.shape[0]
    
    # Compute AABBs and adaptive cell size if needed
    aabb_min, aabb_max = compute_aabb_from_shape(x, shape_type, shape_params)
    if cell_size is None:
        cell_size = compute_adaptive_cell_size(aabb_min, aabb_max)
    
    # Compute cell IDs (no bias for now - TODO: handle negative coords properly)
    cell_ids = (x / cell_size).floor().cast(dtypes.int32)
    
    # Compute hash keys
    hash_keys = spatial_hash(cell_ids, table_size)
    
    # Build count table
    counts = build_count_table(hash_keys, table_size)
    
    # Compute prefix sum for offsets
    prefix_offsets = compute_prefix_sum(counts)
    
    # Sort bodies by hash
    body_indices = Tensor.arange(N, dtype=dtypes.int32)
    sorted_bodies, sorted_hashes = sort_by_hash(body_indices, hash_keys)
    
    # Generate neighbor pairs using tensor operations
    candidate_pairs = generate_neighbor_pairs_tensor(cell_ids, hash_keys, table_size)
    
    return candidate_pairs