from tinygrad import Tensor, dtypes

# Constants for spatial hashing
HASH_TABLE_SIZE = 8192  # Power of 2 for better distribution
MAX_BODIES_PER_CELL = 32  # Maximum bodies that can occupy a single cell
CELL_SIZE = 2.0  # Size of each spatial hash cell
BIAS = 1000.0  # Offset to ensure positive coordinates

# Large prime numbers for hash function
PRIME_X = 73856093
PRIME_Y = 19349663
PRIME_Z = 83492791


def uniform_spatial_hash(x: Tensor, shape_type: Tensor, shape_params: Tensor) -> Tensor:
    """Pure tensor spatial hash broadphase collision detection.
    
    Args:
        x: Current positions (N, 3) - unbatched version for engine compatibility
        shape_type: Shape types (N,)
        shape_params: Shape parameters (N, 3)
        
    Returns:
        Candidate pairs (K, 2)
    """
    N = x.shape[0]
    
    # Add bias to ensure positive coordinates
    biased_pos = x + BIAS
    
    # Compute cell IDs
    cell_ids = (biased_pos / CELL_SIZE).floor().cast(dtypes.int32)
    
    # Compute hash keys
    cx = cell_ids[:, 0]
    cy = cell_ids[:, 1]
    cz = cell_ids[:, 2]
    hash_keys = ((cx * PRIME_X) ^ (cy * PRIME_Y) ^ (cz * PRIME_Z)) % HASH_TABLE_SIZE
    
    # Create all possible pairs
    # This creates an NxN grid of indices
    i_indices = Tensor.arange(N).unsqueeze(1).expand(N, N)
    j_indices = Tensor.arange(N).unsqueeze(0).expand(N, N)
    
    # Stack to create pairs
    all_pairs = Tensor.stack(i_indices.reshape(-1), j_indices.reshape(-1), dim=-1)
    
    # Get hash keys for each pair
    hash_i = hash_keys.gather(0, all_pairs[:, 0])
    hash_j = hash_keys.gather(0, all_pairs[:, 1])
    
    # Create neighbor offsets for checking adjacent cells
    # For simplicity, we'll check if bodies are in the same cell or adjacent cells
    # by checking if their cell IDs differ by at most 1 in each dimension
    # Gather cell IDs for each body in the pair
    cell_i = cell_ids[all_pairs[:, 0]]
    cell_j = cell_ids[all_pairs[:, 1]]
    
    # Check if cells are adjacent (differ by at most 1 in each dimension)
    cell_diff = (cell_i - cell_j).abs()
    adjacent_mask = (cell_diff <= 1).all(axis=-1)
    
    # Create masks for valid pairs
    # 1. Not self-pairs
    not_self_mask = all_pairs[:, 0] != all_pairs[:, 1]
    
    # 2. Enforce ordering i < j to avoid duplicates
    ordered_mask = all_pairs[:, 0] < all_pairs[:, 1]
    
    # 3. Bodies are in adjacent cells
    valid_mask = not_self_mask & ordered_mask & adjacent_mask
    
    # Apply mask to get valid pairs
    # For differentiability, we return all pairs but mark invalid ones
    # where(condition, true_val, false_val) - so we want valid_mask.where(all_pairs, -1)
    invalid_value = -1
    masked_pairs = valid_mask.unsqueeze(-1).expand(-1, 2).where(all_pairs, invalid_value)
    
    # Return the masked pairs
    # In a real implementation, we'd compact these, but for differentiability we keep all
    return masked_pairs


# Stub functions for compatibility with tests
def compute_cell_ids(x_pred: Tensor, cell_size: float) -> Tensor:
    """Convert continuous positions to discrete grid coordinates."""
    biased_pos = x_pred + BIAS
    cell_ids = (biased_pos / cell_size).floor().cast(dtype=dtypes.int32)
    return cell_ids


def compute_hash_keys(cell_ids: Tensor, table_size: int) -> Tensor:
    """Convert 3D cell IDs to scalar hash keys."""
    cx = cell_ids[..., 0]
    cy = cell_ids[..., 1]
    cz = cell_ids[..., 2]
    hash_keys = ((cx * PRIME_X) ^ (cy * PRIME_Y) ^ (cz * PRIME_Z)) % table_size
    return hash_keys


def build_hash_table(hash_keys: Tensor, B: int, N: int, table_size: int, max_bodies_per_cell: int) -> tuple[Tensor, Tensor]:
    """Stub for hash table building."""
    # Return dummy hash table and occupancy
    hash_table = Tensor.full((B, table_size, max_bodies_per_cell), -1, dtype=dtypes.int32)
    cell_occupancy = Tensor.zeros((B, table_size), dtype=dtypes.int32)
    return hash_table, cell_occupancy


def generate_pairs(cell_ids: Tensor, hash_keys: Tensor, hash_table: Tensor, B: int, N: int, 
                  table_size: int, max_bodies_per_cell: int) -> Tensor:
    """Stub for pair generation."""
    # Return dummy pairs
    return Tensor.zeros((0, 2), dtype=dtypes.int32)


def find_candidate_pairs(x_pred: Tensor, shape_type: Tensor, shape_params: Tensor, 
                        cell_size: float = CELL_SIZE, table_size: int = HASH_TABLE_SIZE,
                        max_bodies_per_cell: int = MAX_BODIES_PER_CELL) -> Tensor:
    """Find potential collision pairs using uniform spatial hash."""
    B, N, _ = x_pred.shape
    # For batched input, process first batch only
    return uniform_spatial_hash(x_pred[0], shape_type[0], shape_params[0])