from tinygrad import Tensor, dtypes
import numpy as np

# Constants for spatial hashing
HASH_TABLE_SIZE = 8192  # Power of 2 for better distribution
MAX_BODIES_PER_CELL = 32  # Maximum bodies that can occupy a single cell
CELL_SIZE = 2.0  # Size of each spatial hash cell
BIAS = 1000.0  # Offset to ensure positive coordinates

# Large prime numbers for hash function
PRIME_X = 73856093
PRIME_Y = 19349663
PRIME_Z = 83492791


def find_candidate_pairs(x_pred: Tensor, shape_type: Tensor, shape_params: Tensor, 
                        cell_size: float = CELL_SIZE, table_size: int = HASH_TABLE_SIZE,
                        max_bodies_per_cell: int = MAX_BODIES_PER_CELL) -> Tensor:
    B, N, _ = x_pred.shape
    
    # 1. Compute cell IDs
    cell_ids = compute_cell_ids(x_pred, cell_size)
    
    # 2. Compute hash keys
    hash_keys = compute_hash_keys(cell_ids, table_size)
    
    # 3. Build hash table
    hash_table, cell_occupancy = build_hash_table(hash_keys, B, N, table_size, max_bodies_per_cell)
    
    # 4. Generate candidate pairs
    pairs = generate_pairs(hash_keys, hash_table, cell_occupancy, B, N, table_size, max_bodies_per_cell)
    
    return pairs


def compute_cell_ids(x_pred: Tensor, cell_size: float) -> Tensor:
    # Add bias to ensure positive coordinates
    biased_pos = x_pred + BIAS
    
    # Compute cell coordinates
    cell_ids = (biased_pos / cell_size).floor().cast(dtype=dtypes.int32)
    
    return cell_ids


def compute_hash_keys(cell_ids: Tensor, table_size: int) -> Tensor:
    # Extract x, y, z components
    cx = cell_ids[..., 0]
    cy = cell_ids[..., 1]
    cz = cell_ids[..., 2]
    
    # Compute hash using large primes
    # key = (cx * p1) ^ (cy * p2) ^ (cz * p3) % table_size
    hash_keys = ((cx * PRIME_X) ^ (cy * PRIME_Y) ^ (cz * PRIME_Z)) % table_size
    
    return hash_keys


def build_hash_table(hash_keys: Tensor, B: int, N: int, table_size: int, max_bodies_per_cell: int) -> tuple[Tensor, Tensor]:
    # Initialize hash table with sentinel value -1
    hash_table = Tensor.full((B, table_size, max_bodies_per_cell), -1, dtype=dtypes.int32)
    
    # Simple implementation: insert bodies one by one into their hash buckets
    # This is not the most efficient but works without scatter operations
    
    # Process each batch and body
    hash_table_np = np.full((B, table_size, max_bodies_per_cell), -1, dtype=np.int32)
    cell_count = np.zeros((B, table_size), dtype=np.int32)
    
    hash_keys_np = hash_keys.numpy()
    
    for b in range(B):
        for n in range(N):
            key = int(hash_keys_np[b, n])
            if key >= 0 and key < table_size:
                # Find next available slot in this cell
                slot = cell_count[b, key]
                if slot < max_bodies_per_cell:
                    hash_table_np[b, key, slot] = n
                    cell_count[b, key] += 1
    
    # Convert back to tensors
    hash_table = Tensor(hash_table_np, dtype=dtypes.int32)
    cell_occupancy = Tensor(cell_count, dtype=dtypes.int32)
    
    return hash_table, cell_occupancy


def generate_pairs(hash_keys: Tensor, hash_table: Tensor, cell_occupancy: Tensor,
                  B: int, N: int, table_size: int, max_bodies_per_cell: int) -> Tensor:
    # For simplicity, we'll check all bodies in the same cell
    # A full implementation would check 27 neighboring cells
    
    # Collect all pairs within each cell
    pairs_list = []
    
    # Process each batch
    for b in range(B):
        # For each cell in the hash table
        for cell_idx in range(table_size):
            # Get bodies in this cell
            bodies_in_cell = hash_table[b, cell_idx]  # (max_bodies_per_cell,)
            
            # Filter out sentinel values by converting to numpy first
            bodies_np = bodies_in_cell.numpy()
            valid_bodies = bodies_np[bodies_np >= 0]
            
            # Generate all pairs within this cell
            for i in range(len(valid_bodies)):
                for j in range(i + 1, len(valid_bodies)):
                    pairs_list.append([int(valid_bodies[i]), int(valid_bodies[j])])
    
    # Convert to tensor
    if len(pairs_list) > 0:
        pairs = Tensor(pairs_list, dtype=dtypes.int32)
    else:
        pairs = Tensor.zeros((0, 2), dtype=dtypes.int32)
    
    return pairs


def uniform_spatial_hash(x: Tensor, shape_type: Tensor, shape_params: Tensor) -> Tensor:
    # Add batch dimension for compatibility
    x_batched = x.unsqueeze(0)  # (1, N, 3)
    shape_type_batched = shape_type.unsqueeze(0)  # (1, N)
    shape_params_batched = shape_params.unsqueeze(0)  # (1, N, 3)
    
    # Call main function
    pairs = find_candidate_pairs(x_batched, shape_type_batched, shape_params_batched)
    
    return pairs