from tinygrad import Tensor, dtypes

HASH_TABLE_SIZE = 8192
MAX_BODIES_PER_CELL = 32
CELL_SIZE = 2.0
BIAS = 1000.0

PRIME_X = 73856093
PRIME_Y = 19349663
PRIME_Z = 83492791


def uniform_spatial_hash(x: Tensor, shape_type: Tensor, shape_params: Tensor) -> Tensor:
    N = x.shape[0]
    
    biased_pos = x + BIAS
    
    cell_ids = (biased_pos / CELL_SIZE).floor().cast(dtypes.int32)
    
    cx = cell_ids[:, 0]
    cy = cell_ids[:, 1]
    cz = cell_ids[:, 2]
    hash_keys = ((cx * PRIME_X) ^ (cy * PRIME_Y) ^ (cz * PRIME_Z)) % HASH_TABLE_SIZE
    
    i_indices = Tensor.arange(N).unsqueeze(1).expand(N, N)
    j_indices = Tensor.arange(N).unsqueeze(0).expand(N, N)
    
    all_pairs = Tensor.stack(i_indices.reshape(-1), j_indices.reshape(-1), dim=-1)
    
    hash_i = hash_keys.gather(0, all_pairs[:, 0])
    hash_j = hash_keys.gather(0, all_pairs[:, 1])
    
    cell_i = cell_ids[all_pairs[:, 0]]
    cell_j = cell_ids[all_pairs[:, 1]]
    
    cell_diff = (cell_i - cell_j).abs()
    adjacent_mask = (cell_diff <= 1).all(axis=-1)
    
    not_self_mask = all_pairs[:, 0] != all_pairs[:, 1]
    
    ordered_mask = all_pairs[:, 0] < all_pairs[:, 1]
    
    valid_mask = not_self_mask & ordered_mask & adjacent_mask
    
    invalid_value = -1
    masked_pairs = valid_mask.unsqueeze(-1).expand(-1, 2).where(all_pairs, invalid_value)
    
    return masked_pairs


def compute_cell_ids(x_pred: Tensor, cell_size: float) -> Tensor:
    biased_pos = x_pred + BIAS
    cell_ids = (biased_pos / cell_size).floor().cast(dtype=dtypes.int32)
    return cell_ids


def compute_hash_keys(cell_ids: Tensor, table_size: int) -> Tensor:
    cx = cell_ids[..., 0]
    cy = cell_ids[..., 1]
    cz = cell_ids[..., 2]
    hash_keys = ((cx * PRIME_X) ^ (cy * PRIME_Y) ^ (cz * PRIME_Z)) % table_size
    return hash_keys


def build_hash_table(hash_keys: Tensor, B: int, N: int, table_size: int, max_bodies_per_cell: int) -> tuple[Tensor, Tensor]:
    hash_table = Tensor.full((B, table_size, max_bodies_per_cell), -1, dtype=dtypes.int32)
    cell_occupancy = Tensor.zeros((B, table_size), dtype=dtypes.int32)
    return hash_table, cell_occupancy


def generate_pairs(cell_ids: Tensor, hash_keys: Tensor, hash_table: Tensor, B: int, N: int, 
                  table_size: int, max_bodies_per_cell: int) -> Tensor:
    return Tensor.zeros((0, 2), dtype=dtypes.int32)


def find_candidate_pairs(x_pred: Tensor, shape_type: Tensor, shape_params: Tensor, 
                        cell_size: float = CELL_SIZE, table_size: int = HASH_TABLE_SIZE,
                        max_bodies_per_cell: int = MAX_BODIES_PER_CELL) -> Tensor:
    B, N, _ = x_pred.shape
    return uniform_spatial_hash(x_pred[0], shape_type[0], shape_params[0])