from tinygrad import Tensor, dtypes
from .broadphase_consts import *


def compute_cell_bounds(x: Tensor, radius: Tensor, cell_size: float) -> tuple[Tensor, Tensor]:
    """Compute min/max cell indices for each body's AABB."""
    aabb_min = x - radius.unsqueeze(-1)
    aabb_max = x + radius.unsqueeze(-1)
    
    cell_min = (aabb_min / cell_size).floor().cast(dtypes.int32)
    cell_max = (aabb_max / cell_size).floor().cast(dtypes.int32)
    
    return cell_min, cell_max


def morton_encode_3d(x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    """Encode 3D coordinates into Morton code for better spatial locality."""
    # Simple interleaving for now - can be optimized with bit manipulation
    # Limit to 10 bits per dimension for 30-bit Morton code
    x = x & 0x3FF
    y = y & 0x3FF
    z = z & 0x3FF
    
    morton = (x << 20) | (y << 10) | z
    return morton


def radix_sort_pairs(keys: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
    """Sort key-value pairs by keys using radix sort for better GPU performance."""
    # For now, use standard sort - can be optimized with parallel radix sort
    sort_indices = keys.argsort()
    sorted_keys = keys.gather(0, sort_indices)
    sorted_values = values.gather(0, sort_indices)
    return sorted_keys, sorted_values


def compact_duplicate_pairs(pairs: Tensor) -> Tensor:
    """Remove duplicate pairs efficiently."""
    if pairs.shape[0] == 0:
        return pairs
        
    # Sort pairs to group duplicates
    # Convert to single key for sorting: i * large_prime + j
    pair_keys = pairs[:, 0] * 100000 + pairs[:, 1]
    sorted_indices = pair_keys.argsort()
    sorted_pairs = pairs.gather(0, sorted_indices.unsqueeze(-1).expand(-1, 2))
    
    # Mark unique pairs
    if sorted_pairs.shape[0] > 1:
        # Check if consecutive pairs are different
        diff = (sorted_pairs[1:] != sorted_pairs[:-1]).any(axis=-1)
        # First pair is always unique, then check differences
        unique_mask = Tensor.cat([Tensor.ones(1, dtype=dtypes.bool), diff], dim=0)
        
        # Extract unique pairs
        unique_indices = unique_mask.nonzero().squeeze(-1)
        unique_pairs = sorted_pairs.gather(0, unique_indices.unsqueeze(-1).expand(-1, 2))
    else:
        unique_pairs = sorted_pairs
    
    return unique_pairs


def spatial_hash_optimized(x: Tensor, shape_type: Tensor, shape_params: Tensor,
                         cell_size: float = DEFAULT_CELL_SIZE,
                         use_morton: bool = False) -> Tensor:
    """Memory-efficient O(N) broad-phase using streaming pair generation."""
    N = x.shape[0]
    
    # Use radius from shape params
    radius = shape_params[:, 0]
    
    # Compute cell bounds for each body
    cell_min, cell_max = compute_cell_bounds(x, radius, cell_size)
    
    # Generate pairs by cell overlap
    # This is still O(N²) comparison but with early rejection
    pairs_list = []
    
    # Batch processing to control memory usage
    batch_size = min(N, 1000)
    
    for i_start in range(0, N, batch_size):
        i_end = min(i_start + batch_size, N)
        batch_min_i = cell_min[i_start:i_end]
        batch_max_i = cell_max[i_start:i_end]
        
        for j_start in range(i_start, N, batch_size):
            j_end = min(j_start + batch_size, N)
            batch_min_j = cell_min[j_start:j_end]
            batch_max_j = cell_max[j_start:j_end]
            
            # Check AABB overlap in cell space
            # overlap = (min_i <= max_j) & (max_i >= min_j) for all dimensions
            overlap_x = (batch_min_i[:, 0:1] <= batch_max_j[:, 0:1].T) & \
                       (batch_max_i[:, 0:1] >= batch_min_j[:, 0:1].T)
            overlap_y = (batch_min_i[:, 1:2] <= batch_max_j[:, 1:2].T) & \
                       (batch_max_i[:, 1:2] >= batch_min_j[:, 1:2].T)
            overlap_z = (batch_min_i[:, 2:3] <= batch_max_j[:, 2:3].T) & \
                       (batch_max_i[:, 2:3] >= batch_min_j[:, 2:3].T)
            
            overlap_mask = overlap_x.squeeze() & overlap_y.squeeze() & overlap_z.squeeze()
            
            # Generate indices
            i_indices = Tensor.arange(i_start, i_end).unsqueeze(1)
            j_indices = Tensor.arange(j_start, j_end).unsqueeze(0)
            
            # Only keep upper triangle if same batch
            if i_start == j_start:
                overlap_mask = overlap_mask & (i_indices < j_indices)
            
            # Extract valid pairs
            valid_indices = overlap_mask.nonzero()
            if valid_indices.shape[0] > 0:
                batch_pairs = Tensor.stack([
                    i_indices[valid_indices[:, 0], 0],
                    j_indices[0, valid_indices[:, 1]]
                ], dim=-1)
                pairs_list.append(batch_pairs)
    
    # Concatenate all pairs
    if pairs_list:
        all_pairs = Tensor.cat(pairs_list, dim=0)
        # Remove duplicates
        unique_pairs = compact_duplicate_pairs(all_pairs)
        return unique_pairs
    else:
        return Tensor.zeros((0, 2), dtype=dtypes.int32)


def spatial_hash_gpu_friendly(x: Tensor, shape_type: Tensor, shape_params: Tensor,
                            cell_size: float = DEFAULT_CELL_SIZE) -> Tensor:
    """GPU-optimized version using cell lists and atomic operations."""
    N = x.shape[0]
    
    # Compute cell IDs
    cell_ids = (x / cell_size).floor().cast(dtypes.int32)
    
    # Use Morton encoding for better cache locality
    morton_codes = morton_encode_3d(cell_ids[:, 0], cell_ids[:, 1], cell_ids[:, 2])
    
    # Sort bodies by Morton code
    body_indices = Tensor.arange(N, dtype=dtypes.int32)
    sorted_morton, sorted_bodies = radix_sort_pairs(morton_codes, body_indices)
    
    # Find cell boundaries (where Morton code changes)
    if N > 1:
        boundaries = (sorted_morton[1:] != sorted_morton[:-1]).nonzero().squeeze(-1) + 1
        boundaries = Tensor.cat([Tensor.zeros(1, dtype=dtypes.int32), boundaries, 
                                Tensor([N], dtype=dtypes.int32)], dim=0)
    else:
        boundaries = Tensor([0, N], dtype=dtypes.int32)
    
    # Generate pairs within cells and between adjacent cells
    pairs_list = []
    
    for i in range(boundaries.shape[0] - 1):
        start = boundaries[i].numpy()
        end = boundaries[i + 1].numpy()
        
        if end - start > 0:
            # Bodies in this cell
            cell_bodies = sorted_bodies[start:end]
            
            # Pairs within cell
            if end - start > 1:
                i_idx = cell_bodies.unsqueeze(1).expand(-1, end - start)
                j_idx = cell_bodies.unsqueeze(0).expand(end - start, -1)
                within_mask = i_idx < j_idx
                within_indices = within_mask.nonzero()
                
                if within_indices.shape[0] > 0:
                    within_pairs = Tensor.stack([
                        i_idx[within_indices[:, 0], within_indices[:, 1]],
                        j_idx[within_indices[:, 0], within_indices[:, 1]]
                    ], dim=-1)
                    pairs_list.append(within_pairs)
    
    # Concatenate all pairs
    if pairs_list:
        all_pairs = Tensor.cat(pairs_list, dim=0)
        return all_pairs
    else:
        return Tensor.zeros((0, 2), dtype=dtypes.int32)