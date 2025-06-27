"""Fully differentiable broadphase collision detection using tensor operations.

This module provides a pure tensor implementation of broadphase collision detection
that checks all pairs of bodies for AABB overlaps. Unlike the sweep-and-prune
algorithm, this approach is O(n²) but fully differentiable and GPU-parallelizable.
"""
from tinygrad import Tensor, dtypes
from .types import BodySchema, ShapeType
from .math_utils import quat_to_rotmat

def _generate_all_pairs(num_bodies: int) -> Tensor:
  """Generate all unique pairs (i, j) where i < j using pure tensor operations.
  
  Args:
    num_bodies: Number of bodies in the system
    
  Returns:
    Tensor of shape (num_pairs, 2) with all unique pair indices
  """
  if num_bodies < 2:
    return Tensor.zeros((0, 2), dtype=dtypes.int32)
  
  # Calculate number of unique pairs
  n_pairs = (num_bodies * (num_bodies - 1)) // 2
  
  # Create a range for all pair indices
  pair_idx = Tensor.arange(n_pairs, dtype=dtypes.int32)
  
  # Calculate i and j from the pair index using the formula:
  # For pair index k, we need to find i,j such that k = i*(2n-i-1)/2 + (j-i-1)
  # This can be solved to get i = floor((2n-1 - sqrt((2n-1)^2 - 8k))/2)
  n = num_bodies
  temp = (2*n - 1) - ((2*n - 1)**2 - 8*pair_idx).sqrt()
  i_indices = (temp / 2).cast(dtypes.int32)
  
  # Calculate j from i and pair_idx
  # j = k - i*(2n-i-1)/2 + i + 1
  j_indices = pair_idx - i_indices * (2*n - i_indices - 1) // 2 + i_indices + 1
  
  # Stack to create pairs
  pairs = i_indices.unsqueeze(1).cat(j_indices.unsqueeze(1), dim=1)
  
  return pairs

def differentiable_broadphase(bodies: Tensor) -> tuple[Tensor, Tensor]:
  """Differentiable all-pairs broadphase collision detection.
  
  This function is fully differentiable and uses only tensor operations.
  
  Args:
    bodies: State tensor of shape (N, NUM_PROPERTIES)
    
  Returns:
    Tuple of:
    - pair_indices: Tensor of shape (N*(N-1)/2, 2) with all pair indices
    - collision_mask: Boolean tensor of shape (N*(N-1)/2,) indicating overlaps
  """
  n_bodies = bodies.shape[0]
  
  # Generate all unique pairs using pure tensor operations
  pair_indices = _generate_all_pairs(n_bodies)
  n_pairs = pair_indices.shape[0]
  
  if n_pairs == 0:
    return pair_indices, Tensor.zeros((0,)).cast(dtypes.bool)
  
  # Use gather to extract data for all pairs
  # Gather indices for body A and body B
  indices_a = pair_indices[:, 0]
  indices_b = pair_indices[:, 1]
  
  # Gather full body data for each pair
  # We need to expand indices to match the shape of bodies
  indices_a_expanded = indices_a.unsqueeze(1).expand(-1, bodies.shape[1])
  indices_b_expanded = indices_b.unsqueeze(1).expand(-1, bodies.shape[1])
  bodies_a = bodies.gather(0, indices_a_expanded)  # (n_pairs, NUM_PROPERTIES)
  bodies_b = bodies.gather(0, indices_b_expanded)  # (n_pairs, NUM_PROPERTIES)
  
  # Extract specific properties
  pos_a = bodies_a[:, BodySchema.POS_X:BodySchema.POS_Z+1]
  pos_b = bodies_b[:, BodySchema.POS_X:BodySchema.POS_Z+1]
  quat_a = bodies_a[:, BodySchema.QUAT_W:BodySchema.QUAT_Z+1]
  quat_b = bodies_b[:, BodySchema.QUAT_W:BodySchema.QUAT_Z+1]
  shape_type_a = bodies_a[:, BodySchema.SHAPE_TYPE]
  shape_type_b = bodies_b[:, BodySchema.SHAPE_TYPE]
  shape_params_a = bodies_a[:, BodySchema.SHAPE_PARAM_1:BodySchema.SHAPE_PARAM_3+1]
  shape_params_b = bodies_b[:, BodySchema.SHAPE_PARAM_1:BodySchema.SHAPE_PARAM_3+1]
  
  # Compute rotation matrices
  rot_a = quat_to_rotmat(quat_a)
  rot_b = quat_to_rotmat(quat_b)
  
  # Initialize AABB bounds
  min_a = pos_a.clone()
  max_a = pos_a.clone()
  min_b = pos_b.clone()
  max_b = pos_b.clone()
  
  # Handle spheres  
  sphere_mask_a = (shape_type_a == ShapeType.SPHERE).float().reshape(-1, 1)
  sphere_mask_b = (shape_type_b == ShapeType.SPHERE).float().reshape(-1, 1)
  
  radius_a = shape_params_a[:, 0:1]
  radius_b = shape_params_b[:, 0:1]
  
  min_a = min_a - sphere_mask_a * radius_a
  max_a = max_a + sphere_mask_a * radius_a
  min_b = min_b - sphere_mask_b * radius_b
  max_b = max_b + sphere_mask_b * radius_b
  
  # Handle boxes
  box_mask_a = (shape_type_a == ShapeType.BOX).float().reshape(-1, 1)
  box_mask_b = (shape_type_b == ShapeType.BOX).float().reshape(-1, 1)
  
  # Box corners in local space (8 corners of a unit cube)
  corners_data = [
    -1.0, -1.0, -1.0,  # corner 0
    -1.0, -1.0,  1.0,  # corner 1
    -1.0,  1.0, -1.0,  # corner 2
    -1.0,  1.0,  1.0,  # corner 3
     1.0, -1.0, -1.0,  # corner 4
     1.0, -1.0,  1.0,  # corner 5
     1.0,  1.0, -1.0,  # corner 6
     1.0,  1.0,  1.0   # corner 7
  ]
  corners = Tensor(corners_data, dtype=dtypes.float32).reshape(8, 3)
  
  # For all pairs, compute box bounds (will only apply where box_mask is true)
  # Scale corners by half extents
  scaled_corners_a = corners.unsqueeze(0) * shape_params_a.unsqueeze(1)  # (n_pairs, 8, 3)
  scaled_corners_b = corners.unsqueeze(0) * shape_params_b.unsqueeze(1)
  
  # Rotate corners: (n_pairs, 3, 3) @ (n_pairs, 3, 8) -> (n_pairs, 3, 8)
  rotated_corners_a = rot_a @ scaled_corners_a.transpose(-1, -2)
  rotated_corners_b = rot_b @ scaled_corners_b.transpose(-1, -2)
  
  # Translate to world space
  world_corners_a = pos_a.unsqueeze(-1) + rotated_corners_a  # (n_pairs, 3, 8)
  world_corners_b = pos_b.unsqueeze(-1) + rotated_corners_b
  
  # Find min/max across corners
  box_min_a = world_corners_a.min(axis=2)  # (n_pairs, 3)
  box_max_a = world_corners_a.max(axis=2)  # (n_pairs, 3)
  box_min_b = world_corners_b.min(axis=2)
  box_max_b = world_corners_b.max(axis=2)
  
  # Apply box mask
  min_a = min_a * (1 - box_mask_a) + box_min_a * box_mask_a
  max_a = max_a * (1 - box_mask_a) + box_max_a * box_mask_a
  min_b = min_b * (1 - box_mask_b) + box_min_b * box_mask_b
  max_b = max_b * (1 - box_mask_b) + box_max_b * box_mask_b
  
  # Check AABB overlaps using pure tensor operations
  # Check overlap on each axis
  overlap_x = (max_a[:, 0] >= min_b[:, 0]) & (max_b[:, 0] >= min_a[:, 0])
  overlap_y = (max_a[:, 1] >= min_b[:, 1]) & (max_b[:, 1] >= min_a[:, 1])
  overlap_z = (max_a[:, 2] >= min_b[:, 2]) & (max_b[:, 2] >= min_a[:, 2])
  
  # Collision if all axes overlap
  collision_mask = overlap_x & overlap_y & overlap_z
  
  return pair_indices, collision_mask