"""Narrowphase collision detection for exact contact generation.

After broadphase identifies potentially colliding pairs, narrowphase performs
exact geometric tests to determine if bodies are actually colliding and
generates contact points with normals and penetration depths.

This implementation is fully vectorized and JIT-compatible, operating on
all collision pairs simultaneously without Python loops.
"""
from tinygrad import Tensor, dtypes
from .types import BodySchema, ShapeType
from .math_utils import quat_to_rotmat

def narrowphase(bodies: Tensor, pair_indices: Tensor, collision_mask: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
  """Perform exact collision detection on potentially colliding pairs.
  
  This function is fully vectorized and operates on all pairs simultaneously.
  Invalid pairs (where collision_mask is False) will have zeroed outputs.
  
  Args:
    bodies: State tensor of shape (N, NUM_PROPERTIES)
    pair_indices: Tensor of shape (M, 2) with indices of all pairs
    collision_mask: Boolean tensor of shape (M,) indicating which pairs have overlapping AABBs
    
  Returns:
    Tuple of tensors:
    - contact_normals: (M, 3) Normal vectors pointing from body j to body i
    - contact_depths: (M,) Penetration depths
    - contact_points: (M, 3) Contact points in world space
    - contact_mask: (M,) Boolean mask indicating valid contacts
    - pair_indices: (M, 2) Same as input, for convenience
  """
  n_pairs = pair_indices.shape[0]
  
  # If no pairs, return empty tensors
  if n_pairs == 0:
    return (Tensor.zeros((0, 3)), Tensor.zeros((0,)), Tensor.zeros((0, 3)), 
            Tensor.zeros((0,)).cast(dtypes.bool), pair_indices)
  
  # Gather body data for all pairs using pure tensor operations
  indices_a = pair_indices[:, 0]
  indices_b = pair_indices[:, 1]
  
  # Gather full body data for each pair
  # We need to expand indices to match the shape of bodies
  indices_a_expanded = indices_a.unsqueeze(1).expand(-1, bodies.shape[1])
  indices_b_expanded = indices_b.unsqueeze(1).expand(-1, bodies.shape[1])
  bodies_a = bodies.gather(0, indices_a_expanded)
  bodies_b = bodies.gather(0, indices_b_expanded)
  
  # Extract data for body A (first in each pair)
  pos_a = bodies_a[:, BodySchema.POS_X:BodySchema.POS_Z+1]
  quat_a = bodies_a[:, BodySchema.QUAT_W:BodySchema.QUAT_Z+1]
  shape_type_a = bodies_a[:, BodySchema.SHAPE_TYPE]
  shape_params_a = bodies_a[:, BodySchema.SHAPE_PARAM_1:BodySchema.SHAPE_PARAM_3+1]
  
  # Extract data for body B (second in each pair)
  pos_b = bodies_b[:, BodySchema.POS_X:BodySchema.POS_Z+1]
  quat_b = bodies_b[:, BodySchema.QUAT_W:BodySchema.QUAT_Z+1]
  shape_type_b = bodies_b[:, BodySchema.SHAPE_TYPE]
  shape_params_b = bodies_b[:, BodySchema.SHAPE_PARAM_1:BodySchema.SHAPE_PARAM_3+1]
  
  # Initialize outputs
  contact_normals = Tensor.zeros((n_pairs, 3))
  contact_depths = Tensor.zeros((n_pairs,))
  contact_points = Tensor.zeros((n_pairs, 3))
  contact_mask = Tensor.zeros((n_pairs,)).cast(dtypes.bool)
  
  # Sphere-Sphere collision detection (vectorized)
  is_sphere_sphere = ((shape_type_a == ShapeType.SPHERE) & (shape_type_b == ShapeType.SPHERE)).float()
  
  # Always compute sphere-sphere collisions (masked later)
  # Extract radii
  radius_a = shape_params_a[:, 0:1]  # Keep dimension for broadcasting
  radius_b = shape_params_b[:, 0:1]
  
  # Compute collision info for all sphere pairs
  delta = pos_a - pos_b  # Vector from B to A
  dist_sq = (delta * delta).sum(axis=1, keepdim=True)
  dist = dist_sq.sqrt()
  radii_sum = radius_a + radius_b
  
  # Check collision
  sphere_collision = (dist < radii_sum).float()
  
  # Compute contact normal (direction from B to A)
  # Handle zero distance case by using default up vector
  safe_dist = dist.maximum(1e-6)
  normal = delta / safe_dist
  
  # Contact point on sphere B's surface
  point = pos_b + delta * (radius_b / radii_sum)
  
  # Penetration depth
  depth = (radii_sum - dist).squeeze()
  
  # Apply mask for sphere-sphere pairs that are actually colliding
  sphere_mask = is_sphere_sphere * sphere_collision.squeeze() * collision_mask.float()
  sphere_mask_3d = sphere_mask.unsqueeze(1).expand(-1, 3)
  
  # Update outputs using masks
  contact_normals = contact_normals + sphere_mask_3d * normal
  contact_depths = contact_depths + sphere_mask * depth
  contact_points = contact_points + sphere_mask_3d * point
  contact_mask = contact_mask | (sphere_mask > 0)
  
  # Sphere-Box collision detection (vectorized)
  # Handle both orderings: (sphere, box) and (box, sphere)
  is_sphere_box = ((shape_type_a == ShapeType.SPHERE) & (shape_type_b == ShapeType.BOX)).float()
  is_box_sphere = ((shape_type_a == ShapeType.BOX) & (shape_type_b == ShapeType.SPHERE)).float()
  
  # Process sphere-box (sphere is A, box is B)
  # Get box rotation matrices and their inverses
  rot_box_b = quat_to_rotmat(quat_b)  # (M, 3, 3)
  rot_box_inv_b = rot_box_b.transpose(-1, -2)  # (M, 3, 3)
  
  # Transform sphere center to box local space
  delta_world_sb = pos_a - pos_b  # (M, 3)
  sphere_local_sb = (rot_box_inv_b @ delta_world_sb.unsqueeze(-1)).squeeze(-1)  # (M, 3)
  
  # Clamp to box bounds to find closest point
  closest_local_sb = sphere_local_sb.clip(-shape_params_b, shape_params_b)
  
  # Check collision
  delta_local_sb = sphere_local_sb - closest_local_sb
  dist_sq_sb = (delta_local_sb * delta_local_sb).sum(axis=1)
  dist_sb = dist_sq_sb.sqrt()
  
  # Collision occurs when distance < radius
  sphere_radius_a = shape_params_a[:, 0]
  box_collision_sb = (dist_sb < sphere_radius_a).float()
  
  # Transform closest point back to world space
  closest_world_sb = pos_b + (rot_box_b @ closest_local_sb.unsqueeze(-1)).squeeze(-1)
  
  # Compute normal (from box to sphere)
  safe_dist_sb = dist_sb.maximum(1e-6).unsqueeze(1)
  normal_local_sb = delta_local_sb / safe_dist_sb
  normal_world_sb = (rot_box_b @ normal_local_sb.unsqueeze(-1)).squeeze(-1)
  
  # Penetration depth
  depth_sb = sphere_radius_a - dist_sb
  
  # Apply masks for sphere-box
  combined_mask_sb = is_sphere_box * box_collision_sb * collision_mask.float()
  combined_mask_3d_sb = combined_mask_sb.unsqueeze(1).expand(-1, 3)
  
  # Update outputs
  contact_normals = contact_normals + combined_mask_3d_sb * normal_world_sb
  contact_depths = contact_depths + combined_mask_sb * depth_sb
  contact_points = contact_points + combined_mask_3d_sb * closest_world_sb
  contact_mask = contact_mask | (combined_mask_sb > 0)
  
  # Process box-sphere (box is A, sphere is B) - similar logic but swapped
  rot_box_a = quat_to_rotmat(quat_a)  # (M, 3, 3)
  rot_box_inv_a = rot_box_a.transpose(-1, -2)  # (M, 3, 3)
  
  delta_world_bs = pos_b - pos_a  # (M, 3)
  sphere_local_bs = (rot_box_inv_a @ delta_world_bs.unsqueeze(-1)).squeeze(-1)  # (M, 3)
  
  closest_local_bs = sphere_local_bs.clip(-shape_params_a, shape_params_a)
  
  delta_local_bs = sphere_local_bs - closest_local_bs
  dist_sq_bs = (delta_local_bs * delta_local_bs).sum(axis=1)
  dist_bs = dist_sq_bs.sqrt()
  
  sphere_radius_b = shape_params_b[:, 0]
  box_collision_bs = (dist_bs < sphere_radius_b).float()
  
  closest_world_bs = pos_a + (rot_box_a @ closest_local_bs.unsqueeze(-1)).squeeze(-1)
  
  safe_dist_bs = dist_bs.maximum(1e-6).unsqueeze(1)
  normal_local_bs = delta_local_bs / safe_dist_bs
  normal_world_bs = -(rot_box_a @ normal_local_bs.unsqueeze(-1)).squeeze(-1)  # Flip normal
  
  depth_bs = sphere_radius_b - dist_bs
  
  combined_mask_bs = is_box_sphere * box_collision_bs * collision_mask.float()
  combined_mask_3d_bs = combined_mask_bs.unsqueeze(1).expand(-1, 3)
  
  contact_normals = contact_normals + combined_mask_3d_bs * normal_world_bs
  contact_depths = contact_depths + combined_mask_bs * depth_bs
  contact_points = contact_points + combined_mask_3d_bs * closest_world_bs
  contact_mask = contact_mask | (combined_mask_bs > 0)
  
  return contact_normals, contact_depths, contact_points, contact_mask, pair_indices