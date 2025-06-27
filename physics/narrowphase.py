"""Narrowphase collision detection for exact contact generation.

After broadphase identifies potentially colliding pairs, narrowphase performs
exact geometric tests to determine if bodies are actually colliding and
generates contact points with normals and penetration depths.

Supported collision tests:
- Sphere vs Sphere: Distance between centers vs sum of radii
- Sphere vs Box: Closest point on box to sphere center
- Box vs Box: Not implemented (requires SAT or GJK)
- Capsule collisions: Not implemented

Contact information includes:
- Contact point: Where bodies touch in world space
- Contact normal: Direction to separate bodies (from B to A)
- Penetration depth: How much bodies overlap

This information is used by the solver to calculate impulses that separate
the bodies and simulate realistic collisions.
"""
import numpy as np
from tinygrad import Tensor
from .types import BodySchema, ShapeType, Contact
from .math_utils import quat_to_rotmat_np

def sphere_sphere_collision(pos_i: np.ndarray, radius_i: float, 
                           pos_j: np.ndarray, radius_j: float) -> Contact | None:
  """Test collision between two spheres.
  
  Two spheres collide when the distance between their centers is less than
  the sum of their radii. The contact point lies on the line between centers.
  
  Args:
    pos_i, pos_j: Sphere center positions
    radius_i, radius_j: Sphere radii
    
  Returns:
    Contact if collision detected, None otherwise
  """
  delta = pos_i - pos_j
  dist_sq = np.dot(delta, delta)
  radii_sum = radius_i + radius_j
  
  if dist_sq < radii_sum * radii_sum:
    dist = np.sqrt(dist_sq)
    # Normal points from j to i (direction to separate the spheres)
    normal = delta / dist if dist > 1e-6 else np.array([0, 1, 0])
    # Contact point is on sphere j's surface along the line to sphere i
    point = pos_j + delta * (radius_j / radii_sum)
    depth = radii_sum - dist
    return Contact(pair_indices=(0, 0), normal=normal, depth=depth, point=point)  # Indices set by caller
  return None

def sphere_box_collision(sphere_pos: np.ndarray, sphere_radius: float,
                        box_pos: np.ndarray, box_quat: np.ndarray, 
                        box_half_extents: np.ndarray) -> Contact | None:
  """Test collision between a sphere and an oriented box.
  
  Algorithm:
  1. Transform sphere center to box's local space
  2. Find closest point on axis-aligned box to sphere center
  3. Check if distance to closest point is less than sphere radius
  4. Transform contact info back to world space
  
  Args:
    sphere_pos: Sphere center in world space
    sphere_radius: Sphere radius
    box_pos: Box center in world space
    box_quat: Box orientation quaternion
    box_half_extents: Box half-dimensions along local axes
    
  Returns:
    Contact if collision detected, None otherwise
  """
  # Transform sphere to box's local space
  rot_box_inv = quat_to_rotmat_np(np.array([box_quat]))[0].T
  sphere_center_local = rot_box_inv @ (sphere_pos - box_pos)
  
  # Clamp to box bounds to find closest point
  closest_point_local = np.maximum(-box_half_extents, np.minimum(box_half_extents, sphere_center_local))
  
  # Check collision
  delta_local = sphere_center_local - closest_point_local
  dist_sq = np.dot(delta_local, delta_local)
  
  if dist_sq < sphere_radius * sphere_radius:
    # Transform back to world space
    rot_box = rot_box_inv.T
    closest_point_world = box_pos + rot_box @ closest_point_local
    dist = np.sqrt(dist_sq)
    # Normal points from box to sphere (direction to push sphere out)
    normal = rot_box @ (delta_local / dist) if dist > 1e-6 else rot_box @ np.array([0, 1, 0])
    depth = sphere_radius - dist
    return Contact(pair_indices=(0, 0), normal=normal, depth=depth, point=closest_point_world)
  return None

def narrowphase(bodies: Tensor, pair_indices: Tensor, collision_mask: Tensor) -> list[Contact]:
  """Perform exact collision detection on potentially colliding pairs.
  
  Args:
    bodies: State tensor of shape (N, NUM_PROPERTIES)
    pair_indices: Tensor of shape (M, 2) with indices of all pairs
    collision_mask: Boolean tensor of shape (M,) indicating which pairs have overlapping AABBs
    
  Returns:
    List of Contact objects with collision information
  """
  # Filter to only pairs with overlapping AABBs
  mask_np = collision_mask.numpy()
  if not mask_np.any(): return []
  
  bodies_np = bodies.numpy()
  pair_indices_np = pair_indices.numpy()
  
  # Ensure mask has same length as pair_indices
  if len(mask_np) != len(pair_indices_np):
    print(f"Warning: mask length {len(mask_np)} != pair_indices length {len(pair_indices_np)}")
    return []
    
  active_pairs = pair_indices_np[mask_np]
  contacts = []
  
  # Extract relevant data for all bodies
  positions = bodies_np[:, BodySchema.POS_X:BodySchema.POS_Z+1]
  quats = bodies_np[:, BodySchema.QUAT_W:BodySchema.QUAT_Z+1]
  shape_types = bodies_np[:, BodySchema.SHAPE_TYPE].astype(int)
  shape_params = bodies_np[:, BodySchema.SHAPE_PARAM_1:BodySchema.SHAPE_PARAM_3+1]
  
  for i, j in active_pairs:
    type_i, type_j = shape_types[i], shape_types[j]
    
    # Sphere vs Sphere
    if type_i == ShapeType.SPHERE and type_j == ShapeType.SPHERE:
      contact = sphere_sphere_collision(positions[i], shape_params[i, 0], 
                                       positions[j], shape_params[j, 0])
      if contact:
        contacts.append(Contact((i, j), contact.normal, contact.depth, contact.point))
    
    # Sphere vs Box (ensure sphere is first)
    elif (type_i == ShapeType.SPHERE and type_j == ShapeType.BOX) or (type_i == ShapeType.BOX and type_j == ShapeType.SPHERE):
      if type_i == ShapeType.BOX: i, j = j, i  # Swap so sphere is i
      contact = sphere_box_collision(positions[i], shape_params[i, 0],
                                    positions[j], quats[j], shape_params[j])
      if contact:
        contacts.append(Contact((i, j), contact.normal, contact.depth, contact.point))
  
  if contacts: print(f"Narrowphase generated {len(contacts)} contacts.")
  return contacts