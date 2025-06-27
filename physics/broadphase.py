"""Sweep and Prune broadphase collision detection.

Broadphase collision detection quickly finds pairs of bodies that might be
colliding by checking if their axis-aligned bounding boxes (AABBs) overlap.
This avoids the O(n²) cost of checking every pair of bodies.

Algorithm: Sweep and Prune (SAP)
1. Compute world-space AABBs for all bodies based on their shapes and transforms
2. Project AABBs onto the X-axis, creating min/max endpoints
3. Sort endpoints along X-axis
4. Sweep through sorted list, tracking "active" bodies
5. When we hit a min endpoint, check Y/Z overlap with all active bodies
6. When we hit a max endpoint, remove body from active set

This is O(n log n) for sorting plus O(n*k) for sweep where k is the average
number of overlapping AABBs along X-axis. For most scenes, k << n.

Note: Currently uses numpy for sorting as tinygrad lacks a sort op.
This will be replaced with a GPU-parallel sort in Phase 2 (WGSL kernel).
"""
import numpy as np
from tinygrad import Tensor
from .types import BodySchema, ShapeType
from .math_utils import quat_to_rotmat_np

def compute_aabb(position: np.ndarray, rotation: np.ndarray, shape_type: int, 
                 shape_params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Compute world-space AABB for a single body.
  
  Args:
    position: World position [x, y, z]
    rotation: Rotation matrix (3, 3)
    shape_type: ShapeType enum value
    shape_params: Shape-specific parameters
    
  Returns:
    (min_point, max_point) defining the AABB
  """
  if shape_type == ShapeType.SPHERE:
    radius = shape_params[0]
    return position - radius, position + radius
  elif shape_type == ShapeType.BOX:
    # Transform box corners to world space
    half_extents = shape_params
    corners = np.array([[-1,-1,-1], [-1,-1,1], [-1,1,-1], [-1,1,1], [1,-1,-1], [1,-1,1], [1,1,-1], [1,1,1]], dtype=np.float32)
    world_corners = (rotation @ (corners * half_extents).T).T + position
    return np.min(world_corners, axis=0), np.max(world_corners, axis=0)
  elif shape_type == ShapeType.CAPSULE:
    radius, half_height = shape_params[0], shape_params[1] / 2.0
    # Capsule endpoints in local space
    p1_world = position + rotation @ np.array([0, half_height, 0])
    p2_world = position + rotation @ np.array([0, -half_height, 0])
    return np.minimum(p1_world, p2_world) - radius, np.maximum(p1_world, p2_world) + radius
  else:
    raise ValueError(f"Unknown shape type: {shape_type}")

def broadphase_sweep_and_prune(bodies: Tensor) -> Tensor:
  """Find potentially colliding pairs using Sweep and Prune algorithm.
  
  Args:
    bodies: State tensor of shape (N, NUM_PROPERTIES)
    
  Returns:
    Tensor of shape (M, 2) containing indices of potentially colliding pairs,
    where M is the number of overlapping AABB pairs. Returns empty tensor if no overlaps.
  """
  bodies_np = bodies.numpy()
  positions = bodies_np[:, BodySchema.POS_X:BodySchema.POS_Z+1]
  quats = bodies_np[:, BodySchema.QUAT_W:BodySchema.QUAT_Z+1]
  shape_types = bodies_np[:, BodySchema.SHAPE_TYPE].astype(int)
  shape_params = bodies_np[:, BodySchema.SHAPE_PARAM_1:BodySchema.SHAPE_PARAM_3+1]
  
  # Compute rotation matrices and AABBs for all bodies
  rot_matrices = quat_to_rotmat_np(quats)
  min_pts, max_pts = np.zeros_like(positions), np.zeros_like(positions)
  
  # Compute AABBs for each shape type in batch where possible
  sphere_mask = shape_types == ShapeType.SPHERE
  if np.any(sphere_mask):
    radii = shape_params[sphere_mask, 0:1]
    min_pts[sphere_mask] = positions[sphere_mask] - radii
    max_pts[sphere_mask] = positions[sphere_mask] + radii
  
  box_mask = shape_types == ShapeType.BOX
  if np.any(box_mask):
    box_indices = np.where(box_mask)[0]
    unit_corners = np.array([[-1,-1,-1], [-1,-1,1], [-1,1,-1], [-1,1,1], [1,-1,-1], [1,-1,1], [1,1,-1], [1,1,1]], dtype=np.float32)
    for i in box_indices:
      world_corners = (rot_matrices[i] @ (unit_corners * shape_params[i]).T).T + positions[i]
      min_pts[i], max_pts[i] = np.min(world_corners, axis=0), np.max(world_corners, axis=0)
  
  capsule_mask = shape_types == ShapeType.CAPSULE
  if np.any(capsule_mask):
    for i in np.where(capsule_mask)[0]:
      radius, half_height = shape_params[i, 0], shape_params[i, 1] / 2.0
      p1_world = positions[i] + rot_matrices[i] @ np.array([0, half_height, 0])
      p2_world = positions[i] + rot_matrices[i] @ np.array([0, -half_height, 0])
      min_pts[i], max_pts[i] = np.minimum(p1_world, p2_world) - radius, np.maximum(p1_world, p2_world) + radius
  
  # Create sorted endpoint list for X-axis sweep
  aabbs_np = np.concatenate([min_pts, max_pts], axis=1)  # [x_min, y_min, z_min, x_max, y_max, z_max]
  x_endpoints = [(aabbs_np[i, 0], i, True) for i in range(len(aabbs_np))] + [(aabbs_np[i, 3], i, False) for i in range(len(aabbs_np))]
  x_endpoints.sort(key=lambda x: x[0])
  
  # Sweep along X-axis to find overlapping pairs
  active, overlaps = set(), set()
  for _, body_idx, is_min in x_endpoints:
    if is_min:
      # Check Y and Z overlap with all currently active bodies
      for other_idx in active:
        if (aabbs_np[body_idx, 1] <= aabbs_np[other_idx, 4] and aabbs_np[body_idx, 4] >= aabbs_np[other_idx, 1] and
            aabbs_np[body_idx, 2] <= aabbs_np[other_idx, 5] and aabbs_np[body_idx, 5] >= aabbs_np[other_idx, 2]):
          overlaps.add(tuple(sorted((body_idx, other_idx))))
      active.add(body_idx)
    else: active.discard(body_idx)
  
  return Tensor(np.array(list(overlaps), dtype=np.int32)) if overlaps else Tensor(np.empty((0, 2), dtype=np.int32))