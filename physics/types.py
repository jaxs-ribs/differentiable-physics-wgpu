"""Physics engine type definitions and data schemas.

This module defines the data layout for our Structure of Arrays (SoA) approach.
Each body's state is stored as a row in a wide tensor, with columns representing
different properties. This layout is optimized for GPU parallelism.
"""
from enum import IntEnum
import numpy as np
from typing import NamedTuple

class BodySchema:
  """Column indices for body state tensor.
  
  The state tensor has shape (num_bodies, NUM_PROPERTIES) where each row is a body
  and columns store: position, velocity, orientation (quaternion), angular velocity,
  inverse mass, inverse inertia tensor (3x3 flattened), shape type, and shape parameters.
  """
  POS_X, POS_Y, POS_Z = 0, 1, 2
  VEL_X, VEL_Y, VEL_Z = 3, 4, 5
  QUAT_W, QUAT_X, QUAT_Y, QUAT_Z = 6, 7, 8, 9
  ANG_VEL_X, ANG_VEL_Y, ANG_VEL_Z = 10, 11, 12
  INV_MASS = 13
  INV_INERTIA_XX, INV_INERTIA_XY, INV_INERTIA_XZ = 14, 15, 16
  INV_INERTIA_YX, INV_INERTIA_YY, INV_INERTIA_YZ = 17, 18, 19
  INV_INERTIA_ZX, INV_INERTIA_ZY, INV_INERTIA_ZZ = 20, 21, 22
  SHAPE_TYPE = 23
  SHAPE_PARAM_1, SHAPE_PARAM_2, SHAPE_PARAM_3 = 24, 25, 26
  NUM_PROPERTIES = 27

class ShapeType(IntEnum):
  """Supported collision shapes."""
  SPHERE = 0   # param_1: radius
  BOX = 2      # param_1,2,3: half_extents x,y,z
  CAPSULE = 3  # param_1: half-height, param_2: radius

class Contact(NamedTuple):
  """Contact point information from narrowphase collision detection."""
  pair_indices: tuple[int, int]  # (body_i, body_j)
  normal: np.ndarray  # contact normal pointing from j to i
  depth: float  # penetration depth
  point: np.ndarray  # world-space contact point

def create_body_array(position: np.ndarray, velocity: np.ndarray, orientation: np.ndarray, 
                      angular_vel: np.ndarray, mass: float, inertia: np.ndarray, 
                      shape_type: ShapeType, shape_params: np.ndarray) -> np.ndarray:
  """Create a single body's state array."""
  body = np.zeros(BodySchema.NUM_PROPERTIES, dtype=np.float32)
  body[BodySchema.POS_X:BodySchema.POS_Z+1] = position
  body[BodySchema.VEL_X:BodySchema.VEL_Z+1] = velocity
  body[BodySchema.QUAT_W:BodySchema.QUAT_Z+1] = orientation
  body[BodySchema.ANG_VEL_X:BodySchema.ANG_VEL_Z+1] = angular_vel
  body[BodySchema.INV_MASS] = 1.0 / mass if mass < 1e7 else 0.0
  body[BodySchema.INV_INERTIA_XX:BodySchema.INV_INERTIA_ZZ+1] = np.linalg.inv(inertia).flatten()
  body[BodySchema.SHAPE_TYPE] = shape_type.value
  body[BodySchema.SHAPE_PARAM_1:BodySchema.SHAPE_PARAM_3+1] = shape_params
  return body