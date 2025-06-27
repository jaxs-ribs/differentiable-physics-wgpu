"""Math utilities for physics calculations.

Provides quaternion operations and coordinate transformations needed for
3D rigid body physics. All operations are vectorized for batch processing.
"""
import numpy as np
from tinygrad import Tensor

def quat_multiply(q1: Tensor, q2: Tensor) -> Tensor:
  """Multiply batches of quaternions using Hamilton product.
  
  Args:
    q1, q2: Quaternions as tensors of shape (N, 4) in [w, x, y, z] format
    
  Returns:
    Product q1 * q2 as tensor of shape (N, 4)
  """
  w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
  w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
  return Tensor.stack(w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2, w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2, dim=1)

def quat_rotate(q: Tensor, v: Tensor) -> Tensor:
  """Rotate vectors by quaternions using q * v * q^*.
  
  Args:
    q: Quaternions as tensor of shape (N, 4)
    v: Vectors as tensor of shape (N, 3)
    
  Returns:
    Rotated vectors as tensor of shape (N, 3)
  """
  v_quat = Tensor.cat(Tensor.zeros(v.shape[0], 1), v, dim=1)
  q_conj = Tensor.stack(q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3], dim=1)
  return quat_multiply(quat_multiply(q, v_quat), q_conj)[:, 1:4]

def get_world_inv_inertia(q: Tensor, local_inv_inertia: Tensor) -> Tensor:
  """Transform inverse inertia tensors from local to world space.
  
  Args:
    q: Quaternions as tensor of shape (N, 4)
    local_inv_inertia: Flattened 3x3 inverse inertia tensors of shape (N, 9)
    
  Returns:
    World-space inverse inertia tensors of shape (N, 9)
    
  Note: Currently drops to numpy for matrix operations. Will be replaced
  with pure tensor ops in Phase 2 (WGSL kernel).
  """
  q_np, local_inv_inertia_np = q.numpy(), local_inv_inertia.numpy()
  world_inv_inertias = []
  for i in range(q_np.shape[0]):
    w, x, y, z = q_np[i]
    R = np.array([[1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)], [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)], [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]])
    inv_I_world = R @ local_inv_inertia_np[i].reshape(3, 3) @ R.T
    world_inv_inertias.append(inv_I_world.flatten())
  return Tensor(np.stack(world_inv_inertias), device="CPU")

def quat_to_rotmat_np(q_np: np.ndarray) -> np.ndarray:
  """Convert quaternions to rotation matrices (numpy version for CPU ops).
  
  Args:
    q_np: Quaternions as numpy array of shape (N, 4)
    
  Returns:
    Rotation matrices as numpy array of shape (N, 3, 3)
  """
  w, x, y, z = q_np[:, 0], q_np[:, 1], q_np[:, 2], q_np[:, 3]
  x2, y2, z2 = x*x, y*y, z*z
  wx, wy, wz = w*x, w*y, w*z
  xy, xz, yz = x*y, x*z, y*z
  R = np.zeros((q_np.shape[0], 3, 3))
  R[:, 0, 0] = 1 - 2*(y2 + z2); R[:, 0, 1] = 2*(xy - wz); R[:, 0, 2] = 2*(xz + wy)
  R[:, 1, 0] = 2*(xy + wz); R[:, 1, 1] = 1 - 2*(x2 + z2); R[:, 1, 2] = 2*(yz - wx)
  R[:, 2, 0] = 2*(xz - wy); R[:, 2, 1] = 2*(yz + wx); R[:, 2, 2] = 1 - 2*(x2 + y2)
  return R