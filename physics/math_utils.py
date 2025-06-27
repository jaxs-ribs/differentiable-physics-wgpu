"""Math utilities for physics calculations.

Provides quaternion operations and coordinate transformations needed for
3D rigid body physics. All operations are vectorized for batch processing.
"""
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
  """
  # Get rotation matrices from quaternions
  R = quat_to_rotmat(q)  # (N, 3, 3)
  
  # Reshape local inverse inertia to (N, 3, 3)
  local_I = local_inv_inertia.reshape(-1, 3, 3)
  
  # Transform: R @ I_local @ R^T
  # First: R @ I_local
  RI = R @ local_I  # (N, 3, 3) @ (N, 3, 3) = (N, 3, 3)
  
  # Then: (R @ I_local) @ R^T
  R_T = R.transpose(-1, -2)  # Transpose last two dimensions
  world_I = RI @ R_T  # (N, 3, 3) @ (N, 3, 3) = (N, 3, 3)
  
  # Flatten back to (N, 9)
  return world_I.reshape(-1, 9)

def quat_to_rotmat(q: Tensor) -> Tensor:
  """Convert quaternions to rotation matrices using pure tensor operations.
  
  Args:
    q: Quaternions as tensor of shape (N, 4) in [w, x, y, z] format
    
  Returns:
    Rotation matrices as tensor of shape (N, 3, 3)
  """
  w, x, y, z = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
  # Pre-compute squares and products
  x2, y2, z2 = x*x, y*y, z*z
  wx, wy, wz = w*x, w*y, w*z
  xy, xz, yz = x*y, x*z, y*z
  
  # Build rotation matrix components
  r00 = 1 - 2*(y2 + z2)
  r01 = 2*(xy - wz)
  r02 = 2*(xz + wy)
  r10 = 2*(xy + wz)
  r11 = 1 - 2*(x2 + z2)
  r12 = 2*(yz - wx)
  r20 = 2*(xz - wy)
  r21 = 2*(yz + wx)
  r22 = 1 - 2*(x2 + y2)
  
  # Stack into 3x3 matrices - concat columns, then reshape
  row0 = Tensor.cat(r00, r01, r02, dim=1)
  row1 = Tensor.cat(r10, r11, r12, dim=1)
  row2 = Tensor.cat(r20, r21, r22, dim=1)
  # Stack rows and reshape to (N, 3, 3)
  return Tensor.stack(row0, row1, row2, dim=1).reshape(-1, 3, 3)

