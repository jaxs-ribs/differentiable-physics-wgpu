"""Unit tests for math utilities.

Tests quaternion operations, rotation matrices, and coordinate transformations.
"""
import numpy as np
import pytest
from tinygrad import Tensor, dtypes
from physics.math_utils import quat_multiply, quat_rotate, quat_to_rotmat

class TestQuaternionOperations:
  """Test quaternion math functions."""
  
  def test_quat_to_rotmat_identity(self):
    """Test that identity quaternion produces identity matrix."""
    # Identity quaternion [w, x, y, z] = [1, 0, 0, 0]
    q = Tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtypes.float32)
    R = quat_to_rotmat(q)
    
    # Should produce identity matrix
    expected = np.eye(3, dtype=np.float32)
    np.testing.assert_allclose(R.numpy()[0], expected, atol=1e-6)
  
  def test_quat_to_rotmat_known_rotation(self):
    """Test 90-degree rotation about Z-axis."""
    # 90-degree rotation about Z: [cos(45°), 0, 0, sin(45°)]
    angle = np.pi / 2
    q = Tensor([[np.cos(angle/2), 0.0, 0.0, np.sin(angle/2)]], dtype=dtypes.float32)
    R = quat_to_rotmat(q)
    
    # Expected rotation matrix for 90° about Z
    expected = np.array([
      [0, -1, 0],
      [1, 0, 0],
      [0, 0, 1]
    ], dtype=np.float32)
    
    np.testing.assert_allclose(R.numpy()[0], expected, atol=1e-6)
  
  def test_quat_to_rotmat_batch(self):
    """Test batched quaternion to rotation matrix conversion."""
    # Multiple quaternions
    q = Tensor([
      [1.0, 0.0, 0.0, 0.0],  # identity
      [0.0, 1.0, 0.0, 0.0],  # 180° about X
      [0.0, 0.0, 1.0, 0.0],  # 180° about Y
    ], dtype=dtypes.float32)
    
    R = quat_to_rotmat(q)
    assert R.shape == (3, 3, 3)
    
    # Check identity
    np.testing.assert_allclose(R.numpy()[0], np.eye(3), atol=1e-6)
    
    # Check 180° about X (flips Y and Z)
    expected_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
    np.testing.assert_allclose(R.numpy()[1], expected_x, atol=1e-6)
  
  def test_quat_multiply(self):
    """Test quaternion multiplication."""
    # Two 90-degree rotations about Z should give 180-degree rotation
    angle = np.pi / 2
    q1 = Tensor([[np.cos(angle/2), 0.0, 0.0, np.sin(angle/2)]], dtype=dtypes.float32)
    q2 = q1.clone()
    
    q_result = quat_multiply(q1, q2)
    
    # Expected: 180-degree rotation about Z
    expected_angle = np.pi
    expected = np.array([[np.cos(expected_angle/2), 0.0, 0.0, np.sin(expected_angle/2)]], dtype=np.float32)
    np.testing.assert_allclose(q_result.numpy(), expected, atol=1e-6)
  
  def test_quat_rotate(self):
    """Test vector rotation by quaternion."""
    # 90-degree rotation about Z
    angle = np.pi / 2
    q = Tensor([[np.cos(angle/2), 0.0, 0.0, np.sin(angle/2)]], dtype=dtypes.float32)
    
    # Rotate unit X vector
    v = Tensor([[1.0, 0.0, 0.0]], dtype=dtypes.float32)
    v_rotated = quat_rotate(q, v)
    
    # Should become unit Y vector
    expected = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(v_rotated.numpy(), expected, atol=1e-6)
  
  def test_quat_to_rotmat_consistency(self):
    """Test that tensor and numpy versions produce same results."""
    # Random quaternions (normalized)
    np.random.seed(42)
    q_np = np.random.randn(5, 4).astype(np.float32)
    q_np = q_np / np.linalg.norm(q_np, axis=1, keepdims=True)
    
    # Convert using tensor method
    q_tensor = Tensor(q_np)
    R_tensor = quat_to_rotmat(q_tensor).numpy()
    
    # Manually compute numpy version for comparison
    R_numpy = np.zeros((5, 3, 3), dtype=np.float32)
    for i in range(5):
        w, x, y, z = q_np[i]
        R_numpy[i] = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ], dtype=np.float32)
    
    np.testing.assert_allclose(R_tensor, R_numpy, atol=1e-6)
  
  def test_rotation_matrix_properties(self):
    """Test that generated rotation matrices are valid (orthogonal, det=1)."""
    # Random quaternions
    np.random.seed(42)
    q = np.random.randn(10, 4).astype(np.float32)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    
    R = quat_to_rotmat(Tensor(q)).numpy()
    
    for i in range(10):
      # Check orthogonality: R @ R.T = I
      should_be_identity = R[i] @ R[i].T
      np.testing.assert_allclose(should_be_identity, np.eye(3), atol=1e-6)
      
      # Check determinant = 1 (proper rotation, not reflection)
      det = np.linalg.det(R[i])
      np.testing.assert_allclose(det, 1.0, atol=1e-6)