"""Unit tests for mathematical utility functions."""
import pytest
import numpy as np
from tinygrad import Tensor
from physics.math_utils import quat_mul, quat_exp, quat_normalize, cross_product, apply_quaternion_to_vector


class TestQuaternionMath:
    """Test suite for quaternion mathematical operations."""
    
    def test_quat_mul_identity(self):
        """Test quaternion multiplication with identity."""
        # Identity quaternion
        q_identity = Tensor([1.0, 0.0, 0.0, 0.0])
        
        # Test quaternion
        q = Tensor([0.707, 0.707, 0.0, 0.0])  # 90 deg rotation around X
        
        # q * identity = q
        result = quat_mul(q, q_identity)
        np.testing.assert_allclose(result.numpy(), q.numpy(), rtol=1e-5)
        
        # identity * q = q
        result = quat_mul(q_identity, q)
        np.testing.assert_allclose(result.numpy(), q.numpy(), rtol=1e-5)
    
    def test_quat_mul_inverse(self):
        """Test quaternion multiplication with its conjugate gives identity."""
        # Test quaternion (90 deg around Y)
        q = Tensor([0.707, 0.0, 0.707, 0.0])
        
        # Conjugate
        q_conj = Tensor([0.707, 0.0, -0.707, 0.0])
        
        # q * q_conj should give identity
        result = quat_mul(q, q_conj)
        expected = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-3, atol=1e-5)
    
    def test_quat_exp_zero(self):
        """Test quaternion exponential of zero vector gives identity."""
        v = Tensor([0.0, 0.0, 0.0])
        q = quat_exp(v)
        
        expected = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(q.numpy(), expected, rtol=1e-5)
    
    def test_quat_exp_rotation(self):
        """Test quaternion exponential for known rotations."""
        # The quat_exp function expects v = axis * half_angle
        # For 90 degree rotation around X: half_angle = π/4
        # So v = [1, 0, 0] * π/4 = [π/4, 0, 0]
        half_angle = np.pi/4
        v = Tensor([half_angle, 0.0, 0.0])
        q = quat_exp(v)
        
        # quat_exp computes: [cos(||v||/2), sin(||v||/2) * v/||v||]
        # ||v|| = π/4, so ||v||/2 = π/8
        # Expected: [cos(π/8), sin(π/8), 0, 0]
        angle = half_angle / 2  # π/8
        expected = np.array([np.cos(angle), np.sin(angle), 0.0, 0.0])
        np.testing.assert_allclose(q.numpy(), expected, rtol=1e-5)
    
    def test_quat_normalize(self):
        """Test quaternion normalization."""
        # Unnormalized quaternions
        q = Tensor([[2.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [3.0, 4.0, 0.0, 0.0]])
        
        q_norm = quat_normalize(q)
        q_norm_np = q_norm.numpy()
        
        # Check all have unit norm
        norms = np.linalg.norm(q_norm_np, axis=1)
        np.testing.assert_allclose(norms, np.ones(3), rtol=1e-5)
        
        # Check direction is preserved
        # First quaternion should be [1, 0, 0, 0]
        np.testing.assert_allclose(q_norm_np[0], [1.0, 0.0, 0.0, 0.0], rtol=1e-5)
        
        # Second should be [0.5, 0.5, 0.5, 0.5]
        expected = np.array([0.5, 0.5, 0.5, 0.5])
        np.testing.assert_allclose(q_norm_np[1], expected, rtol=1e-5)
    
    def test_cross_product(self):
        """Test cross product computation."""
        # Standard basis vectors
        x = Tensor([[1.0, 0.0, 0.0]])
        y = Tensor([[0.0, 1.0, 0.0]])
        z = Tensor([[0.0, 0.0, 1.0]])
        
        # x × y = z
        result = cross_product(x, y)
        np.testing.assert_allclose(result.numpy(), z.numpy(), rtol=1e-5)
        
        # y × z = x
        result = cross_product(y, z)
        np.testing.assert_allclose(result.numpy(), x.numpy(), rtol=1e-5)
        
        # z × x = y
        result = cross_product(z, x)
        np.testing.assert_allclose(result.numpy(), y.numpy(), rtol=1e-5)
        
        # x × x = 0
        result = cross_product(x, x)
        np.testing.assert_allclose(result.numpy(), np.zeros((1, 3)), rtol=1e-5)
    
    def test_cross_product_batch(self):
        """Test cross product with batched inputs."""
        v1 = Tensor([[1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [1.0, 1.0, 0.0]])
        v2 = Tensor([[0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0],
                     [1.0, 0.0, 0.0]])
        
        result = cross_product(v1, v2)
        
        # Expected results
        expected = np.array([[0.0, 0.0, 1.0],   # x × y = z
                            [1.0, 0.0, 0.0],   # y × z = x
                            [0.0, 0.0, -1.0]]) # (x+y) × x = -z
        
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)
    
    def test_apply_quaternion_to_vector(self):
        """Test rotating a vector by a quaternion."""
        # 90 degree rotation around Y axis
        q = Tensor([0.707, 0.0, 0.707, 0.0])  # cos(45°), 0, sin(45°), 0
        
        # Vector pointing in +X direction
        v = Tensor([1.0, 0.0, 0.0])
        
        # After 90° rotation around Y, should point in -Z direction
        v_rot = apply_quaternion_to_vector(q.unsqueeze(0), v.unsqueeze(0))
        expected = np.array([[0.0, 0.0, -1.0]])
        
        np.testing.assert_allclose(v_rot.numpy(), expected, rtol=1e-3, atol=1e-4)
    
    def test_apply_quaternion_identity(self):
        """Test that identity quaternion doesn't change vectors."""
        q_identity = Tensor([[1.0, 0.0, 0.0, 0.0]])
        
        # Test with multiple vectors
        vectors = Tensor([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0],
                         [-1.0, -2.0, -3.0]])
        
        # Apply identity rotation to each vector
        for i in range(3):
            v = vectors[i:i+1]
            v_rot = apply_quaternion_to_vector(q_identity, v)
            np.testing.assert_allclose(v_rot.numpy(), v.numpy(), rtol=1e-5)