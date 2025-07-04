"""Tests for XPBD velocity reconciliation."""
import pytest
import numpy as np
from tinygrad import Tensor
from physics.xpbd.velocity_update import reconcile_velocities


def test_no_position_change_zero_velocity():
    """When positions don't change, reconciled velocities should be zero."""
    # Single body
    x_old = Tensor([[1.0, 2.0, 3.0]])
    x_proj = Tensor([[1.0, 2.0, 3.0]])  # Same position
    q_old = Tensor([[1.0, 0.0, 0.0, 0.0]])  # Identity quaternion
    q_proj = Tensor([[1.0, 0.0, 0.0, 0.0]])  # Same orientation
    v_old = Tensor([[4.0, 5.0, 6.0]])  # Old velocity doesn't matter
    omega_old = Tensor([[0.1, 0.2, 0.3]])
    dt = 0.016
    
    v_new, omega_new = reconcile_velocities(x_proj, q_proj, x_old, q_old, v_old, omega_old, dt)
    
    # Velocities should be zero since no position change
    np.testing.assert_allclose(v_new.numpy(), [[0.0, 0.0, 0.0]], rtol=1e-5)
    np.testing.assert_allclose(omega_new.numpy(), [[0.0, 0.0, 0.0]], atol=1e-5)


def test_linear_velocity_from_position_change():
    """Linear velocity should be computed from position difference."""
    # Single body moves 1 unit in X over 0.1 seconds
    x_old = Tensor([[0.0, 0.0, 0.0]])
    x_proj = Tensor([[1.0, 0.0, 0.0]])
    q_old = Tensor([[1.0, 0.0, 0.0, 0.0]])
    q_proj = Tensor([[1.0, 0.0, 0.0, 0.0]])  # No rotation
    v_old = Tensor([[0.0, 0.0, 0.0]])  # Previous velocity doesn't matter
    omega_old = Tensor([[0.0, 0.0, 0.0]])
    dt = 0.1
    
    v_new, omega_new = reconcile_velocities(x_proj, q_proj, x_old, q_old, v_old, omega_old, dt)
    
    # Expected velocity: (1.0 - 0.0) / 0.1 = 10.0 in X
    expected_v = np.array([[10.0, 0.0, 0.0]])
    np.testing.assert_allclose(v_new.numpy(), expected_v, rtol=1e-5)
    np.testing.assert_allclose(omega_new.numpy(), [[0.0, 0.0, 0.0]], rtol=1e-5)


def test_angular_velocity_from_rotation():
    """Angular velocity should be computed from orientation change."""
    # 90-degree rotation around Z-axis
    x_old = Tensor([[0.0, 0.0, 0.0]])
    x_proj = Tensor([[0.0, 0.0, 0.0]])  # No translation
    
    # Identity quaternion
    q_old = Tensor([[1.0, 0.0, 0.0, 0.0]])
    
    # 90-degree rotation around Z: q = [cos(45°), 0, 0, sin(45°)]
    angle = np.pi / 2
    q_proj = Tensor([[np.cos(angle/2), 0.0, 0.0, np.sin(angle/2)]])
    
    v_old = Tensor([[0.0, 0.0, 0.0]])
    omega_old = Tensor([[0.0, 0.0, 0.0]])
    dt = 0.1
    
    v_new, omega_new = reconcile_velocities(x_proj, q_proj, x_old, q_old, v_old, omega_old, dt)
    
    # Expected angular velocity: 90 degrees / 0.1s = pi/2 / 0.1 rad/s around Z
    expected_omega_z = (np.pi / 2) / 0.1
    np.testing.assert_allclose(v_new.numpy(), [[0.0, 0.0, 0.0]], rtol=1e-5)
    np.testing.assert_allclose(omega_new.numpy()[0, 2], expected_omega_z, rtol=1e-3)
    np.testing.assert_allclose(omega_new.numpy()[0, :2], [0.0, 0.0], atol=1e-3)


def test_combined_motion():
    """Test combined translation and rotation."""
    # Move in X and rotate around Y
    x_old = Tensor([[0.0, 0.0, 0.0]])
    x_proj = Tensor([[2.0, 0.0, 0.0]])
    
    q_old = Tensor([[1.0, 0.0, 0.0, 0.0]])
    # 45-degree rotation around Y: q = [cos(22.5°), 0, sin(22.5°), 0]
    angle = np.pi / 4
    q_proj = Tensor([[np.cos(angle/2), 0.0, np.sin(angle/2), 0.0]])
    
    v_old = Tensor([[0.0, 0.0, 0.0]])
    omega_old = Tensor([[0.0, 0.0, 0.0]])
    dt = 0.2
    
    v_new, omega_new = reconcile_velocities(x_proj, q_proj, x_old, q_old, v_old, omega_old, dt)
    
    # Expected linear velocity: 2.0 / 0.2 = 10.0 in X
    expected_v = np.array([[10.0, 0.0, 0.0]])
    # Expected angular velocity: 45 degrees / 0.2s around Y
    expected_omega_y = (np.pi / 4) / 0.2
    
    np.testing.assert_allclose(v_new.numpy(), expected_v, rtol=1e-5)
    np.testing.assert_allclose(omega_new.numpy()[0, 1], expected_omega_y, rtol=1e-3)


def test_multiple_bodies_soa():
    """Test with multiple bodies in SoA format."""
    # Three bodies with different motions
    x_old = Tensor([
        [0.0, 0.0, 0.0],  # Body 0: will translate
        [1.0, 1.0, 1.0],  # Body 1: will rotate
        [2.0, 2.0, 2.0],  # Body 2: no motion
    ])
    
    x_proj = Tensor([
        [0.5, 0.0, 0.0],  # Body 0: moved +0.5 in X
        [1.0, 1.0, 1.0],  # Body 1: no translation
        [2.0, 2.0, 2.0],  # Body 2: no change
    ])
    
    q_old = Tensor([
        [1.0, 0.0, 0.0, 0.0],  # All start with identity
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ])
    
    # Body 1 rotates 30 degrees around X
    angle = np.pi / 6
    q_proj = Tensor([
        [1.0, 0.0, 0.0, 0.0],  # Body 0: no rotation
        [np.cos(angle/2), np.sin(angle/2), 0.0, 0.0],  # Body 1: rotated
        [1.0, 0.0, 0.0, 0.0],  # Body 2: no rotation
    ])
    
    v_old = Tensor([[0.0, 0.0, 0.0]] * 3)
    omega_old = Tensor([[0.0, 0.0, 0.0]] * 3)
    dt = 0.05
    
    v_new, omega_new = reconcile_velocities(x_proj, q_proj, x_old, q_old, v_old, omega_old, dt)
    
    # Check body 0: linear velocity = 0.5 / 0.05 = 10.0 in X
    np.testing.assert_allclose(v_new.numpy()[0], [10.0, 0.0, 0.0], rtol=1e-5)
    np.testing.assert_allclose(omega_new.numpy()[0], [0.0, 0.0, 0.0], atol=1e-5)
    
    # Check body 1: angular velocity around X
    expected_omega_x = (np.pi / 6) / 0.05
    np.testing.assert_allclose(v_new.numpy()[1], [0.0, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(omega_new.numpy()[1, 0], expected_omega_x, rtol=1e-3)
    
    # Check body 2: no motion
    np.testing.assert_allclose(v_new.numpy()[2], [0.0, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(omega_new.numpy()[2], [0.0, 0.0, 0.0], atol=1e-5)


def test_small_timestep_stability():
    """Test numerical stability with very small timestep."""
    x_old = Tensor([[0.0, 0.0, 0.0]])
    x_proj = Tensor([[0.0001, 0.0, 0.0]])  # Very small movement
    q_old = Tensor([[1.0, 0.0, 0.0, 0.0]])
    q_proj = Tensor([[1.0, 0.0, 0.0, 0.0]])
    v_old = Tensor([[0.0, 0.0, 0.0]])
    omega_old = Tensor([[0.0, 0.0, 0.0]])
    dt = 0.0001  # Very small timestep
    
    v_new, omega_new = reconcile_velocities(x_proj, q_proj, x_old, q_old, v_old, omega_old, dt)
    
    # Should compute velocity = 0.0001 / 0.0001 = 1.0
    expected_v = np.array([[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(v_new.numpy(), expected_v, rtol=1e-4)