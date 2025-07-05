"""Unit tests for advanced narrowphase collision detection."""
import pytest
import numpy as np
from tinygrad import Tensor, dtypes
from physics.xpbd.narrowphase import (
    box_box_test, box_sphere_test, capsule_plane_test,
    capsule_sphere_test, capsule_capsule_test, capsule_box_test
)
from physics.types import ShapeType


def test_box_box_collision():
    """Test box-box collision detection."""
    # Two identical boxes, overlapping by 0.5 units
    x_a = Tensor([[0.0, 0.0, 0.0]])
    x_b = Tensor([[1.5, 0.0, 0.0]])
    q_a = Tensor([[1.0, 0.0, 0.0, 0.0]])  # Identity
    q_b = Tensor([[1.0, 0.0, 0.0, 0.0]])
    params_a = Tensor([[1.0, 1.0, 1.0]])  # Half-extents [1,1,1]
    params_b = Tensor([[1.0, 1.0, 1.0]])
    
    penetration, normal, contact_point = box_box_test(x_a, x_b, q_a, q_b, params_a, params_b)
    
    # Expected: boxes overlap by 0.5 units
    assert np.isclose(penetration.numpy()[0], 0.5, atol=1e-6)
    
    # Normal points from a to b for first axis in SAT (positive x)
    normal_np = normal.numpy().squeeze()
    assert np.isclose(normal_np[0], 1.0, atol=1e-6)
    assert np.isclose(normal_np[1], 0.0, atol=1e-6)
    assert np.isclose(normal_np[2], 0.0, atol=1e-6)


def test_box_sphere_collision():
    """Test box-sphere collision detection."""
    # Box at origin, sphere at (2,0,0) with radius 0.5
    x_box = Tensor([[0.0, 0.0, 0.0]])
    x_sphere = Tensor([[2.0, 0.0, 0.0]])
    q_box = Tensor([[1.0, 0.0, 0.0, 0.0]])
    q_sphere = Tensor([[1.0, 0.0, 0.0, 0.0]])  # Identity quaternion
    params_box = Tensor([[1.0, 1.0, 1.0]])  # Half-extents
    params_sphere = Tensor([[0.5, 0.0, 0.0]])  # Radius = 0.5
    
    penetration, normal, contact_point = box_sphere_test(x_box, x_sphere, q_box, q_sphere, params_box, params_sphere)
    
    # Expected: sphere touches box face, penetration = 0.5 - (2 - 1) = -0.5 (no collision)
    assert penetration.numpy()[0] < 0


def test_box_sphere_collision_with_penetration():
    """Test box-sphere collision with actual penetration."""
    # Box at origin, sphere at (1.3,0,0) with radius 0.5
    x_box = Tensor([[0.0, 0.0, 0.0]])
    x_sphere = Tensor([[1.3, 0.0, 0.0]])
    q_box = Tensor([[1.0, 0.0, 0.0, 0.0]])
    q_sphere = Tensor([[1.0, 0.0, 0.0, 0.0]])  # Identity quaternion
    params_box = Tensor([[1.0, 1.0, 1.0]])
    params_sphere = Tensor([[0.5, 0.0, 0.0]])
    
    penetration, normal, contact_point = box_sphere_test(x_box, x_sphere, q_box, q_sphere, params_box, params_sphere)
    
    # Expected: penetration = 0.5 - (1.3 - 1) = 0.2
    assert np.isclose(penetration.numpy()[0], 0.2, atol=1e-6)


def test_capsule_plane_collision():
    """Test capsule-plane collision detection."""
    # Vertical capsule above horizontal plane
    x_capsule = Tensor([[0.0, 2.0, 0.0]])
    x_plane = Tensor([[0.0, 0.0, 0.0]])
    q_capsule = Tensor([[1.0, 0.0, 0.0, 0.0]])  # Upright
    q_plane = Tensor([[1.0, 0.0, 0.0, 0.0]])
    params_capsule = Tensor([[0.5, 1.0, 0.0]])  # radius=0.5, half_height=1.0
    params_plane = Tensor([[10.0, 0.1, 10.0]])  # Thin plane
    
    penetration, normal, contact_point = capsule_plane_test(x_capsule, x_plane, q_capsule, q_plane, params_capsule, params_plane)
    
    # Bottom of capsule is at y=1.0 (center 2.0 - half_height 1.0)
    # Plane surface at y=0.1
    # Distance = 1.0 - 0.1 = 0.9
    # Penetration = radius - distance = 0.5 - 0.9 = -0.4 (no collision)
    assert penetration.numpy()[0] < 0


def test_capsule_sphere_collision():
    """Test capsule-sphere collision detection."""
    # Vertical capsule, sphere to the side
    x_capsule = Tensor([[0.0, 0.0, 0.0]])
    x_sphere = Tensor([[1.5, 0.0, 0.0]])
    q_capsule = Tensor([[1.0, 0.0, 0.0, 0.0]])
    q_sphere = Tensor([[1.0, 0.0, 0.0, 0.0]])  # Identity quaternion
    params_capsule = Tensor([[0.5, 1.0, 0.0]])  # radius=0.5, half_height=1.0
    params_sphere = Tensor([[0.7, 0.0, 0.0]])  # radius=0.7
    
    penetration, normal, contact_point = capsule_sphere_test(x_capsule, x_sphere, q_capsule, q_sphere, params_capsule, params_sphere)
    
    # Expected: distance = 1.5, radii sum = 0.5 + 0.7 = 1.2
    # Penetration = 1.2 - 1.5 = -0.3 (no collision)
    assert penetration.numpy()[0] < 0


def test_capsule_capsule_collision():
    """Test capsule-capsule collision detection."""
    # Two parallel vertical capsules
    x_a = Tensor([[0.0, 0.0, 0.0]])
    x_b = Tensor([[1.0, 0.0, 0.0]])
    q_a = Tensor([[1.0, 0.0, 0.0, 0.0]])
    q_b = Tensor([[1.0, 0.0, 0.0, 0.0]])
    params_a = Tensor([[0.3, 1.0, 0.0]])  # radius=0.3, half_height=1.0
    params_b = Tensor([[0.4, 0.5, 0.0]])  # radius=0.4, half_height=0.5
    
    penetration, normal, contact_point = capsule_capsule_test(x_a, x_b, q_a, q_b, params_a, params_b)
    
    # Expected: distance = 1.0, radii sum = 0.3 + 0.4 = 0.7
    # Penetration = 0.7 - 1.0 = -0.3 (no collision)
    assert penetration.numpy()[0] < 0


def test_capsule_box_collision():
    """Test capsule-box collision detection."""
    # Horizontal capsule, box below
    x_capsule = Tensor([[0.0, 2.0, 0.0]])
    x_box = Tensor([[0.0, 0.0, 0.0]])
    # Rotate capsule 90 degrees around Z (horizontal)
    angle = np.pi / 2
    q_capsule = Tensor([[np.cos(angle/2), 0.0, 0.0, np.sin(angle/2)]])
    q_box = Tensor([[1.0, 0.0, 0.0, 0.0]])
    params_capsule = Tensor([[0.3, 1.0, 0.0]])  # radius=0.3, half_height=1.0
    params_box = Tensor([[1.0, 0.5, 1.0]])  # Half-extents
    
    penetration, normal, contact_point = capsule_box_test(x_capsule, x_box, q_capsule, q_box, params_capsule, params_box)
    
    # Capsule bottom at y = 2.0 - 0.3 = 1.7
    # Box top at y = 0.5
    # Distance = 1.7 - 0.5 = 1.2
    # Penetration = 0.3 - 1.2 = -0.9 (no collision)
    assert penetration.numpy()[0] < 0


def test_rotated_box_collision():
    """Test collision between rotated boxes."""
    x_a = Tensor([[0.0, 0.0, 0.0]])
    x_b = Tensor([[1.5, 0.0, 0.0]])
    # Rotate box B by 45 degrees around Y
    angle = np.pi / 4
    q_a = Tensor([[1.0, 0.0, 0.0, 0.0]])
    q_b = Tensor([[np.cos(angle/2), 0.0, np.sin(angle/2), 0.0]])
    params_a = Tensor([[1.0, 1.0, 1.0]])
    params_b = Tensor([[1.0, 1.0, 1.0]])
    
    penetration, normal, contact_point = box_box_test(x_a, x_b, q_a, q_b, params_a, params_b)
    
    # With rotation, the effective width of box B increases
    # Should have collision
    assert penetration.numpy()[0] > 0


def test_multiple_bodies_soa():
    """Test collision detection with multiple bodies in SoA format."""
    # Three box-sphere pairs
    x_box = Tensor([
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [10.0, 0.0, 0.0]
    ])
    x_sphere = Tensor([
        [1.2, 0.0, 0.0],  # Will collide
        [7.0, 0.0, 0.0],  # No collision
        [11.0, 0.0, 0.0]  # Will collide
    ])
    q_box = Tensor([[1.0, 0.0, 0.0, 0.0]] * 3)
    q_sphere = Tensor([[1.0, 0.0, 0.0, 0.0]] * 3)  # Identity quaternions
    params_box = Tensor([[1.0, 1.0, 1.0]] * 3)
    params_sphere = Tensor([
        [0.5, 0.0, 0.0],  # radius = 0.5
        [0.3, 0.0, 0.0],  # radius = 0.3
        [0.4, 0.0, 0.0]   # radius = 0.4
    ])
    
    penetration, normal, contact_point = box_sphere_test(x_box, x_sphere, q_box, q_sphere, params_box, params_sphere)
    
    # Check first pair: penetration = 0.5 - (1.2 - 1.0) = 0.3
    assert np.isclose(penetration.numpy()[0], 0.3, atol=1e-6)
    
    # Check second pair: no collision
    assert penetration.numpy()[1] < 0
    
    # Check third pair: penetration = 0.4 - (11.0 - 10.0 - 1.0) = 0.4
    assert np.isclose(penetration.numpy()[2], 0.4, atol=1e-6)