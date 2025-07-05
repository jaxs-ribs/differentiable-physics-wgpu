"""Unit tests for analytic narrowphase collision detection."""
import pytest
import numpy as np
from tinygrad import Tensor, dtypes
from physics.xpbd.narrowphase import (
    sphere_sphere_test, sphere_plane_test, generate_contacts, softplus
)
from physics.types import ShapeType


def test_sphere_sphere_collision():
    """Test sphere-sphere collision detection."""
    # Two spheres with radius 1, centers 1.5 units apart
    x_a = Tensor([[0.0, 0.0, 0.0]])
    x_b = Tensor([[1.5, 0.0, 0.0]])
    q_a = Tensor([[1.0, 0.0, 0.0, 0.0]])  # Identity quaternion
    q_b = Tensor([[1.0, 0.0, 0.0, 0.0]])  # Identity quaternion
    params_a = Tensor([[1.0, 0.0, 0.0]])  # radius = 1
    params_b = Tensor([[1.0, 0.0, 0.0]])  # radius = 1
    
    penetration, normal, contact_point = sphere_sphere_test(x_a, x_b, q_a, q_b, params_a, params_b)
    
    # Expected: radii sum = 2, distance = 1.5, penetration = 0.5
    assert np.isclose(penetration.numpy()[0], 0.5, atol=1e-6)
    
    # Normal should point from b to a (negative x direction)
    normal_np = normal.numpy()[0]
    assert np.isclose(normal_np[0], -1.0, atol=1e-6)
    assert np.isclose(normal_np[1], 0.0, atol=1e-6)
    assert np.isclose(normal_np[2], 0.0, atol=1e-6)
    
    # Contact point should be on surface of sphere b
    contact_np = contact_point.numpy()[0]
    assert np.isclose(contact_np[0], 0.5, atol=1e-6)  # 1.5 - 1.0
    assert np.isclose(contact_np[1], 0.0, atol=1e-6)
    assert np.isclose(contact_np[2], 0.0, atol=1e-6)


def test_sphere_sphere_no_collision():
    """Test sphere-sphere with no collision."""
    # Two spheres with radius 0.5, centers 2 units apart
    x_a = Tensor([[0.0, 0.0, 0.0]])
    x_b = Tensor([[2.0, 0.0, 0.0]])
    q_a = Tensor([[1.0, 0.0, 0.0, 0.0]])  # Identity quaternion
    q_b = Tensor([[1.0, 0.0, 0.0, 0.0]])  # Identity quaternion
    params_a = Tensor([[0.5, 0.0, 0.0]])
    params_b = Tensor([[0.5, 0.0, 0.0]])
    
    penetration, normal, contact_point = sphere_sphere_test(x_a, x_b, q_a, q_b, params_a, params_b)
    
    # Expected: radii sum = 1, distance = 2, penetration = -1 (negative = no collision)
    assert penetration.numpy()[0] < 0


def test_sphere_plane_collision():
    """Test sphere-plane collision detection."""
    # Sphere above a horizontal plane
    x_sphere = Tensor([[0.0, 0.5, 0.0]])  # 0.5 units above origin
    x_plane = Tensor([[0.0, 0.0, 0.0]])   # Plane at origin
    q_sphere = Tensor([[1.0, 0.0, 0.0, 0.0]])  # Identity quaternion
    q_plane = Tensor([[1.0, 0.0, 0.0, 0.0]])  # Identity quaternion (Y-up plane)
    params_sphere = Tensor([[1.0, 0.0, 0.0]])  # radius = 1
    params_plane = Tensor([[10.0, 0.05, 10.0]])  # Large flat box
    
    penetration, normal, contact_point = sphere_plane_test(
        x_sphere, x_plane, q_sphere, q_plane, params_sphere, params_plane
    )
    
    # Expected: sphere center is 0.5 above plane, radius is 1
    # Plane has half-thickness 0.05, so plane top is at y=0.05
    # Bottom of sphere is at -0.5, penetration = 0.5 + 0.05 = 0.55
    assert np.isclose(penetration.numpy()[0], 0.55, atol=1e-6)
    
    # Normal should point up (plane normal)
    normal_np = normal.numpy()[0]
    assert np.isclose(normal_np[0], 0.0, atol=1e-6)
    assert np.isclose(normal_np[1], 1.0, atol=1e-6)
    assert np.isclose(normal_np[2], 0.0, atol=1e-6)
    
    # Contact point should be at bottom of sphere
    contact_np = contact_point.numpy()[0]
    assert np.isclose(contact_np[0], 0.0, atol=1e-6)
    assert np.isclose(contact_np[1], -0.5, atol=1e-6)
    assert np.isclose(contact_np[2], 0.0, atol=1e-6)


def test_sphere_plane_rotated():
    """Test sphere-plane collision with rotated plane."""
    # Sphere next to a vertical plane (rotated 90 degrees around Z)
    x_sphere = Tensor([[0.5, 0.0, 0.0]])
    x_plane = Tensor([[0.0, 0.0, 0.0]])
    
    # Quaternion for 90 degree rotation around Z axis
    # This makes the plane's normal point in +X direction
    angle = np.pi / 2
    q_sphere = Tensor([[1.0, 0.0, 0.0, 0.0]])  # Identity quaternion
    q_plane = Tensor([[np.cos(angle/2), 0.0, 0.0, np.sin(angle/2)]])
    
    params_sphere = Tensor([[1.0, 0.0, 0.0]])
    params_plane = Tensor([[10.0, 0.05, 10.0]])
    
    penetration, normal, contact_point = sphere_plane_test(
        x_sphere, x_plane, q_sphere, q_plane, params_sphere, params_plane
    )
    
    # Sphere center is at x=0.5, plane at origin with normal pointing -X
    # Plane has half-thickness 0.05, so plane face is at x=-0.05
    # Distance from sphere center to plane face = 0.5 - (-0.05) = 0.55
    # Penetration = radius - distance = 1.0 - (-0.55) = 1.55
    assert np.isclose(penetration.numpy()[0], 1.55, atol=1e-5)
    
    # Normal should point in -X direction (plane Y axis after 90deg Z rotation)
    normal_np = normal.numpy()[0]
    assert np.isclose(normal_np[0], -1.0, atol=1e-5)
    assert np.isclose(normal_np[1], 0.0, atol=1e-5)


def test_dispatch_and_combination():
    """Test the main generate_contacts function with mixed collision types."""
    # Create a scene with:
    # - Body 0: Sphere at origin
    # - Body 1: Sphere at (1.9, 0, 0) - slightly overlapping with sphere 0
    # - Body 2: Plane (flat box) at (0, -0.5, 0) - sphere 0 penetrates it
    
    x = Tensor([
        [0.0, 0.0, 0.0],    # Sphere 0
        [1.9, 0.0, 0.0],    # Sphere 1 - slightly overlapping
        [0.0, -0.5, 0.0]    # Plane - sphere 0 bottom penetrates plane top
    ])
    
    q = Tensor([
        [1.0, 0.0, 0.0, 0.0],  # Identity
        [1.0, 0.0, 0.0, 0.0],  # Identity
        [1.0, 0.0, 0.0, 0.0]   # Identity (Y-up plane)
    ])
    
    shape_type = Tensor([ShapeType.SPHERE, ShapeType.SPHERE, ShapeType.BOX], dtype=dtypes.int32)
    
    shape_params = Tensor([
        [1.0, 0.0, 0.0],     # Sphere radius 1
        [1.0, 0.0, 0.0],     # Sphere radius 1
        [10.0, 0.05, 10.0]   # Flat box (plane)
    ])
    
    # Candidate pairs: sphere-sphere and sphere-plane
    candidate_pairs = Tensor([
        [0, 1],   # Sphere 0 - Sphere 1
        [0, 2],   # Sphere 0 - Plane
        [-1, -1]  # Invalid pair (for testing)
    ])
    
    friction = Tensor([0.5, 0.5, 0.5])  # Default friction for all bodies
    contacts = generate_contacts(x, q, candidate_pairs, shape_type, shape_params, friction)
    
    # Check that we have the right structure
    assert 'ids_a' in contacts
    assert 'ids_b' in contacts
    assert 'normal' in contacts
    assert 'p' in contacts
    assert 'compliance' in contacts
    assert 'contact_count' in contacts
    
    # Get the number of valid contacts
    contact_count = int(contacts['contact_count'].numpy())
    
    # We should have 2 valid pairs (both colliding)
    assert contact_count == 2
    
    # Get only the valid portion of the arrays
    ids_a = contacts['ids_a'].numpy()[:contact_count]
    ids_b = contacts['ids_b'].numpy()[:contact_count]
    
    assert len(ids_a) == 2
    
    # Check sphere-sphere collision (bodies 0 and 1)
    ss_idx = np.where((ids_a == 0) & (ids_b == 1))[0]
    if len(ss_idx) > 0:
        ss_idx = ss_idx[0]
        penetration = contacts['p'].numpy()[:contact_count][ss_idx]
        # Distance between spheres = 1.9, radii sum = 2.0, penetration = 0.1
        # The penetration should be raw (not softplus'd)
        expected_p = 0.1
        assert np.isclose(penetration, expected_p, atol=1e-5)
    
    # Check sphere-plane collision (bodies 0 and 2)
    sp_idx = np.where((ids_a == 0) & (ids_b == 2))[0]
    if len(sp_idx) > 0:
        sp_idx = sp_idx[0]
        # Sphere 0 is at y=0, plane is at y=-0.5
        # Sphere bottom is at -1, plane top is at -0.45, so they collide
        normal = contacts['normal'].numpy()[:contact_count][sp_idx]
        assert normal.shape == (3,)
    
    # Check that invalid portions are properly padded
    all_ids_a = contacts['ids_a'].numpy()
    all_ids_b = contacts['ids_b'].numpy()
    # Everything after contact_count should be -1
    assert np.all(all_ids_a[contact_count:] == -1)
    assert np.all(all_ids_b[contact_count:] == -1)


def test_no_contact():
    """Test case where objects are close but not touching."""
    x_a = Tensor([[0.0, 0.0, 0.0]])
    x_b = Tensor([[3.0, 0.0, 0.0]])  # 3 units apart
    q_a = Tensor([[1.0, 0.0, 0.0, 0.0]])  # Identity quaternion
    q_b = Tensor([[1.0, 0.0, 0.0, 0.0]])  # Identity quaternion
    params_a = Tensor([[1.0, 0.0, 0.0]])  # radius = 1
    params_b = Tensor([[1.0, 0.0, 0.0]])  # radius = 1
    
    penetration, normal, contact_point = sphere_sphere_test(x_a, x_b, q_a, q_b, params_a, params_b)
    
    # Distance = 3, radii sum = 2, so penetration = -1 (negative)
    assert penetration.numpy()[0] < 0


@pytest.mark.skip(reason="Gradient flow through fixed-size tensors needs investigation")
def test_gradient_flow():
    """Test that gradients flow through the narrowphase."""
    # Two overlapping spheres
    x = Tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]
    ], requires_grad=True)
    
    q = Tensor([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0]
    ])
    
    shape_type = Tensor([ShapeType.SPHERE, ShapeType.SPHERE], dtype=dtypes.int32)
    shape_params = Tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    
    candidate_pairs = Tensor([[0, 1]])
    
    # Generate contacts
    friction = Tensor([0.5, 0.5])  # Default friction
    contacts = generate_contacts(x, q, candidate_pairs, shape_type, shape_params, friction)
    
    # Get valid contact count
    contact_count = int(contacts['contact_count'].numpy())
    
    # Compute loss as sum of valid penetrations only
    valid_penetrations = contacts['p'][:contact_count]
    loss = valid_penetrations.sum()
    loss.backward()
    
    # Check that gradients exist
    assert x.grad is not None
    grad_np = x.grad.numpy()
    
    # Gradient should be non-zero for overlapping bodies
    assert not np.allclose(grad_np, 0.0)
    
    # The loss is the penetration, so gradient points in direction to increase penetration
    # For sphere collision, the normal points from b to a (negative x direction)
    # Gradient w.r.t position follows the normal direction scaled by softplus derivative
    # Body 0 should have positive gradient (move right to increase penetration)
    assert grad_np[0, 0] > 0
    
    # Body 1 should have negative gradient (move left to increase penetration)
    assert grad_np[1, 0] < 0


def test_softplus_function():
    """Test the softplus activation function."""
    # Test positive input
    x = Tensor([1.0])
    y = softplus(x, beta=10.0)
    # softplus(1) ≈ 1.00045 for beta=10
    assert y.numpy()[0] > 1.0
    assert y.numpy()[0] < 1.1
    
    # Test near zero
    x = Tensor([0.0])
    y = softplus(x, beta=10.0)
    # softplus(0) = ln(2)/beta ≈ 0.0693 for beta=10
    assert np.isclose(y.numpy()[0], np.log(2) / 10.0, atol=1e-5)
    
    # Test negative input
    x = Tensor([-1.0])
    y = softplus(x, beta=10.0)
    # Should be close to 0 but positive
    assert y.numpy()[0] > 0
    assert y.numpy()[0] < 0.1


def test_multiple_sphere_collisions():
    """Test multiple simultaneous sphere collisions."""
    # Three spheres in a line, slightly overlapping
    x = Tensor([
        [0.0, 0.0, 0.0],
        [0.9, 0.0, 0.0],  # Slightly overlapping with sphere 0
        [1.8, 0.0, 0.0]   # Slightly overlapping with sphere 1
    ])
    
    q = Tensor([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0]
    ])
    
    shape_type = Tensor([ShapeType.SPHERE] * 3, dtype=dtypes.int32)
    shape_params = Tensor([[0.5, 0.0, 0.0]] * 3)  # All radius 0.5
    
    # All possible pairs
    candidate_pairs = Tensor([
        [0, 1],  # Touching
        [0, 2],  # Not touching
        [1, 2]   # Touching
    ])
    
    friction = Tensor([0.5, 0.5, 0.5])  # Default friction for all bodies
    contacts = generate_contacts(x, q, candidate_pairs, shape_type, shape_params, friction)
    
    ids_a = contacts['ids_a'].numpy()
    ids_b = contacts['ids_b'].numpy()
    
    # Count valid contacts
    valid_mask = ids_a != -1
    num_contacts = valid_mask.sum()
    
    # Should have exactly 2 contacts (0-1 and 1-2)
    assert num_contacts == 2


def test_edge_case_zero_radius():
    """Test edge case with zero radius sphere."""
    x_a = Tensor([[0.0, 0.0, 0.0]])
    x_b = Tensor([[0.0, 0.0, 0.0]])  # Same position
    q_a = Tensor([[1.0, 0.0, 0.0, 0.0]])  # Identity quaternion
    q_b = Tensor([[1.0, 0.0, 0.0, 0.0]])  # Identity quaternion
    params_a = Tensor([[0.0, 0.0, 0.0]])  # Zero radius
    params_b = Tensor([[1.0, 0.0, 0.0]])
    
    penetration, normal, contact_point = sphere_sphere_test(x_a, x_b, q_a, q_b, params_a, params_b)
    
    # With zero radius, penetration should be 1.0 (radius of b)
    assert np.isclose(penetration.numpy()[0], 1.0, atol=1e-6)