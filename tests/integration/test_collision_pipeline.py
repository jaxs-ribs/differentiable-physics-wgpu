"""Integration tests for the collision detection pipeline (broadphase + narrowphase)."""
import pytest
import numpy as np
from tinygrad import Tensor
from physics.xpbd.broadphase import uniform_spatial_hash
from physics.xpbd.narrowphase import generate_contacts
from physics.types import ShapeType, Contact
from scripts.scene_builder import SceneBuilder


def test_broadphase_narrowphase_integration():
    """Test that broadphase and narrowphase work together correctly."""
    # Create positions for spheres that should collide
    positions = np.array([
        [0.0, 0.0, 0.0],   # Sphere 1
        [1.5, 0.0, 0.0],   # Sphere 2 - overlapping with 1
        [10.0, 0.0, 0.0],  # Sphere 3 - far away
    ], dtype=np.float32)
    
    # All spheres with radius 1
    shape_types = np.array([ShapeType.SPHERE, ShapeType.SPHERE, ShapeType.SPHERE], dtype=np.int32)
    shape_params = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=np.float32)
    
    # Identity quaternions
    orientations = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32)
    
    # Convert to tensors
    x = Tensor(positions)
    q = Tensor(orientations)
    shape_type = Tensor(shape_types)
    shape_params_tensor = Tensor(shape_params)
    
    # Run broadphase
    candidate_pairs = uniform_spatial_hash(x, shape_type, shape_params_tensor)
    
    # Broadphase should find at least the (0,1) pair
    pairs_np = candidate_pairs.numpy()
    assert len(pairs_np) > 0
    
    # Check that spheres 0 and 1 are paired
    has_collision_pair = any(
        (min(p) == 0 and max(p) == 1) for p in pairs_np
    )
    assert has_collision_pair
    
    # Run narrowphase
    contacts = generate_contacts(x, q, candidate_pairs, shape_type, shape_params_tensor)
    
    # Should generate contacts for overlapping spheres
    assert len(contacts) > 0
    
    # Find the contact between spheres 0 and 1
    contact_01 = None
    for contact in contacts:
        if contact.pair_indices == (0, 1) or contact.pair_indices == (1, 0):
            contact_01 = contact
            break
    
    assert contact_01 is not None
    assert contact_01.depth > 0  # Should have penetration
    assert np.allclose(np.linalg.norm(contact_01.normal), 1.0)  # Normal should be unit vector


def test_no_false_positives():
    """Test that distant objects don't generate collisions."""
    # Create well-separated spheres
    positions = np.array([
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 5.0],
    ], dtype=np.float32)
    
    shape_types = np.array([ShapeType.SPHERE] * 4, dtype=np.int32)
    shape_params = np.array([[0.5, 0, 0]] * 4, dtype=np.float32)  # Small radius
    orientations = np.array([[1, 0, 0, 0]] * 4, dtype=np.float32)
    
    x = Tensor(positions)
    q = Tensor(orientations)
    shape_type = Tensor(shape_types)
    shape_params_tensor = Tensor(shape_params)
    
    # Run collision pipeline
    candidate_pairs = uniform_spatial_hash(x, shape_type, shape_params_tensor)
    contacts = generate_contacts(x, q, candidate_pairs, shape_type, shape_params_tensor)
    
    # Should have no contacts (all spheres are well separated)
    assert len(contacts) == 0


def test_multiple_collisions():
    """Test handling multiple simultaneous collisions."""
    # Create a cluster of overlapping spheres
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.25, 0.43, 0.0],  # Forms triangle with first two
    ], dtype=np.float32)
    
    shape_types = np.array([ShapeType.SPHERE] * 3, dtype=np.int32)
    shape_params = np.array([[0.5, 0, 0]] * 3, dtype=np.float32)
    orientations = np.array([[1, 0, 0, 0]] * 3, dtype=np.float32)
    
    x = Tensor(positions)
    q = Tensor(orientations)
    shape_type = Tensor(shape_types)
    shape_params_tensor = Tensor(shape_params)
    
    # Run collision pipeline
    candidate_pairs = uniform_spatial_hash(x, shape_type, shape_params_tensor)
    contacts = generate_contacts(x, q, candidate_pairs, shape_type, shape_params_tensor)
    
    # Should have 3 contacts (0-1, 0-2, 1-2)
    assert len(contacts) == 3
    
    # Verify all pairs are detected
    detected_pairs = set()
    for contact in contacts:
        pair = tuple(sorted(contact.pair_indices))
        detected_pairs.add(pair)
    
    expected_pairs = {(0, 1), (0, 2), (1, 2)}
    assert detected_pairs == expected_pairs


def test_mixed_shapes_collision():
    """Test collision detection between different shape types."""
    # Sphere and box collision
    positions = np.array([
        [0.0, 0.0, 0.0],   # Sphere at origin
        [1.5, 0.0, 0.0],   # Box close to sphere
    ], dtype=np.float32)
    
    shape_types = np.array([ShapeType.SPHERE, ShapeType.BOX], dtype=np.int32)
    shape_params = np.array([
        [1.0, 0, 0],      # Sphere radius 1
        [1.0, 1.0, 1.0],  # Box half-extents
    ], dtype=np.float32)
    orientations = np.array([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32)
    
    x = Tensor(positions)
    q = Tensor(orientations)
    shape_type = Tensor(shape_types)
    shape_params_tensor = Tensor(shape_params)
    
    # Run collision pipeline
    candidate_pairs = uniform_spatial_hash(x, shape_type, shape_params_tensor)
    
    # Broadphase should find the pair
    assert len(candidate_pairs.numpy()) > 0
    
    contacts = generate_contacts(x, q, candidate_pairs, shape_type, shape_params_tensor)
    
    # Should detect collision between sphere and box
    assert len(contacts) > 0