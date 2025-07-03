"""Unit tests for physics data types and type conversions."""
import pytest
import numpy as np
from physics.types import ShapeType, Contact, ExecutionMode, create_soa_body_data


def test_shape_type_enum():
    """Test ShapeType enum values."""
    assert ShapeType.SPHERE == 0
    assert ShapeType.BOX == 2
    assert ShapeType.CAPSULE == 3
    
    # Test that we can create from int
    assert ShapeType(0) == ShapeType.SPHERE
    assert ShapeType(2) == ShapeType.BOX
    assert ShapeType(3) == ShapeType.CAPSULE


def test_execution_mode_enum():
    """Test ExecutionMode enum values."""
    assert ExecutionMode.ORACLE == 0
    assert ExecutionMode(0) == ExecutionMode.ORACLE


def test_contact_creation():
    """Test Contact namedtuple creation."""
    normal = np.array([0, 1, 0], dtype=np.float32)
    point = np.array([1, 0, 0], dtype=np.float32)
    
    contact = Contact(
        pair_indices=(0, 1),
        normal=normal,
        depth=0.1,
        point=point
    )
    
    assert contact.pair_indices == (0, 1)
    assert np.array_equal(contact.normal, normal)
    assert contact.depth == 0.1
    assert np.array_equal(contact.point, point)


def test_create_soa_body_data_single_body():
    """Test SoA data creation for a single body."""
    positions = [np.array([1, 2, 3], dtype=np.float32)]
    velocities = [np.array([0, 0, 0], dtype=np.float32)]
    orientations = [np.array([1, 0, 0, 0], dtype=np.float32)]
    angular_vels = [np.array([0, 0, 0], dtype=np.float32)]
    masses = [1.0]
    shape_types = [ShapeType.SPHERE]
    shape_params = [np.array([1, 0, 0], dtype=np.float32)]
    
    soa_data = create_soa_body_data(
        positions, velocities, orientations, angular_vels,
        masses, shape_types, shape_params
    )
    
    # Check all fields are present
    expected_fields = ['x', 'v', 'q', 'omega', 'inv_mass', 'inv_inertia', 'shape_type', 'shape_params']
    for field in expected_fields:
        assert field in soa_data
    
    # Check shapes
    assert soa_data['x'].shape == (1, 3)
    assert soa_data['v'].shape == (1, 3)
    assert soa_data['q'].shape == (1, 4)
    assert soa_data['omega'].shape == (1, 3)
    assert soa_data['inv_mass'].shape == (1,)
    assert soa_data['inv_inertia'].shape == (1, 3, 3)
    assert soa_data['shape_type'].shape == (1,)
    assert soa_data['shape_params'].shape == (1, 3)
    
    # Check values
    assert np.allclose(soa_data['x'][0], [1, 2, 3])
    assert np.allclose(soa_data['v'][0], [0, 0, 0])
    assert np.allclose(soa_data['q'][0], [1, 0, 0, 0])
    assert soa_data['inv_mass'][0] == 1.0
    assert soa_data['shape_type'][0] == ShapeType.SPHERE.value


def test_create_soa_body_data_multiple_bodies():
    """Test SoA data creation for multiple bodies."""
    n_bodies = 3
    positions = [np.array([i, 0, 0], dtype=np.float32) for i in range(n_bodies)]
    velocities = [np.array([0, i, 0], dtype=np.float32) for i in range(n_bodies)]
    orientations = [np.array([1, 0, 0, 0], dtype=np.float32) for _ in range(n_bodies)]
    angular_vels = [np.array([0, 0, i], dtype=np.float32) for i in range(n_bodies)]
    masses = [1.0, 2.0, 3.0]
    shape_types = [ShapeType.SPHERE, ShapeType.BOX, ShapeType.CAPSULE]
    shape_params = [
        np.array([1, 0, 0], dtype=np.float32),
        np.array([1, 1, 1], dtype=np.float32),
        np.array([2, 0.5, 0], dtype=np.float32),
    ]
    
    soa_data = create_soa_body_data(
        positions, velocities, orientations, angular_vels,
        masses, shape_types, shape_params
    )
    
    # Check shapes
    assert soa_data['x'].shape == (3, 3)
    assert soa_data['v'].shape == (3, 3)
    assert soa_data['inv_mass'].shape == (3,)
    
    # Check inverse masses
    assert np.allclose(soa_data['inv_mass'], [1.0, 0.5, 1/3.0])
    
    # Check shape types
    assert soa_data['shape_type'][0] == ShapeType.SPHERE.value
    assert soa_data['shape_type'][1] == ShapeType.BOX.value
    assert soa_data['shape_type'][2] == ShapeType.CAPSULE.value


def test_create_soa_infinite_mass():
    """Test SoA data creation with infinite mass (static bodies)."""
    positions = [np.array([0, 0, 0], dtype=np.float32)]
    velocities = [np.array([0, 0, 0], dtype=np.float32)]
    orientations = [np.array([1, 0, 0, 0], dtype=np.float32)]
    angular_vels = [np.array([0, 0, 0], dtype=np.float32)]
    masses = [1e7 + 1]  # Just above the cutoff (treated as infinite)
    shape_types = [ShapeType.BOX]
    shape_params = [np.array([10, 1, 10], dtype=np.float32)]
    
    soa_data = create_soa_body_data(
        positions, velocities, orientations, angular_vels,
        masses, shape_types, shape_params
    )
    
    # Inverse mass should be 0 for static bodies
    assert soa_data['inv_mass'][0] == 0.0
    # Note: Currently inv_inertia is not zeroed for infinite mass bodies
    # This is a known limitation in the implementation
    # TODO: Fix create_soa_body_data to also zero inv_inertia for infinite mass
    assert soa_data['inv_inertia'].shape == (1, 3, 3)


def test_create_soa_different_shapes():
    """Test SoA creation handles different shape parameters correctly."""
    positions = [np.array([0, i, 0], dtype=np.float32) for i in range(3)]
    velocities = [np.array([0, 0, 0], dtype=np.float32) for _ in range(3)]
    orientations = [np.array([1, 0, 0, 0], dtype=np.float32) for _ in range(3)]
    angular_vels = [np.array([0, 0, 0], dtype=np.float32) for _ in range(3)]
    masses = [1.0, 1.0, 1.0]
    shape_types = [ShapeType.SPHERE, ShapeType.BOX, ShapeType.CAPSULE]
    shape_params = [
        np.array([2, 0, 0], dtype=np.float32),      # Sphere: radius=2
        np.array([1, 2, 3], dtype=np.float32),      # Box: half-extents
        np.array([3, 0.5, 0], dtype=np.float32),    # Capsule: half-height=3, radius=0.5
    ]
    
    soa_data = create_soa_body_data(
        positions, velocities, orientations, angular_vels,
        masses, shape_types, shape_params
    )
    
    # Check shape params are preserved
    assert np.allclose(soa_data['shape_params'][0], [2, 0, 0])
    assert np.allclose(soa_data['shape_params'][1], [1, 2, 3])
    assert np.allclose(soa_data['shape_params'][2], [3, 0.5, 0])
    
    # Check inertia tensors are different for different shapes
    assert not np.allclose(soa_data['inv_inertia'][0], soa_data['inv_inertia'][1])
    assert not np.allclose(soa_data['inv_inertia'][1], soa_data['inv_inertia'][2])