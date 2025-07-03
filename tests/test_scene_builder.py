"""Tests for SceneBuilder utility."""
import pytest
import numpy as np
from scripts.scene_builder import SceneBuilder
from physics.types import ShapeType


def test_scene_builder_creation():
    """Test that SceneBuilder can be created and initialized."""
    builder = SceneBuilder()
    assert builder.count() == 0
    assert len(builder.positions) == 0
    assert len(builder.masses) == 0


def test_add_single_body():
    """Test adding a single body to the scene."""
    builder = SceneBuilder()
    
    builder.add_body(
        position=[1, 2, 3],
        mass=1.5,
        shape_type=ShapeType.SPHERE,
        shape_params=[0.5, 0, 0]
    )
    
    assert builder.count() == 1
    assert np.allclose(builder.positions[0], [1, 2, 3])
    assert builder.masses[0] == 1.5
    assert builder.shape_types[0] == ShapeType.SPHERE
    assert np.allclose(builder.shape_params[0], [0.5, 0, 0])


def test_add_body_with_defaults():
    """Test that default values are applied correctly."""
    builder = SceneBuilder()
    
    builder.add_body(
        position=[0, 0, 0],
        mass=1.0,
        shape_type=ShapeType.BOX,
        shape_params=[1, 1, 1]
    )
    
    # Check defaults
    assert np.allclose(builder.velocities[0], [0, 0, 0])
    assert np.allclose(builder.orientations[0], [1, 0, 0, 0])  # Identity quaternion
    assert np.allclose(builder.angular_vels[0], [0, 0, 0])


def test_add_body_with_all_parameters():
    """Test adding a body with all parameters specified."""
    builder = SceneBuilder()
    
    builder.add_body(
        position=[1, 2, 3],
        velocity=[0.1, 0.2, 0.3],
        orientation=[0.707, 0.707, 0, 0],  # 90 degree rotation around x
        angular_vel=[0.5, 0, 0],
        mass=2.0,
        shape_type=ShapeType.CAPSULE,
        shape_params=[1.5, 0.5, 0]
    )
    
    assert np.allclose(builder.positions[0], [1, 2, 3])
    assert np.allclose(builder.velocities[0], [0.1, 0.2, 0.3])
    assert np.allclose(builder.angular_vels[0], [0.5, 0, 0])
    assert builder.masses[0] == 2.0
    assert builder.shape_types[0] == ShapeType.CAPSULE


def test_method_chaining():
    """Test that add_body returns self for method chaining."""
    builder = SceneBuilder()
    
    result = builder.add_body(
        position=[0, 0, 0],
        mass=1.0,
        shape_type=ShapeType.SPHERE,
        shape_params=[1, 0, 0]
    ).add_body(
        position=[1, 1, 1],
        mass=2.0,
        shape_type=ShapeType.BOX,
        shape_params=[1, 1, 1]
    )
    
    assert result is builder
    assert builder.count() == 2


def test_build_scene():
    """Test building a scene and getting SoA data."""
    builder = SceneBuilder()
    
    builder.add_body(
        position=[0, 0, 0],
        mass=1.0,
        shape_type=ShapeType.SPHERE,
        shape_params=[1, 0, 0]
    ).add_body(
        position=[1, 1, 1],
        velocity=[1, 0, 0],
        mass=2.0,
        shape_type=ShapeType.BOX,
        shape_params=[0.5, 0.5, 0.5]
    )
    
    soa_data = builder.build()
    
    # Check that all expected keys are present
    expected_keys = ['x', 'v', 'q', 'omega', 'inv_mass', 'inv_inertia', 'shape_type', 'shape_params']
    assert all(key in soa_data for key in expected_keys)
    
    # Check shapes
    assert soa_data['x'].shape == (2, 3)  # 2 bodies, 3D positions
    assert soa_data['v'].shape == (2, 3)  # 2 bodies, 3D velocities
    assert soa_data['q'].shape == (2, 4)  # 2 bodies, quaternions
    assert soa_data['inv_mass'].shape == (2,)  # 2 inverse masses


def test_quaternion_normalization():
    """Test that quaternions are automatically normalized."""
    builder = SceneBuilder()
    
    # Add body with unnormalized quaternion
    builder.add_body(
        position=[0, 0, 0],
        orientation=[2, 0, 0, 0],  # Not normalized
        mass=1.0,
        shape_type=ShapeType.SPHERE,
        shape_params=[1, 0, 0]
    )
    
    # Check that quaternion was normalized
    quat = builder.orientations[0]
    assert np.isclose(np.linalg.norm(quat), 1.0)


def test_clear_scene():
    """Test clearing the scene."""
    builder = SceneBuilder()
    
    builder.add_body(
        position=[0, 0, 0],
        mass=1.0,
        shape_type=ShapeType.SPHERE,
        shape_params=[1, 0, 0]
    )
    
    assert builder.count() == 1
    
    result = builder.clear()
    assert result is builder  # Should return self
    assert builder.count() == 0


def test_validation_negative_mass():
    """Test that negative mass raises ValueError."""
    builder = SceneBuilder()
    
    with pytest.raises(ValueError, match="Mass must be positive"):
        builder.add_body(
            position=[0, 0, 0],
            mass=-1.0,
            shape_type=ShapeType.SPHERE,
            shape_params=[1, 0, 0]
        )


def test_validation_invalid_position():
    """Test that invalid position raises ValueError."""
    builder = SceneBuilder()
    
    with pytest.raises(ValueError, match="Position must be a 3D vector"):
        builder.add_body(
            position=[0, 0],  # Only 2D
            mass=1.0,
            shape_type=ShapeType.SPHERE,
            shape_params=[1, 0, 0]
        )


def test_validation_invalid_velocity():
    """Test that invalid velocity raises ValueError."""
    builder = SceneBuilder()
    
    with pytest.raises(ValueError, match="Velocity must be a 3D vector"):
        builder.add_body(
            position=[0, 0, 0],
            velocity=[1, 2, 3, 4],  # 4D instead of 3D
            mass=1.0,
            shape_type=ShapeType.SPHERE,
            shape_params=[1, 0, 0]
        )


def test_validation_invalid_quaternion():
    """Test that invalid quaternion raises ValueError."""
    builder = SceneBuilder()
    
    with pytest.raises(ValueError, match="Orientation must be a 4D quaternion"):
        builder.add_body(
            position=[0, 0, 0],
            orientation=[1, 0, 0],  # Only 3D
            mass=1.0,
            shape_type=ShapeType.SPHERE,
            shape_params=[1, 0, 0]
        )


def test_validation_invalid_shape_params():
    """Test that invalid shape params raise ValueError."""
    builder = SceneBuilder()
    
    with pytest.raises(ValueError, match="Shape params must be a 3D vector"):
        builder.add_body(
            position=[0, 0, 0],
            mass=1.0,
            shape_type=ShapeType.SPHERE,
            shape_params=[1]  # Only 1D
        )


def test_validation_empty_scene_build():
    """Test that building empty scene raises ValueError."""
    builder = SceneBuilder()
    
    with pytest.raises(ValueError, match="Cannot build empty scene"):
        builder.build()


def test_validation_invalid_shape_type():
    """Test that invalid shape type raises ValueError."""
    builder = SceneBuilder()
    
    with pytest.raises(ValueError, match="Shape type must be a ShapeType enum value"):
        builder.add_body(
            position=[0, 0, 0],
            mass=1.0,
            shape_type="invalid",  # Not a ShapeType
            shape_params=[1, 0, 0]
        )