"""Tests for XPBD engine creation and initialization."""
import pytest
import numpy as np
from physics.engine import TensorPhysicsEngine
from physics.types import ShapeType
from scripts.scene_builder import SceneBuilder


def test_engine_creation_single_body():
    """Test that XPBD engine can be created with a single body."""
    # Create simple scene with one sphere using SceneBuilder
    builder = SceneBuilder()
    builder.add_body(
        position=[0, 0, 0],
        mass=1.0,
        shape_type=ShapeType.SPHERE,
        shape_params=[1, 0, 0]
    )
    
    soa_data = builder.build()
    
    engine = TensorPhysicsEngine(
        x=soa_data['x'],
        q=soa_data['q'],
        v=soa_data['v'],
        omega=soa_data['omega'],
        inv_mass=soa_data['inv_mass'],
        inv_inertia=soa_data['inv_inertia'],
        shape_type=soa_data['shape_type'],
        shape_params=soa_data['shape_params']
    )
    assert engine is not None
    assert engine.dt == 0.016  # default timestep
    assert engine.restitution == 0.1  # default restitution


def test_engine_creation_custom_params():
    """Test engine creation with custom parameters."""
    builder = SceneBuilder()
    builder.add_body(
        position=[0, 0, 0],
        mass=2.0,
        shape_type=ShapeType.BOX,
        shape_params=[1, 1, 1]
    )
    
    soa_data = builder.build()
    custom_gravity = np.array([0, -5.0, 0], dtype=np.float32)
    custom_dt = 0.01
    custom_restitution = 0.5
    
    engine = TensorPhysicsEngine(
        x=soa_data['x'],
        q=soa_data['q'],
        v=soa_data['v'],
        omega=soa_data['omega'],
        inv_mass=soa_data['inv_mass'],
        inv_inertia=soa_data['inv_inertia'],
        shape_type=soa_data['shape_type'],
        shape_params=soa_data['shape_params'],
        gravity=custom_gravity,
        dt=custom_dt,
        restitution=custom_restitution
    )
    
    assert engine.dt == custom_dt
    assert engine.restitution == custom_restitution
    assert np.allclose(engine.gravity.numpy(), custom_gravity)


def test_engine_jit_compilation():
    """Test that JIT compilation of XPBD pipeline doesn't crash."""
    builder = SceneBuilder()
    builder.add_body(position=[0, 0, 0], mass=1.0, shape_type=ShapeType.SPHERE, shape_params=[1, 0, 0])
    builder.add_body(position=[2, 0, 0], mass=1.0, shape_type=ShapeType.SPHERE, shape_params=[1, 0, 0])
    
    soa_data = builder.build()
    engine = TensorPhysicsEngine(
        x=soa_data['x'], q=soa_data['q'], v=soa_data['v'], omega=soa_data['omega'],
        inv_mass=soa_data['inv_mass'], inv_inertia=soa_data['inv_inertia'],
        shape_type=soa_data['shape_type'], shape_params=soa_data['shape_params']
    )
    
    # The engine should have JIT-compiled functions
    assert engine.jitted_step is not None
    assert engine.jitted_n_step is not None


def test_stack_scene_creation(multi_body_stack_scene):
    """Test that complex scenes can be created with XPBD engine."""
    engine = multi_body_stack_scene
    state = engine.get_state()
    
    # Should have 6 bodies (ground + 5 boxes)
    assert state['x'].shape[0] == 6
    assert state['x'].shape[1] == 3  # 3D positions
    
    # Verify engine is initialized
    assert engine.restitution == 0.1
    assert np.allclose(engine.gravity.numpy(), [0, -9.81, 0])


def test_random_scene_creation(random_bodies_scene):
    """Test that random scenes work with XPBD engine."""
    engine = random_bodies_scene
    state = engine.get_state()
    
    # Should have 20 random bodies
    assert state['x'].shape[0] == 20
    assert state['x'].shape[1] == 3  # 3D positions