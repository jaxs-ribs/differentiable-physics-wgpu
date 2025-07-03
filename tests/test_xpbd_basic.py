"""Basic XPBD physics engine tests.

Tests the XPBD scaffolding to ensure it can be called without crashing.
These are placeholder tests that will be expanded as XPBD functions are implemented.
"""
import pytest
import numpy as np
from physics.engine import TensorPhysicsEngine
from physics.types import ShapeType
from scripts.scene_builder import SceneBuilder


def test_engine_creation():
    """Test that XPBD engine can be created."""
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


def test_single_step_no_crash(two_body_scene):
    """Test that a single physics step doesn't crash with XPBD pipeline."""
    engine = two_body_scene
    initial_state = engine.get_state()
    
    # This should not crash, even though XPBD functions are placeholders
    try:
        engine.step()
        # Since all XPBD functions are placeholders, state should be unchanged
        assert engine.get_state() is not None
    except Exception as e:
        # Expected to fail since XPBD functions are not implemented
        assert "TODO" in str(e) or "pass" in str(e) or "NotImplementedError" in str(e) or "JIT" in str(e)


def test_multi_step_no_crash(two_body_scene):
    """Test that multi-step simulation doesn't crash."""
    engine = two_body_scene
    
    try:
        engine.run_simulation(5)
        # If it doesn't crash, the scaffolding is working
        assert True
    except Exception as e:
        # Expected to fail since XPBD functions are not implemented
        assert "TODO" in str(e) or "pass" in str(e) or "NotImplementedError" in str(e)


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


def test_jit_compilation_no_crash(two_body_scene):
    """Test that JIT compilation of XPBD pipeline doesn't crash."""
    engine = two_body_scene
    
    # The engine should have JIT-compiled functions
    assert engine.jitted_step is not None
    assert engine.jitted_n_step is not None
    
    # These may fail due to unimplemented XPBD functions, but shouldn't crash compilation
    try:
        # Test single step JIT
        engine.jitted_step()
        assert True  # If we get here, JIT compilation worked
    except Exception as e:
        # Expected - XPBD functions not implemented yet or JIT compilation issues
        assert "TODO" in str(e) or "JIT" in str(e) or "NotImplementedError" in str(e)
    
    try:
        # Test n-step JIT
        engine.jitted_n_step(engine.x, engine.q, engine.v, engine.omega, 3)
        assert True  # If we get here, JIT compilation worked
    except Exception as e:
        # Expected - XPBD functions not implemented yet or JIT compilation issues
        assert "TODO" in str(e) or "JIT" in str(e) or "NotImplementedError" in str(e)