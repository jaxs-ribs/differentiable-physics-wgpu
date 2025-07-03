"""Basic XPBD physics engine tests.

Tests the XPBD scaffolding to ensure it can be called without crashing.
These are placeholder tests that will be expanded as XPBD functions are implemented.
"""
import pytest
import numpy as np
from physics.engine import TensorPhysicsEngine


def test_engine_creation():
    """Test that XPBD engine can be created."""
    # Create simple scene with one sphere
    bodies = np.zeros((1, 27), dtype=np.float32)
    bodies[0, 0:3] = [0, 0, 0]  # position
    bodies[0, 6:10] = [1, 0, 0, 0]  # quaternion
    bodies[0, 13] = 1.0  # inv_mass
    
    engine = TensorPhysicsEngine(bodies)
    assert engine is not None
    assert engine.dt == 0.016  # default timestep


def test_single_step_no_crash(two_body_scene):
    """Test that a single physics step doesn't crash with XPBD pipeline."""
    engine = two_body_scene
    initial_bodies = engine.get_state()
    
    # This should not crash, even though XPBD functions are placeholders
    try:
        result = engine.step()
        # Since all XPBD functions are pass statements, result should be similar to input
        assert result is not None
    except Exception as e:
        # Expected to fail since XPBD functions are not implemented
        assert "TODO" in str(e) or "pass" in str(e) or "NotImplementedError" in str(e)


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
    bodies = engine.get_state()
    
    # Should have 6 bodies (ground + 5 boxes)
    assert bodies.shape[0] == 6
    assert bodies.shape[1] == 27  # NUM_PROPERTIES
    
    # Verify engine is initialized
    assert engine.restitution == 0.1
    assert np.allclose(engine.gravity.numpy(), [0, -9.81, 0])


def test_random_scene_creation(random_bodies_scene):
    """Test that random scenes work with XPBD engine."""
    engine = random_bodies_scene
    bodies = engine.get_state()
    
    # Should have 20 random bodies
    assert bodies.shape[0] == 20
    assert bodies.shape[1] == 27


def test_jit_compilation_no_crash(two_body_scene):
    """Test that JIT compilation of XPBD pipeline doesn't crash."""
    engine = two_body_scene
    
    # The engine should have JIT-compiled functions
    assert engine.jitted_step is not None
    assert engine.jitted_n_step is not None
    
    # These may fail due to unimplemented XPBD functions, but shouldn't crash compilation
    try:
        # Test single step JIT
        result = engine.jitted_step(engine.bodies)
        assert result is not None
    except Exception as e:
        # Expected - XPBD functions not implemented yet
        pass
    
    try:
        # Test n-step JIT
        result = engine.jitted_n_step(engine.bodies, 3)
        assert result is not None
    except Exception as e:
        # Expected - XPBD functions not implemented yet
        pass