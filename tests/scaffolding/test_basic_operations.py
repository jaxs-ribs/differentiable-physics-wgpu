"""Tests for basic XPBD engine operations."""
import pytest
import numpy as np
from physics.engine import TensorPhysicsEngine


def test_single_step_no_crash(two_body_scene):
    """Test that a single physics step doesn't crash."""
    engine = two_body_scene
    initial_state = engine.get_state()
    
    # This should not crash
    engine.step()
    
    # State should have changed (bodies should fall due to gravity)
    final_state = engine.get_state()
    assert not np.allclose(initial_state['x'], final_state['x'])
    assert not np.allclose(initial_state['v'], final_state['v'])


def test_multi_step_simulation(two_body_scene):
    """Test that multi-step simulation works correctly."""
    engine = two_body_scene
    initial_y = engine.get_state()['x'][:, 1].copy()
    
    # Run simulation for multiple steps
    engine.run_simulation(10)
    
    # Bodies should have fallen
    final_y = engine.get_state()['x'][:, 1]
    assert np.all(final_y < initial_y)


def test_step_with_custom_dt(two_body_scene):
    """Test stepping with custom timestep."""
    engine = two_body_scene
    original_dt = engine.dt
    
    # Step with custom dt
    custom_dt = 0.001
    engine.step(dt=custom_dt)
    
    # Engine should have updated dt
    assert engine.dt == custom_dt
    
    # Step without specifying dt
    engine.step()
    assert engine.dt == custom_dt  # Should retain the new dt


def test_get_state(two_body_scene):
    """Test getting engine state."""
    engine = two_body_scene
    state = engine.get_state()
    
    # Check all required fields are present
    required_fields = ['x', 'q', 'v', 'omega', 'inv_mass', 'inv_inertia', 'shape_type', 'shape_params']
    for field in required_fields:
        assert field in state
        assert isinstance(state[field], np.ndarray)
    
    # Check dimensions
    n_bodies = state['x'].shape[0]
    assert state['x'].shape == (n_bodies, 3)
    assert state['q'].shape == (n_bodies, 4)
    assert state['v'].shape == (n_bodies, 3)
    assert state['omega'].shape == (n_bodies, 3)
    assert state['inv_mass'].shape == (n_bodies,)
    assert state['inv_inertia'].shape == (n_bodies, 3, 3)
    assert state['shape_type'].shape == (n_bodies,)
    assert state['shape_params'].shape == (n_bodies, 3)


def test_set_state(two_body_scene):
    """Test setting engine state."""
    engine = two_body_scene
    original_state = engine.get_state()
    
    # Modify positions and velocities
    new_x = original_state['x'] + np.array([1, 0, 0])
    new_v = original_state['v'] + np.array([0, 1, 0])
    new_q = original_state['q'].copy()
    new_omega = original_state['omega'] + np.array([0, 0, 1])
    
    engine.set_state(new_x, new_q, new_v, new_omega)
    
    # Check state was updated
    updated_state = engine.get_state()
    assert np.allclose(updated_state['x'], new_x)
    assert np.allclose(updated_state['v'], new_v)
    assert np.allclose(updated_state['q'], new_q)
    assert np.allclose(updated_state['omega'], new_omega)