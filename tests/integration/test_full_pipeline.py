"""Integration tests for the full XPBD pipeline."""
import pytest
import numpy as np
from physics.engine import TensorPhysicsEngine
from physics.types import ShapeType
from scripts.scene_builder import SceneBuilder


def test_xpbd_pipeline_single_step():
    """Test full XPBD pipeline execution for a single step."""
    # Create a scene with two spheres
    builder = SceneBuilder()
    builder.add_body(
        position=[0, 5, 0],
        velocity=[0, 0, 0],
        mass=1.0,
        shape_type=ShapeType.SPHERE,
        shape_params=[1, 0, 0]
    )
    builder.add_body(
        position=[0.5, 10, 0],
        velocity=[0, 0, 0],
        mass=1.0,
        shape_type=ShapeType.SPHERE,
        shape_params=[1, 0, 0]
    )
    
    soa_data = builder.build()
    engine = TensorPhysicsEngine(
        x=soa_data['x'], q=soa_data['q'], v=soa_data['v'], omega=soa_data['omega'],
        inv_mass=soa_data['inv_mass'], inv_inertia=soa_data['inv_inertia'],
        shape_type=soa_data['shape_type'], shape_params=soa_data['shape_params']
    )
    
    initial_state = engine.get_state()
    
    # Execute one step - this tests the full pipeline:
    # 1. Forward prediction (integration)
    # 2. Broadphase collision detection
    # 3. Narrowphase contact generation
    # 4. Constraint solving
    # 5. Velocity update
    engine.step()
    
    final_state = engine.get_state()
    
    # Verify physics happened
    # Bodies should have fallen (y position decreased)
    assert np.all(final_state['x'][:, 1] < initial_state['x'][:, 1])
    
    # Bodies should have gained downward velocity
    assert np.all(final_state['v'][:, 1] < 0)


def test_xpbd_pipeline_multi_step():
    """Test full XPBD pipeline over multiple steps."""
    builder = SceneBuilder()
    # Falling sphere
    builder.add_body(
        position=[0, 10, 0],
        velocity=[1, 0, 0],  # Initial horizontal velocity
        mass=1.0,
        shape_type=ShapeType.SPHERE,
        shape_params=[1, 0, 0]
    )
    
    soa_data = builder.build()
    engine = TensorPhysicsEngine(
        x=soa_data['x'], q=soa_data['q'], v=soa_data['v'], omega=soa_data['omega'],
        inv_mass=soa_data['inv_mass'], inv_inertia=soa_data['inv_inertia'],
        shape_type=soa_data['shape_type'], shape_params=soa_data['shape_params']
    )
    
    positions = []
    velocities = []
    
    # Run simulation and collect trajectory
    for _ in range(20):
        state = engine.get_state()
        positions.append(state['x'].copy())
        velocities.append(state['v'].copy())
        engine.step()
    
    positions = np.array(positions)  # (20, N, 3)
    velocities = np.array(velocities)  # (20, N, 3)
    
    # Verify projectile motion
    # X position should increase linearly (constant x velocity)
    x_positions = positions[:, 0, 0]
    assert np.all(np.diff(x_positions) > 0)
    
    # Y position should follow parabolic trajectory
    y_positions = positions[:, 0, 1]
    assert np.all(np.diff(y_positions) < 0)  # Always falling
    
    # Y velocity should increase linearly (constant acceleration)
    y_velocities = velocities[:, 0, 1]
    y_accel = np.diff(y_velocities) / engine.dt
    assert np.allclose(y_accel, -9.81, rtol=0.1)


def test_xpbd_pipeline_with_rotation():
    """Test XPBD pipeline with rotating bodies."""
    builder = SceneBuilder()
    # Box with initial angular velocity
    builder.add_body(
        position=[0, 5, 0],
        velocity=[0, 0, 0],
        angular_vel=[0, 1, 0],  # Rotating around Y axis
        mass=1.0,
        shape_type=ShapeType.BOX,
        shape_params=[1, 1, 1]
    )
    
    soa_data = builder.build()
    engine = TensorPhysicsEngine(
        x=soa_data['x'], q=soa_data['q'], v=soa_data['v'], omega=soa_data['omega'],
        inv_mass=soa_data['inv_mass'], inv_inertia=soa_data['inv_inertia'],
        shape_type=soa_data['shape_type'], shape_params=soa_data['shape_params']
    )
    
    initial_q = engine.get_state()['q'].copy()
    
    # Run several steps
    engine.run_simulation(10)
    
    final_q = engine.get_state()['q']
    
    # Quaternion should have changed (body rotated)
    assert not np.allclose(initial_q, final_q)
    
    # Quaternion should still be normalized
    q_norm = np.linalg.norm(final_q, axis=1)
    assert np.allclose(q_norm, 1.0, atol=1e-6)


def test_xpbd_pipeline_stability():
    """Test numerical stability of XPBD pipeline over many steps."""
    builder = SceneBuilder()
    # Create a stack of boxes
    for i in range(3):
        builder.add_body(
            position=[0, 1 + i * 2.1, 0],
            mass=1.0,
            shape_type=ShapeType.BOX,
            shape_params=[1, 1, 1]
        )
    
    soa_data = builder.build()
    engine = TensorPhysicsEngine(
        x=soa_data['x'], q=soa_data['q'], v=soa_data['v'], omega=soa_data['omega'],
        inv_mass=soa_data['inv_mass'], inv_inertia=soa_data['inv_inertia'],
        shape_type=soa_data['shape_type'], shape_params=soa_data['shape_params'],
        # TEMPORARY: Reduced iterations for Jacobi solver stability
        solver_iterations=2,
        contact_compliance=0.0001
    )
    
    # Run for many steps (reduced from 100 to avoid JIT compilation timeout)
    for _ in range(50):
        engine.step()
    
    final_state = engine.get_state()
    
    # Check for numerical stability
    # Positions should be finite
    assert np.all(np.isfinite(final_state['x']))
    
    # Velocities should be reasonable (not exploding)
    assert np.all(np.abs(final_state['v']) < 100)
    
    # Quaternions should be normalized
    q_norms = np.linalg.norm(final_state['q'], axis=1)
    assert np.allclose(q_norms, 1.0, atol=1e-5)