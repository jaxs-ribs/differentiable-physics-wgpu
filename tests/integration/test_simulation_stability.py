"""Integration test for simulation stability.

Verifies that the simulation remains stable and doesn't "explode" with
unrealistic velocities or positions, which is a common failure mode in
physics simulations due to numerical instabilities.
"""
import os
import numpy as np
import pytest
from tinygrad import Tensor
from physics.types import BodySchema

class TestSimulationStability:
  """Test that simulations remain stable over time."""
  
  def calculate_total_kinetic_energy(self, bodies: Tensor) -> float:
    """Calculate total kinetic energy to detect explosions."""
    bodies_np = bodies.numpy()
    
    # Extract velocities
    linear_vels = bodies_np[:, BodySchema.VEL_X:BodySchema.VEL_Z+1]
    angular_vels = bodies_np[:, BodySchema.ANG_VEL_X:BodySchema.ANG_VEL_Z+1]
    
    # Extract masses
    inv_masses = bodies_np[:, BodySchema.INV_MASS]
    masses = np.where(inv_masses > 0, 1.0 / inv_masses, 0.0)
    
    # Linear kinetic energy
    linear_ke = 0.5 * masses * np.sum(linear_vels**2, axis=1)
    
    # Simplified angular KE (just use magnitude of angular velocity)
    angular_ke = 0.5 * masses * np.sum(angular_vels**2, axis=1)
    
    return float(np.sum(linear_ke) + np.sum(angular_ke))
  
  def check_positions_reasonable(self, bodies: Tensor, max_distance: float = 500.0) -> bool:
    """Check that no body has flown off to infinity."""
    bodies_np = bodies.numpy()
    positions = bodies_np[:, BodySchema.POS_X:BodySchema.POS_Z+1]
    distances = np.linalg.norm(positions, axis=1)
    return np.all(distances < max_distance)
  
  def test_box_stack_is_stable(self, multi_body_stack_scene):
    """Test that a stack of boxes settles down and doesn't explode.
    
    This tests the stability of collision resolution and constraint solving.
    """
    engine = multi_body_stack_scene
    
    # Track maximum kinetic energy over time
    max_ke = 0.0
    ke_history = []
    
    # Run simulation for many steps
    num_steps = 200 if os.environ.get('CI') == 'true' else 500
    for step in range(num_steps):
      engine.step(0.016)  # ~60Hz
      
      ke = self.calculate_total_kinetic_energy(engine.bodies)
      ke_history.append(ke)
      max_ke = max(max_ke, ke)
      
      # Early exit if explosion detected
      if ke > 10000.0:  # Allow higher energy during settling
        pytest.fail(f"Kinetic energy exploded to {ke:.2f} at step {step}")
      
      # Check positions are reasonable
      if not self.check_positions_reasonable(engine.bodies):
        pytest.fail(f"Bodies flew off to unreasonable positions at step {step}")
    
    # After many steps, the stack should have settled
    final_ke = ke_history[-1]
    avg_final_ke = np.mean(ke_history[-50:])  # Average of last 50 steps
    
    print(f"Max KE during simulation: {max_ke:.6f}")
    print(f"Final KE: {final_ke:.6f}")
    print(f"Average final KE: {avg_final_ke:.6f}")
    
    # Stack should settle to near-zero kinetic energy
    assert avg_final_ke < 0.1, f"Stack didn't settle, final KE: {avg_final_ke:.6f}"
    
    # Maximum KE shouldn't be too high (indicates temporary instability)
    assert max_ke < 50.0, f"Maximum KE too high: {max_ke:.6f}"
  
  def test_high_velocity_collision_stable(self):
    """Test that high-velocity collisions don't cause instability."""
    from physics.engine import TensorPhysicsEngine
    from physics.types import create_body_array, ShapeType
    
    bodies_list = []
    
    # Fast-moving sphere
    sphere1 = create_body_array(
      position=np.array([-5, 0, 0], dtype=np.float32),
      velocity=np.array([20, 0, 0], dtype=np.float32),  # High velocity
      orientation=np.array([1, 0, 0, 0], dtype=np.float32),
      angular_vel=np.zeros(3, dtype=np.float32),
      mass=1.0,
      inertia=np.eye(3, dtype=np.float32) * 0.4,
      shape_type=ShapeType.SPHERE,
      shape_params=np.array([1.0, 0, 0], dtype=np.float32)
    )
    bodies_list.append(sphere1)
    
    # Stationary sphere
    sphere2 = create_body_array(
      position=np.array([0, 0, 0], dtype=np.float32),
      velocity=np.zeros(3, dtype=np.float32),
      orientation=np.array([1, 0, 0, 0], dtype=np.float32),
      angular_vel=np.zeros(3, dtype=np.float32),
      mass=1.0,
      inertia=np.eye(3, dtype=np.float32) * 0.4,
      shape_type=ShapeType.SPHERE,
      shape_params=np.array([1.0, 0, 0], dtype=np.float32)
    )
    bodies_list.append(sphere2)
    
    bodies = np.stack(bodies_list)
    engine = TensorPhysicsEngine(bodies, gravity=np.zeros(3, dtype=np.float32))
    
    # Run collision
    initial_ke = self.calculate_total_kinetic_energy(engine.bodies)
    
    for step in range(100):
      engine.step(0.01)
      
      ke = self.calculate_total_kinetic_energy(engine.bodies)
      
      # Energy shouldn't increase
      assert ke <= initial_ke * 1.1, f"Energy increased too much: {ke:.2f} vs {initial_ke:.2f}"
      
      # Positions should remain reasonable
      assert self.check_positions_reasonable(engine.bodies, max_distance=50.0)
  
  def test_random_scene_stability(self, random_bodies_scene):
    """Test that a chaotic scene with many bodies remains stable."""
    engine = random_bodies_scene
    
    initial_positions = engine.bodies.numpy()[:, BodySchema.POS_X:BodySchema.POS_Z+1].copy()
    
    # Run for many steps
    explosion_detected = False
    for step in range(200):
      engine.step(0.01)
      
      # Check for explosion indicators
      bodies_np = engine.bodies.numpy()
      velocities = bodies_np[:, BodySchema.VEL_X:BodySchema.VEL_Z+1]
      max_velocity = np.max(np.linalg.norm(velocities, axis=1))
      
      if max_velocity > 100.0:
        explosion_detected = True
        pytest.fail(f"Velocity explosion detected: max velocity {max_velocity:.2f} at step {step}")
      
      if not self.check_positions_reasonable(engine.bodies, max_distance=200.0):
        explosion_detected = True
        pytest.fail(f"Position explosion detected at step {step}")
    
    assert not explosion_detected, "Simulation remained stable"
    
    # Check that bodies haven't all collapsed to origin (another failure mode)
    final_positions = engine.bodies.numpy()[:, BodySchema.POS_X:BodySchema.POS_Z+1]
    avg_distance_from_origin = np.mean(np.linalg.norm(final_positions, axis=1))
    assert avg_distance_from_origin > 1.0, "Bodies collapsed to origin"
  
  def test_zero_timestep_stability(self, two_body_scene):
    """Test that zero or very small timesteps don't cause issues."""
    engine = two_body_scene
    
    # Test zero timestep
    initial_state = engine.bodies.numpy().copy()
    engine.step(0.0)
    final_state = engine.bodies.numpy()
    
    # State shouldn't change with zero timestep
    np.testing.assert_allclose(initial_state, final_state, atol=1e-6)
    
    # Test very small timestep
    for _ in range(10):
      engine.step(1e-6)
    
    # Should still be stable
    assert self.check_positions_reasonable(engine.bodies)