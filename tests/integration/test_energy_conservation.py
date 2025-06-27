"""Integration test for energy conservation in the physics simulation.

Verifies that the total energy of the system does not spontaneously increase,
which would indicate numerical errors or bugs in the physics implementation.
"""
import numpy as np
import pytest
from tinygrad import Tensor
from physics.types import BodySchema

class TestEnergyConservation:
  """Test physical invariants like energy conservation."""
  
  def calculate_kinetic_energy(self, bodies: Tensor) -> float:
    """Calculate total kinetic energy of the system.
    
    KE = sum(0.5 * m * v^2 + 0.5 * I * w^2)
    
    Args:
      bodies: State tensor of shape (N, NUM_PROPERTIES)
      
    Returns:
      Total kinetic energy as float
    """
    bodies_np = bodies.numpy()
    
    # Extract velocities and angular velocities
    linear_vels = bodies_np[:, BodySchema.VEL_X:BodySchema.VEL_Z+1]
    angular_vels = bodies_np[:, BodySchema.ANG_VEL_X:BodySchema.ANG_VEL_Z+1]
    
    # Extract masses (1/inv_mass)
    inv_masses = bodies_np[:, BodySchema.INV_MASS]
    masses = np.where(inv_masses > 0, 1.0 / inv_masses, 0.0)
    
    # Linear kinetic energy: 0.5 * m * v^2
    linear_ke = 0.5 * masses * np.sum(linear_vels**2, axis=1)
    
    # Angular kinetic energy: 0.5 * w^T * I * w
    angular_ke = np.zeros(len(bodies_np))
    for i in range(len(bodies_np)):
      # Extract inverse inertia tensor
      inv_I = bodies_np[i, BodySchema.INV_INERTIA_XX:BodySchema.INV_INERTIA_ZZ+1].reshape(3, 3)
      
      # Only calculate if body has mass (inv_I might be zero for static bodies)
      if inv_masses[i] > 0 and np.any(inv_I > 0):
        # Invert to get actual inertia tensor
        I = np.linalg.inv(inv_I)
        w = angular_vels[i]
        angular_ke[i] = 0.5 * np.dot(w, I @ w)
    
    return float(np.sum(linear_ke) + np.sum(angular_ke))
  
  def test_total_energy_does_not_increase(self, two_body_scene):
    """Test that total kinetic energy doesn't increase in zero gravity.
    
    Uses the two_body_scene fixture which has a sphere moving toward a box.
    """
    engine = two_body_scene
    
    # Set gravity to zero for pure energy conservation test
    engine.gravity = Tensor(np.array([0, 0, 0], dtype=np.float32))
    
    # Calculate initial energy
    initial_energy = self.calculate_kinetic_energy(engine.bodies)
    print(f"Initial kinetic energy: {initial_energy:.6f}")
    
    # Run simulation for many steps
    energies = [initial_energy]
    for step in range(200):
      engine.step(0.01)  # Small timestep for accuracy
      energy = self.calculate_kinetic_energy(engine.bodies)
      energies.append(energy)
    
    # Check that energy never increases beyond numerical tolerance
    max_energy = max(energies)
    energy_increase = max_energy - initial_energy
    
    print(f"Final kinetic energy: {energies[-1]:.6f}")
    print(f"Max energy: {max_energy:.6f}")
    print(f"Energy increase: {energy_increase:.6f}")
    
    # Allow for small numerical errors (1% tolerance)
    assert energy_increase <= initial_energy * 0.01, \
      f"Energy increased by {energy_increase:.6f} ({energy_increase/initial_energy*100:.2f}%)"
  
  def test_energy_conservation_multi_body(self, random_bodies_scene):
    """Test energy conservation with many interacting bodies."""
    engine = random_bodies_scene
    
    # Set gravity to zero
    engine.gravity = Tensor(np.array([0, 0, 0], dtype=np.float32))
    
    # Calculate initial energy
    initial_energy = self.calculate_kinetic_energy(engine.bodies)
    
    # Run for fewer steps due to complexity
    energies = []
    for step in range(50):
      engine.step(0.01)
      energy = self.calculate_kinetic_energy(engine.bodies)
      energies.append(energy)
    
    # Check maximum energy
    max_energy = max(energies) if energies else initial_energy
    energy_increase = max_energy - initial_energy
    
    # Allow slightly more tolerance for multi-body (2%)
    assert energy_increase <= initial_energy * 0.02, \
      f"Energy increased by {energy_increase:.6f} ({energy_increase/initial_energy*100:.2f}%)"
  
  def test_energy_dissipation_with_gravity(self, multi_body_stack_scene):
    """Test that energy is properly dissipated (not created) with gravity."""
    engine = multi_body_stack_scene
    
    # Calculate initial total energy (kinetic + potential)
    initial_ke = self.calculate_kinetic_energy(engine.bodies)
    
    # Calculate initial potential energy
    bodies_np = engine.bodies.numpy()
    positions = bodies_np[:, BodySchema.POS_Y]
    inv_masses = bodies_np[:, BodySchema.INV_MASS]
    masses = np.where(inv_masses > 0, 1.0 / inv_masses, 0.0)
    g = 9.81
    initial_pe = np.sum(masses * g * positions)
    initial_total = initial_ke + initial_pe
    
    print(f"Initial total energy: {initial_total:.6f} (KE: {initial_ke:.6f}, PE: {initial_pe:.6f})")
    
    # Run simulation
    for step in range(100):
      engine.step(0.01)
    
    # Calculate final energies
    final_ke = self.calculate_kinetic_energy(engine.bodies)
    bodies_np = engine.bodies.numpy()
    positions = bodies_np[:, BodySchema.POS_Y]
    final_pe = np.sum(masses * g * positions)
    final_total = final_ke + final_pe
    
    print(f"Final total energy: {final_total:.6f} (KE: {final_ke:.6f}, PE: {final_pe:.6f})")
    
    # Total energy should not increase (can decrease due to collisions)
    assert final_total <= initial_total * 1.01, \
      f"Total energy increased by {final_total - initial_total:.6f}"