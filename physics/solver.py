"""Impulse-based collision resolution solver.

This module resolves collisions by applying impulses to separate colliding bodies
and simulate realistic bouncing/sliding behavior. We use a sequential impulse
solver that processes each contact individually.

Physics background:
- Impulse (J): Change in momentum, J = m * Δv
- Conservation of momentum: Total momentum before = after collision
- Restitution (e): "bounciness", 0 = perfectly inelastic, 1 = perfectly elastic
- Contact constraint: Bodies shouldn't penetrate, relative velocity along normal ≥ 0

The solver calculates impulses that:
1. Prevent penetration (bodies moving apart along contact normal)
2. Apply restitution for bouncing
3. Conserve momentum
4. Account for rotational effects through the inertia tensor

Additionally, we apply position correction to fix penetration that has already
occurred due to discrete time stepping. This is a non-physical "nudge" to
separate overlapping bodies.
"""
import numpy as np
from tinygrad import Tensor
from .types import BodySchema, Contact
from .math_utils import get_world_inv_inertia

def calculate_impulse_magnitude(inv_mass_i: float, inv_mass_j: float,
                               inv_I_i: np.ndarray, inv_I_j: np.ndarray,
                               r_i: np.ndarray, r_j: np.ndarray,
                               normal: np.ndarray, v_rel_normal: float,
                               restitution: float) -> float:
  """Calculate impulse magnitude using the impulse-momentum equation.
  
  The impulse magnitude j is derived from:
  - Relative velocity constraint: v_rel_new · n = -e * v_rel_old · n
  - Impulse-momentum relation: Δv = j/m for linear, Δω = I^(-1) * (r × j*n) for angular
  
  Args:
    inv_mass_i, inv_mass_j: Inverse masses (0 for static bodies)
    inv_I_i, inv_I_j: Inverse inertia tensors in world space (3x3)
    r_i, r_j: Contact point relative to body centers
    normal: Contact normal (unit vector)
    v_rel_normal: Relative velocity along normal (negative = approaching)
    restitution: Coefficient of restitution (0-1)
    
  Returns:
    Impulse magnitude (scalar, applied along normal)
  """
  numerator = -(1 + restitution) * v_rel_normal
  # Effective mass calculation accounts for both linear and rotational inertia
  linear_term = inv_mass_i + inv_mass_j
  angular_term_i = np.dot(normal, np.cross(inv_I_i @ np.cross(r_i, normal), r_i))
  angular_term_j = np.dot(normal, np.cross(inv_I_j @ np.cross(r_j, normal), r_j))
  denominator = linear_term + angular_term_i + angular_term_j
  return numerator / denominator

def resolve_collisions(bodies: Tensor, contacts: list[Contact], restitution: float = 0.1) -> Tensor:
  """Resolve collisions by applying impulses based on contact information.
  
  For each contact:
  1. Calculate relative velocity at contact point
  2. Check if bodies are separating (no impulse needed)
  3. Calculate impulse magnitude based on physics constraints
  4. Apply linear impulse: Δv = j*n / m
  5. Apply angular impulse: Δω = I^(-1) * (r × j*n)
  6. Apply position correction to separate penetrating bodies
  
  Args:
    bodies: State tensor of shape (N, NUM_PROPERTIES)
    contacts: List of Contact objects from narrowphase
    restitution: Coefficient of restitution (0 = no bounce, 1 = perfect bounce)
    
  Returns:
    Updated state tensor with resolved collisions
  """
  if not contacts: return bodies
  
  bodies_np = bodies.numpy()
  
  # Extract all relevant arrays for efficiency
  positions = bodies_np[:, BodySchema.POS_X:BodySchema.POS_Z+1]
  velocities = bodies_np[:, BodySchema.VEL_X:BodySchema.VEL_Z+1]
  ang_velocities = bodies_np[:, BodySchema.ANG_VEL_X:BodySchema.ANG_VEL_Z+1]
  inv_masses = bodies_np[:, BodySchema.INV_MASS]
  inv_inertias_local = bodies_np[:, BodySchema.INV_INERTIA_XX:BodySchema.INV_INERTIA_ZZ+1]
  quats = bodies_np[:, BodySchema.QUAT_W:BodySchema.QUAT_Z+1]
  
  # Transform inverse inertias to world space
  inv_inertias_world = get_world_inv_inertia(Tensor(quats), Tensor(inv_inertias_local)).numpy()
  
  for contact in contacts:
    idx_i, idx_j = contact.pair_indices
    normal, point, depth = contact.normal, contact.point, contact.depth
    
    # Skip if both bodies are static
    inv_mass_i, inv_mass_j = inv_masses[idx_i], inv_masses[idx_j]
    if inv_mass_i + inv_mass_j < 1e-7: continue
    
    # Get body properties
    vel_i, vel_j = velocities[idx_i], velocities[idx_j]
    ang_vel_i, ang_vel_j = ang_velocities[idx_i], ang_velocities[idx_j]
    inv_I_i, inv_I_j = inv_inertias_world[idx_i].reshape(3, 3), inv_inertias_world[idx_j].reshape(3, 3)
    pos_i, pos_j = positions[idx_i], positions[idx_j]
    
    # Calculate relative velocity at contact point
    # v_contact = v_center + ω × r
    r_i, r_j = point - pos_i, point - pos_j
    v_rel = (vel_i + np.cross(ang_vel_i, r_i)) - (vel_j + np.cross(ang_vel_j, r_j))
    v_rel_normal = np.dot(v_rel, normal)
    
    # Skip if bodies are separating
    if v_rel_normal > 0: continue
    
    # Calculate and apply impulse
    j = calculate_impulse_magnitude(inv_mass_i, inv_mass_j, inv_I_i, inv_I_j, 
                                   r_i, r_j, normal, v_rel_normal, restitution)
    impulse_vec = j * normal
    
    # Update velocities (Δv = j/m, Δω = I^(-1) * (r × j))
    velocities[idx_i] += impulse_vec * inv_mass_i
    velocities[idx_j] -= impulse_vec * inv_mass_j
    ang_velocities[idx_i] += inv_I_i @ np.cross(r_i, impulse_vec)
    ang_velocities[idx_j] -= inv_I_j @ np.cross(r_j, impulse_vec)
    
    # Position correction (Baumgarte stabilization)
    # Push bodies apart by a fraction of penetration depth
    correction_percent = 0.2  # How much penetration to fix per frame
    slop = 0.01  # Allowed penetration to prevent jitter
    correction_amount = max(depth - slop, 0.0) / (inv_mass_i + inv_mass_j) * correction_percent
    correction_vec = correction_amount * normal
    positions[idx_i] += correction_vec * inv_mass_i
    positions[idx_j] -= correction_vec * inv_mass_j
  
  # Update state tensor
  bodies_np[:, BodySchema.POS_X:BodySchema.POS_Z+1] = positions
  bodies_np[:, BodySchema.VEL_X:BodySchema.VEL_Z+1] = velocities
  bodies_np[:, BodySchema.ANG_VEL_X:BodySchema.ANG_VEL_Z+1] = ang_velocities
  return Tensor(bodies_np)