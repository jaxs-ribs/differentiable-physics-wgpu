"""Impulse-based collision resolution solver.

This module resolves collisions by applying impulses to separate colliding bodies
and simulate realistic bouncing/sliding behavior. This implementation is fully
vectorized and JIT-compatible, processing all contacts simultaneously.
"""
from tinygrad import Tensor, dtypes
from .types import BodySchema
from .math_utils import get_world_inv_inertia

def resolve_collisions(bodies: Tensor, pair_indices: Tensor, contact_normals: Tensor, 
                      contact_depths: Tensor, contact_points: Tensor, contact_mask: Tensor,
                      restitution: float = 0.1) -> Tensor:
  """Resolve collisions by applying impulses based on contact information.
  
  This function is fully vectorized and operates on all contacts simultaneously.
  It uses scatter operations to correctly handle multiple collisions per body.
  
  Args:
    bodies: State tensor of shape (N, NUM_PROPERTIES)
    pair_indices: Tensor of shape (M, 2) with body indices for each contact
    contact_normals: Tensor of shape (M, 3) with contact normals
    contact_depths: Tensor of shape (M,) with penetration depths
    contact_points: Tensor of shape (M, 3) with contact points
    contact_mask: Tensor of shape (M,) boolean mask for valid contacts
    restitution: Coefficient of restitution (0 = no bounce, 1 = perfect bounce)
    
  Returns:
    Updated state tensor with resolved collisions
  """
  n_bodies = bodies.shape[0]
  n_contacts = pair_indices.shape[0]
  
  # For JIT compatibility with empty contacts, we need to ensure indices are valid
  # When n_contacts is 0, create dummy indices that won't affect the result
  if n_contacts == 0:
    # Create dummy data that will result in zero changes
    pair_indices = Tensor.zeros((1, 2), dtype=dtypes.int32)
    contact_normals = Tensor.zeros((1, 3))
    contact_depths = Tensor.zeros((1,))
    contact_points = Tensor.zeros((1, 3))
    contact_mask = Tensor.zeros((1,))
    n_contacts = 1
  
  # Gather body data for all contacts using pure tensor operations
  indices_a = pair_indices[:, 0]
  indices_b = pair_indices[:, 1]
  
  # Gather full body data for each pair
  # We need to expand indices to match the shape of bodies
  indices_a_expanded = indices_a.unsqueeze(1).expand(-1, bodies.shape[1])
  indices_b_expanded = indices_b.unsqueeze(1).expand(-1, bodies.shape[1])
  bodies_a = bodies.gather(0, indices_a_expanded)
  bodies_b = bodies.gather(0, indices_b_expanded)
  
  # Extract data for body A (first in each pair)
  pos_a = bodies_a[:, BodySchema.POS_X:BodySchema.POS_Z+1]
  vel_a = bodies_a[:, BodySchema.VEL_X:BodySchema.VEL_Z+1]
  ang_vel_a = bodies_a[:, BodySchema.ANG_VEL_X:BodySchema.ANG_VEL_Z+1]
  inv_mass_a = bodies_a[:, BodySchema.INV_MASS]
  quat_a = bodies_a[:, BodySchema.QUAT_W:BodySchema.QUAT_Z+1]
  inv_inertia_local_a = bodies_a[:, BodySchema.INV_INERTIA_XX:BodySchema.INV_INERTIA_ZZ+1]
  
  # Extract data for body B (second in each pair)
  pos_b = bodies_b[:, BodySchema.POS_X:BodySchema.POS_Z+1]
  vel_b = bodies_b[:, BodySchema.VEL_X:BodySchema.VEL_Z+1]
  ang_vel_b = bodies_b[:, BodySchema.ANG_VEL_X:BodySchema.ANG_VEL_Z+1]
  inv_mass_b = bodies_b[:, BodySchema.INV_MASS]
  quat_b = bodies_b[:, BodySchema.QUAT_W:BodySchema.QUAT_Z+1]
  inv_inertia_local_b = bodies_b[:, BodySchema.INV_INERTIA_XX:BodySchema.INV_INERTIA_ZZ+1]
  
  # Transform inverse inertias to world space
  inv_I_world_a = get_world_inv_inertia(quat_a, inv_inertia_local_a).reshape(-1, 3, 3)
  inv_I_world_b = get_world_inv_inertia(quat_b, inv_inertia_local_b).reshape(-1, 3, 3)
  
  # Calculate relative positions from centers to contact points
  r_a = contact_points - pos_a  # (M, 3)
  r_b = contact_points - pos_b  # (M, 3)
  
  # Calculate relative velocity at contact points
  # v_contact = v_center + ω × r
  v_contact_a = vel_a + cross_product(ang_vel_a, r_a)
  v_contact_b = vel_b + cross_product(ang_vel_b, r_b)
  v_rel = v_contact_a - v_contact_b  # (M, 3)
  
  # Project relative velocity onto normal
  v_rel_normal = (v_rel * contact_normals).sum(axis=1)  # (M,)
  
  # Skip contacts that are separating
  # If normal points from B to A, and v_rel = v_a - v_b, then:
  # v_rel_normal < 0 means bodies are approaching (need impulse)
  # v_rel_normal > 0 means bodies are separating (skip)
  active_mask = contact_mask & (v_rel_normal < 0)
  active_mask_3d = active_mask.unsqueeze(1).expand(-1, 3)
  
  # Calculate impulse magnitudes for all contacts
  # j = -(1 + e) * v_rel·n / (1/m_a + 1/m_b + n·((I_a^-1 * (r_a × n)) × r_a) + n·((I_b^-1 * (r_b × n)) × r_b))
  
  # Cross products for angular terms
  r_cross_n_a = cross_product(r_a, contact_normals.expand(n_contacts, -1))  # (M, 3)
  r_cross_n_b = cross_product(r_b, contact_normals.expand(n_contacts, -1))  # (M, 3)
  
  # Angular velocity changes: I^-1 * (r × n)
  ang_delta_a = (inv_I_world_a @ r_cross_n_a.unsqueeze(-1)).squeeze(-1)  # (M, 3)
  ang_delta_b = (inv_I_world_b @ r_cross_n_b.unsqueeze(-1)).squeeze(-1)  # (M, 3)
  
  # Angular terms in denominator: n · (ang_delta × r)
  angular_term_a = (contact_normals * cross_product(ang_delta_a, r_a)).sum(axis=1)
  angular_term_b = (contact_normals * cross_product(ang_delta_b, r_b)).sum(axis=1)
  
  # Calculate impulse magnitudes
  # Standard impulse formula: j = (1 + e) * |v_rel·n| / (denominators)
  # We want the magnitude to be positive
  numerator = (1.0 + restitution) * (-v_rel_normal)  # -v_rel_normal because v_rel_normal < 0 for approaching
  denominator = inv_mass_a + inv_mass_b + angular_term_a + angular_term_b
  denominator = denominator.maximum(1e-6)  # Prevent division by zero
  j_magnitude = numerator / denominator
  
  # DEBUG: Print first active contact
  if False and active_mask.sum() > 0:
    idx = active_mask.numpy().nonzero()[0][0]
    print(f"\nDEBUG SOLVER: Contact {idx}")
    print(f"  v_rel_normal[{idx}] = {v_rel_normal[idx].numpy()}")
    print(f"  numerator[{idx}] = {numerator[idx].numpy()}")
    print(f"  denominator[{idx}] = {denominator[idx].numpy()}")
    print(f"  j_magnitude[{idx}] = {j_magnitude[idx].numpy()}")
  
  # Apply active mask
  j_magnitude = j_magnitude * active_mask.float()
  
  # Calculate impulse vectors
  # The impulse on A is in the direction of the normal (from B to A)
  # The impulse on B is in the opposite direction
  impulse_vectors_a = j_magnitude.unsqueeze(1) * contact_normals  # (M, 3)
  impulse_vectors_b = -impulse_vectors_a  # Equal and opposite
  
  # Calculate velocity changes
  delta_vel_a = impulse_vectors_a * inv_mass_a.unsqueeze(1)  # (M, 3)
  delta_vel_b = impulse_vectors_b * inv_mass_b.unsqueeze(1)  # (M, 3)
  
  # Calculate angular velocity changes
  delta_ang_vel_a = (inv_I_world_a @ cross_product(r_a, impulse_vectors_a).unsqueeze(-1)).squeeze(-1)
  delta_ang_vel_b = (inv_I_world_b @ cross_product(r_b, impulse_vectors_b).unsqueeze(-1)).squeeze(-1)
  
  # Position correction (Baumgarte stabilization)
  correction_percent = 0.2
  slop = 0.01
  correction_amount = ((contact_depths - slop).maximum(0.0) / (inv_mass_a + inv_mass_b).maximum(1e-6) * correction_percent)
  correction_vec = correction_amount.unsqueeze(1) * contact_normals * active_mask_3d.float()
  
  delta_pos_a = correction_vec * inv_mass_a.unsqueeze(1)
  delta_pos_b = -correction_vec * inv_mass_b.unsqueeze(1)
  
  # Now we need to scatter the changes back to the bodies tensor
  # We'll use scatter operations to accumulate the changes
  
  # Create full delta tensors
  delta_bodies = Tensor.zeros_like(bodies)
  
  # Prepare indices for scattering
  # We need to flatten the multi-dimensional updates
  n_bodies = bodies.shape[0]
  
  # For positions (3 components per body)
  pos_indices_a = indices_a.unsqueeze(1).expand(-1, 3) * BodySchema.NUM_PROPERTIES + \
                   Tensor.arange(BodySchema.POS_X, BodySchema.POS_Z+1).unsqueeze(0)
  pos_indices_b = indices_b.unsqueeze(1).expand(-1, 3) * BodySchema.NUM_PROPERTIES + \
                   Tensor.arange(BodySchema.POS_X, BodySchema.POS_Z+1).unsqueeze(0)
  
  # For velocities (3 components per body)
  vel_indices_a = indices_a.unsqueeze(1).expand(-1, 3) * BodySchema.NUM_PROPERTIES + \
                   Tensor.arange(BodySchema.VEL_X, BodySchema.VEL_Z+1).unsqueeze(0)
  vel_indices_b = indices_b.unsqueeze(1).expand(-1, 3) * BodySchema.NUM_PROPERTIES + \
                   Tensor.arange(BodySchema.VEL_X, BodySchema.VEL_Z+1).unsqueeze(0)
  
  # For angular velocities (3 components per body)
  ang_vel_indices_a = indices_a.unsqueeze(1).expand(-1, 3) * BodySchema.NUM_PROPERTIES + \
                       Tensor.arange(BodySchema.ANG_VEL_X, BodySchema.ANG_VEL_Z+1).unsqueeze(0)
  ang_vel_indices_b = indices_b.unsqueeze(1).expand(-1, 3) * BodySchema.NUM_PROPERTIES + \
                       Tensor.arange(BodySchema.ANG_VEL_X, BodySchema.ANG_VEL_Z+1).unsqueeze(0)
  
  # Flatten the bodies and delta arrays for scatter
  bodies_flat = bodies.flatten()
  delta_flat = Tensor.zeros_like(bodies_flat)
  
  # Scatter position changes
  delta_flat = delta_flat.scatter_reduce(0, pos_indices_a.flatten(), delta_pos_a.flatten(), "sum")
  delta_flat = delta_flat.scatter_reduce(0, pos_indices_b.flatten(), delta_pos_b.flatten(), "sum")
  
  # Scatter velocity changes
  delta_flat = delta_flat.scatter_reduce(0, vel_indices_a.flatten(), delta_vel_a.flatten(), "sum")
  delta_flat = delta_flat.scatter_reduce(0, vel_indices_b.flatten(), delta_vel_b.flatten(), "sum")
  
  # Scatter angular velocity changes
  delta_flat = delta_flat.scatter_reduce(0, ang_vel_indices_a.flatten(), delta_ang_vel_a.flatten(), "sum")
  delta_flat = delta_flat.scatter_reduce(0, ang_vel_indices_b.flatten(), delta_ang_vel_b.flatten(), "sum")
  
  # Reshape back and apply the accumulated changes
  delta_bodies = delta_flat.reshape(bodies.shape)
  # Create a new tensor to avoid in-place modification issues
  # This clone() is critical - without it, collision resolution applies twice!
  return bodies.clone() + delta_bodies

def cross_product(a: Tensor, b: Tensor) -> Tensor:
  """Compute cross product of two 3D vector tensors.
  
  Args:
    a, b: Tensors of shape (..., 3)
    
  Returns:
    Cross product tensor of shape (..., 3)
  """
  # a × b = [a_y*b_z - a_z*b_y, a_z*b_x - a_x*b_z, a_x*b_y - a_y*b_x]
  ax, ay, az = a[..., 0:1], a[..., 1:2], a[..., 2:3]
  bx, by, bz = b[..., 0:1], b[..., 1:2], b[..., 2:3]
  
  cx = ay * bz - az * by
  cy = az * bx - ax * bz
  cz = ax * by - ay * bx
  
  return Tensor.cat(cx, cy, cz, dim=-1)