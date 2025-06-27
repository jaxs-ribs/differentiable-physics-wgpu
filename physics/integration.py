"""Semi-implicit Euler integration for rigid body dynamics.

This module handles the time-stepping of rigid bodies, updating positions and
orientations based on velocities and angular velocities. We use semi-implicit
Euler (also known as symplectic Euler) which updates velocities first, then
uses the new velocities to update positions. This provides better stability
than explicit Euler for physics simulations.

Key concepts:
- Position (x): Updated using velocity after forces are applied
- Velocity (v): Updated by forces (currently just gravity)
- Orientation (q): Represented as quaternion, updated by angular velocity
- Angular velocity (ω): Currently unchanged (no torques implemented yet)
- Static bodies: Have inv_mass = 0 and are not affected by forces

The integration preserves the unit length of quaternions through normalization
to prevent numerical drift over time.
"""
from tinygrad import Tensor
from .types import BodySchema
from .math_utils import quat_multiply

def integrate(bodies: Tensor, dt: float, gravity: Tensor) -> Tensor:
  """Update body states using semi-implicit Euler integration.
  
  Args:
    bodies: State tensor of shape (N, NUM_PROPERTIES) containing all body data
    dt: Time step in seconds
    gravity: Gravity acceleration vector as Tensor([gx, gy, gz])
    
  Returns:
    Updated state tensor with new positions, velocities, and orientations
    
  Integration steps:
    1. Apply gravity to dynamic bodies (inv_mass > 0)
    2. Update velocities: v_new = v_old + a * dt
    3. Update positions: x_new = x_old + v_new * dt (semi-implicit)
    4. Update orientations from angular velocities using quaternion calculus
    5. Normalize quaternions to maintain unit length
  """
  pos = bodies[:, BodySchema.POS_X:BodySchema.POS_Z+1]
  vel = bodies[:, BodySchema.VEL_X:BodySchema.VEL_Z+1]
  quat = bodies[:, BodySchema.QUAT_W:BodySchema.QUAT_Z+1]
  ang_vel = bodies[:, BodySchema.ANG_VEL_X:BodySchema.ANG_VEL_Z+1]
  inv_mass = bodies[:, BodySchema.INV_MASS:BodySchema.INV_MASS+1]
  
  # Dynamic bodies have inv_mass > 0; static bodies have inv_mass = 0
  is_dynamic = (inv_mass > 1e-7).reshape(-1, 1)
  
  # Apply gravity and update velocity (v += g * dt for dynamic bodies only)
  new_vel = vel.where(is_dynamic, vel + gravity * dt)
  
  # Update position using new velocity (semi-implicit Euler)
  new_pos = pos.where(is_dynamic, pos + new_vel * dt)
  
  # Update orientation from angular velocity
  # q_dot = 0.5 * ω * q (where ω is pure quaternion [0, ωx, ωy, ωz])
  ang_vel_norm = ang_vel.pow(2).sum(axis=1).sqrt()
  has_ang_vel = (ang_vel_norm > 1e-6).reshape(-1, 1)
  w_quat = Tensor.cat(Tensor.zeros(ang_vel.shape[0], 1), ang_vel, dim=1)
  q_dot = quat_multiply(w_quat, quat) * 0.5
  updated_quat = quat + q_dot * dt
  # Normalize to prevent drift from unit quaternion constraint
  norm_quat = updated_quat / updated_quat.pow(2).sum(axis=1, keepdim=True).sqrt()
  new_quat = quat.where(has_ang_vel, norm_quat)
  
  # Assemble updated state (creates new tensor, preserving immutability)
  new_bodies = bodies.detach()
  new_bodies[:, BodySchema.VEL_X:BodySchema.VEL_Z+1] = new_vel
  new_bodies[:, BodySchema.POS_X:BodySchema.POS_Z+1] = new_pos
  new_bodies[:, BodySchema.QUAT_W:BodySchema.QUAT_Z+1] = new_quat
  return new_bodies