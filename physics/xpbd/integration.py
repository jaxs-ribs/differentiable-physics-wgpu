from tinygrad import Tensor
from ..math_utils import quat_mul, quat_exp, quat_normalize, cross_product, apply_quaternion_to_vector


def predict_state(x: Tensor, q: Tensor, v: Tensor, omega: Tensor, 
                  inv_mass: Tensor, inv_inertia: Tensor, gravity: Tensor, dt: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Perform forward prediction step of XPBD physics simulation.
    
    Updates velocities based on forces and predicts next state for positions and orientations.
    
    Args:
        x: Current positions (N, 3)
        q: Current orientations as quaternions (N, 4) in format [w, x, y, z]
        v: Current linear velocities (N, 3)
        omega: Current angular velocities (N, 3)
        inv_mass: Inverse masses (N,)
        inv_inertia: Inverse inertia tensors (N, 3, 3)
        gravity: Gravity vector (3,)
        dt: Time step
    
    Returns:
        Tuple of (x_pred, q_pred, v_new, omega_new)
        - x_pred: Predicted positions (N, 3)
        - q_pred: Predicted orientations (N, 4)
        - v_new: Updated linear velocities (N, 3)
        - omega_new: Updated angular velocities (N, 3)
    """
    # 1. Update linear velocity based on external forces
    # f_total = m * g for each body (gravity is the only force for now)
    # v_new = v + f_total * inv_mass * dt
    # Broadcast gravity to all bodies and scale by inverse mass
    gravity_force = gravity.unsqueeze(0).expand(x.shape[0], -1)
    v_new = v + gravity_force * inv_mass.unsqueeze(-1) * dt
    
    # 2. Update angular velocity based on torques
    # For now, we'll skip the gyroscopic term to keep things simple
    # In a full implementation, we would compute: τ_total = -ω × Iω
    # And then: ω_new = ω + I^(-1) * τ_total * dt
    omega_new = omega  # No torques for now
    
    # 3. Predict position: x_pred = x + v_new * dt
    x_pred = x + v_new * dt
    
    # 4. Predict orientation using quaternion exponential map
    # q_pred = normalize(q + 0.5 * [0, ω] ⊗ q * dt)
    # More stable: q_pred = normalize(q ⊗ exp(0.5 * ω * dt))
    half_omega_dt = 0.5 * omega_new * dt
    delta_q = quat_exp(half_omega_dt)
    q_pred = quat_normalize(quat_mul(q, delta_q))
    
    return x_pred, q_pred, v_new, omega_new


# Keep the old function for backwards compatibility temporarily
def integrate(x: Tensor, q: Tensor, v: Tensor, omega: Tensor, inv_mass: Tensor, dt: float, gravity: Tensor) -> tuple[Tensor, Tensor]:
    """Legacy integrate function - redirects to predict_state."""
    # Create dummy inverse inertia for backwards compatibility
    inv_inertia = Tensor.ones((x.shape[0], 3, 3))
    x_pred, q_pred, _, _ = predict_state(x, q, v, omega, inv_mass, inv_inertia, gravity, dt)
    return x_pred, q_pred