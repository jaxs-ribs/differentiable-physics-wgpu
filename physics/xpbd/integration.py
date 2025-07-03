from tinygrad import Tensor
from ..math_utils import quat_mul, quat_exp, quat_normalize, cross_product, apply_quaternion_to_vector


def predict_state(x: Tensor, q: Tensor, v: Tensor, omega: Tensor, 
                  inv_mass: Tensor, inv_inertia: Tensor, gravity: Tensor, dt: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
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

