from tinygrad import Tensor
from ..math_utils import quat_mul, quat_exp, quat_normalize, cross_product, apply_quaternion_to_vector


def predict_state(x: Tensor, q: Tensor, v: Tensor, omega: Tensor, 
                  inv_mass: Tensor, inv_inertia: Tensor, gravity: Tensor, dt: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    gravity_force = gravity.unsqueeze(0).expand(x.shape[0], -1)
    v_new = v + gravity_force * inv_mass.unsqueeze(-1) * dt
    omega_new = omega
    x_pred = x + v_new * dt
    half_omega_dt = 0.5 * omega_new * dt
    delta_q = quat_exp(half_omega_dt)
    q_pred = quat_normalize(quat_mul(q, delta_q))
    return x_pred, q_pred, v_new, omega_new

