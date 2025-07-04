from tinygrad import Tensor
from ..math_utils import quat_mul

def reconcile_velocities(x_proj: Tensor, q_proj: Tensor, x_old: Tensor, q_old: Tensor, 
                        v_old: Tensor, omega_old: Tensor, dt: float) -> tuple[Tensor, Tensor]:
    v_reconciled = (x_proj - x_old) / dt
    
    q_old_conj = q_old * Tensor([1, -1, -1, -1]).unsqueeze(0)
    
    delta_q = quat_mul(q_proj, q_old_conj)
    
    delta_q_w = delta_q[:, 0:1]
    delta_q_xyz = delta_q[:, 1:]
    
    angle = 2.0 * delta_q_w.clip(-1.0, 1.0).acos()
    
    sin_half_angle = (angle / 2.0).sin()
    axis = (sin_half_angle.abs() > 1e-6).where(
        delta_q_xyz / sin_half_angle, 
        Tensor.zeros_like(delta_q_xyz)
    )
    
    omega_reconciled = (angle / dt) * axis
    
    return v_reconciled, omega_reconciled