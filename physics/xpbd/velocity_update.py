from tinygrad import Tensor

def reconcile_velocities(x_proj: Tensor, q_proj: Tensor, x_old: Tensor, q_old: Tensor, 
                        v_old: Tensor, omega_old: Tensor, dt: float) -> tuple[Tensor, Tensor]:
  return v_old, omega_old