import numpy as np
from tinygrad import Tensor, TinyJit
from .xpbd.broadphase import uniform_spatial_hash
from .xpbd.narrowphase import generate_contacts
from .xpbd.solver import solve_constraints
from .xpbd.velocity_update import reconcile_velocities
from .xpbd.integration import predict_state

def _physics_step_static(x: Tensor, q: Tensor, v: Tensor, omega: Tensor, 
                         inv_mass: Tensor, inv_inertia: Tensor, shape_type: Tensor, shape_params: Tensor,
                         dt: float, gravity: Tensor, restitution: float = 0.1) -> tuple[Tensor, Tensor, Tensor, Tensor]:
  # Store original state for velocity reconciliation
  x_old, q_old = x, q
  
  # 1. Forward Prediction: Apply external forces and predict new state
  x_pred, q_pred, v_new, omega_new = predict_state(x, q, v, omega, inv_mass, inv_inertia, gravity, dt)
  
  # 2. Collision Detection: Broadphase using uniform spatial hash
  candidate_pairs = uniform_spatial_hash(x_pred, shape_type, shape_params)
  
  # 3. Collision Detection: Narrowphase to generate contacts
  contacts = generate_contacts(x_pred, q_pred, candidate_pairs, shape_type, shape_params)
  
  # 4. XPBD Solve: Position constraint solving
  x_proj, q_proj = solve_constraints(x_pred, q_pred, contacts, inv_mass, inv_inertia, iterations=8)
  
  # 5. Velocity Reconciliation: Update velocities from position changes
  v_final, omega_final = reconcile_velocities(x_proj, q_proj, x_old, q_old, v_new, omega_new, dt)
  
  return x_proj, q_proj, v_final, omega_final

def _n_step_simulation(x: Tensor, q: Tensor, v: Tensor, omega: Tensor,
                      inv_mass: Tensor, inv_inertia: Tensor, shape_type: Tensor, shape_params: Tensor,
                      dt: float, gravity: Tensor, num_steps: int, restitution: float = 0.1) -> tuple[Tensor, Tensor, Tensor, Tensor]:
  for _ in range(num_steps):
    x, q, v, omega = _physics_step_static(x, q, v, omega, inv_mass, inv_inertia, shape_type, shape_params, dt, gravity, restitution)
  return x, q, v, omega

class TensorPhysicsEngine:
  
  def __init__(self, x: np.ndarray, q: np.ndarray, v: np.ndarray, omega: np.ndarray,
               inv_mass: np.ndarray, inv_inertia: np.ndarray, shape_type: np.ndarray, shape_params: np.ndarray,
               gravity: np.ndarray = np.array([0, -9.81, 0], dtype=np.float32),
               dt: float = 0.016, restitution: float = 0.1):
    # Convert to tensors and store SoA state
    self.x = Tensor(x.astype(np.float32))
    self.q = Tensor(q.astype(np.float32))
    self.v = Tensor(v.astype(np.float32))
    self.omega = Tensor(omega.astype(np.float32))
    self.inv_mass = Tensor(inv_mass.astype(np.float32))
    self.inv_inertia = Tensor(inv_inertia.astype(np.float32))
    self.shape_type = Tensor(shape_type.astype(np.int32))
    self.shape_params = Tensor(shape_params.astype(np.float32))
    
    self.gravity = Tensor(gravity.astype(np.float32))
    self.dt = dt
    self.restitution = restitution
    
    self.jitted_n_step = TinyJit(lambda x, q, v, omega, num_steps: _n_step_simulation(
      x, q, v, omega, self.inv_mass, self.inv_inertia, self.shape_type, self.shape_params,
      self.dt, self.gravity, num_steps, self.restitution))
    self.jitted_step = TinyJit(lambda x, q, v, omega: _physics_step_static(
      x, q, v, omega, self.inv_mass, self.inv_inertia, self.shape_type, self.shape_params,
      self.dt, self.gravity, self.restitution))
    
  def _physics_step(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return _physics_step_static(self.x, self.q, self.v, self.omega, 
                               self.inv_mass, self.inv_inertia, self.shape_type, self.shape_params,
                               self.dt, self.gravity, self.restitution)
  
  def run_simulation(self, num_steps: int) -> None:
    self.x, self.q, self.v, self.omega = self.jitted_n_step(self.x, self.q, self.v, self.omega, num_steps)
  
  def step(self, dt: float | None = None) -> None:
    if dt is not None and dt != self.dt:
      self.dt = dt
      self.jitted_step = TinyJit(lambda x, q, v, omega: _physics_step_static(
        x, q, v, omega, self.inv_mass, self.inv_inertia, self.shape_type, self.shape_params,
        self.dt, self.gravity, self.restitution))
    
    self.x, self.q, self.v, self.omega = self.jitted_step(self.x, self.q, self.v, self.omega)
  
  def get_state(self) -> dict[str, np.ndarray]:
    return {
      'x': self.x.numpy(),
      'q': self.q.numpy(),
      'v': self.v.numpy(),
      'omega': self.omega.numpy(),
      'inv_mass': self.inv_mass.numpy(),
      'inv_inertia': self.inv_inertia.numpy(),
      'shape_type': self.shape_type.numpy(),
      'shape_params': self.shape_params.numpy()
    }
  
  def set_state(self, x: np.ndarray, q: np.ndarray, v: np.ndarray, omega: np.ndarray) -> None:
    self.x = Tensor(x.astype(np.float32))
    self.q = Tensor(q.astype(np.float32))
    self.v = Tensor(v.astype(np.float32))
    self.omega = Tensor(omega.astype(np.float32))

PhysicsEngine = TensorPhysicsEngine