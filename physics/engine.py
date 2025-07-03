import numpy as np
from tinygrad import Tensor, TinyJit
from .xpbd.broadphase import uniform_spatial_hash
from .xpbd.narrowphase import generate_contacts
from .xpbd.solver import solve_constraints
from .xpbd.velocity_update import reconcile_velocities

def _physics_step_static(bodies: Tensor, dt: float, gravity: Tensor, restitution: float = 0.1) -> Tensor:
  # Store original positions for velocity reconciliation
  bodies_old = bodies
  
  # 1. Apply external forces (gravity) to predict positions
  # TODO: Implement external force prediction (Milestone 1)
  bodies_pred = bodies  # Placeholder - should predict positions from forces
  
  # 2. Broadphase collision detection using uniform spatial hash
  candidate_pairs = uniform_spatial_hash(bodies_pred)
  
  # 3. Narrowphase collision detection to generate contacts
  contacts = generate_contacts(bodies_pred, candidate_pairs)
  
  # 4. Solve position constraints iteratively
  bodies_proj = solve_constraints(bodies_pred, contacts, iterations=8)
  
  # 5. Reconcile velocities from position changes
  bodies = reconcile_velocities(bodies_proj, bodies_old, dt)
  
  return bodies

def _n_step_simulation(initial_bodies: Tensor, dt: float, gravity: Tensor, num_steps: int, 
                      restitution: float = 0.1) -> Tensor:
  bodies = initial_bodies
  for _ in range(num_steps):
    bodies = _physics_step_static(bodies, dt, gravity, restitution)
  return bodies

class TensorPhysicsEngine:
  
  def __init__(self, bodies: np.ndarray, gravity: np.ndarray = np.array([0, -9.81, 0], dtype=np.float32),
               dt: float = 0.016, restitution: float = 0.1):
    self.bodies = Tensor(bodies.astype(np.float32))
    self.gravity = Tensor(gravity.astype(np.float32))
    self.dt = dt
    self.restitution = restitution
    
    self.jitted_n_step = TinyJit(lambda bodies, num_steps: _n_step_simulation(
      bodies, self.dt, self.gravity, num_steps, self.restitution))
    self.jitted_step = TinyJit(self._physics_step)
    
  def _physics_step(self, bodies: Tensor) -> Tensor:
    return _physics_step_static(bodies, self.dt, self.gravity, self.restitution)
  
  def run_simulation(self, num_steps: int) -> None:
    self.bodies = self.jitted_n_step(self.bodies, num_steps)
  
  def step(self, dt: float | None = None) -> Tensor:
    if dt is not None and dt != self.dt:
      self.dt = dt
      self.jitted_step = TinyJit(self._physics_step)
    
    self.bodies = self.jitted_step(self.bodies)
    return self.bodies
  
  def get_state(self) -> np.ndarray:
    return self.bodies.numpy()
  
  def set_bodies(self, bodies: np.ndarray) -> None:
    self.bodies = Tensor(bodies.astype(np.float32))

PhysicsEngine = TensorPhysicsEngine