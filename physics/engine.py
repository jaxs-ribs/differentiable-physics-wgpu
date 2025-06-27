"""Physics engine core that orchestrates the simulation pipeline."""
import numpy as np
from tinygrad import Tensor
from .types import BodySchema
from .integration import integrate
from .broadphase import broadphase_sweep_and_prune, differentiable_broadphase
from .narrowphase import narrowphase
from .solver import resolve_collisions

class PhysicsEngine:
  """Main physics engine that runs the simulation pipeline.
  
  The physics pipeline consists of four phases executed each timestep:
  1. Broadphase: Find potentially colliding pairs using AABBs
  2. Narrowphase: Exact collision detection to generate contacts
  3. Solver: Apply impulses to resolve collisions
  4. Integration: Update positions and orientations from velocities
  
  This ordering ensures collisions are resolved before motion, preventing
  bodies from sinking into each other over time.
  """
  
  def __init__(self, dt: float = 0.016, gravity: np.ndarray = np.array([0, -9.81, 0], dtype=np.float32), 
               use_differentiable: bool = True):
    """Initialize physics engine.
    
    Args:
      dt: Fixed timestep in seconds (default 60 Hz)
      gravity: Gravity acceleration vector [x, y, z] in m/s²
      use_differentiable: Use differentiable broadphase (default True)
    """
    self.dt = dt
    self.gravity = Tensor(gravity)
    self.bodies = None
    self.use_differentiable = use_differentiable
    
  def set_bodies(self, body_list: list[np.ndarray]):
    """Set the bodies for simulation from a list of body arrays."""
    self.bodies = Tensor(np.stack(body_list))
    
  def step(self) -> None:
    """Execute one physics simulation step."""
    # 1. Broadphase collision detection
    if self.use_differentiable:
      pair_indices, collision_mask = differentiable_broadphase(self.bodies)
      # 2. Narrowphase collision detection
      contacts = narrowphase(self.bodies, pair_indices, collision_mask)
    else:
      # Legacy sweep and prune
      pairs = broadphase_sweep_and_prune(self.bodies)
      # Convert to new format for narrowphase
      if pairs.shape[0] > 0:
        collision_mask = Tensor.ones(pairs.shape[0], dtype=Tensor.default_type).bool()
        contacts = narrowphase(self.bodies, pairs, collision_mask)
      else:
        contacts = []
    
    # 3. Collision resolution
    self.bodies = resolve_collisions(self.bodies, contacts)
    
    # 4. Integration (motion)
    self.bodies = integrate(self.bodies, self.dt, self.gravity)
    
  def get_state(self) -> np.ndarray:
    """Get current state of all bodies as numpy array."""
    return self.bodies.numpy()