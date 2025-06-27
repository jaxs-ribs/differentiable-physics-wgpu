"""Physics engine core that orchestrates the simulation pipeline."""
import numpy as np
from tinygrad import Tensor
from .types import BodySchema
from .integration import integrate
from .broadphase import broadphase_sweep_and_prune
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
  
  def __init__(self, dt: float = 0.016, gravity: np.ndarray = np.array([0, -9.81, 0], dtype=np.float32)):
    """Initialize physics engine.
    
    Args:
      dt: Fixed timestep in seconds (default 60 Hz)
      gravity: Gravity acceleration vector [x, y, z] in m/s²
    """
    self.dt = dt
    self.gravity = Tensor(gravity)
    self.bodies = None
    
  def set_bodies(self, body_list: list[np.ndarray]):
    """Set the bodies for simulation from a list of body arrays."""
    self.bodies = Tensor(np.stack(body_list))
    
  def step(self) -> None:
    """Execute one physics simulation step."""
    # 1. Broadphase collision detection
    pairs = broadphase_sweep_and_prune(self.bodies)
    
    # 2. Narrowphase collision detection
    contacts = narrowphase(self.bodies, pairs)
    
    # 3. Collision resolution
    self.bodies = resolve_collisions(self.bodies, contacts)
    
    # 4. Integration (motion)
    self.bodies = integrate(self.bodies, self.dt, self.gravity)
    
  def get_state(self) -> np.ndarray:
    """Get current state of all bodies as numpy array."""
    return self.bodies.numpy()