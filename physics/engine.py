"""Physics engine core that orchestrates the simulation pipeline.

This implementation is fully JIT-compilable using TinyJit, with all operations
expressed as pure tensor computations without Python loops or branches.
"""
import numpy as np
from tinygrad import Tensor, TinyJit
from .types import BodySchema
from .integration import integrate
from .broadphase_tensor import differentiable_broadphase
from .narrowphase import narrowphase
from .solver import resolve_collisions

def _physics_step_static(bodies: Tensor, dt: float, gravity: Tensor, restitution: float = 0.1) -> Tensor:
  """Static physics step function for use in N-step simulation.
  
  This function is defined outside the class so it can be called
  in the N-step loop without method overhead.
  
  Args:
    bodies: State tensor of shape (N, NUM_PROPERTIES)
    dt: Timestep in seconds
    gravity: Gravity acceleration vector
    restitution: Coefficient of restitution for collisions
    
  Returns:
    Updated state tensor
  """
  # 1. Broadphase collision detection
  pair_indices, collision_mask = differentiable_broadphase(bodies)
  
  # 2. Narrowphase collision detection
  contact_normals, contact_depths, contact_points, contact_mask, pair_indices = narrowphase(
    bodies, pair_indices, collision_mask
  )
  
  # 3. Collision resolution
  bodies = resolve_collisions(
    bodies, pair_indices, contact_normals, contact_depths, 
    contact_points, contact_mask, restitution=0.1
  )
  
  # 4. Integration (motion)
  bodies = integrate(bodies, dt, gravity)
  
  return bodies

def _n_step_simulation(initial_bodies: Tensor, dt: float, gravity: Tensor, num_steps: int, restitution: float = 0.1) -> Tensor:
  """N-step simulation function that can be JIT-compiled.
  
  This function contains the simulation loop and will be unrolled by TinyJit
  to create a single fused computation graph for the entire simulation.
  
  Args:
    initial_bodies: Initial state tensor of shape (N, NUM_PROPERTIES)
    dt: Fixed timestep in seconds
    gravity: Gravity acceleration vector
    num_steps: Number of simulation steps to run
    restitution: Coefficient of restitution for collisions
    
  Returns:
    Final state tensor after N steps
  """
  bodies = initial_bodies
  for _ in range(num_steps):
    bodies = _physics_step_static(bodies, dt, gravity)
  return bodies

class TensorPhysicsEngine:
  """JIT-compilable physics engine using pure tensor operations.
  
  The physics pipeline consists of four phases executed each timestep:
  1. Broadphase: Find potentially colliding pairs using AABBs
  2. Narrowphase: Exact collision detection to generate contacts
  3. Solver: Apply impulses to resolve collisions
  4. Integration: Update positions and orientations from velocities
  
  All operations are vectorized and JIT-compiled for maximum performance.
  This engine now supports N-step JIT compilation for entire simulations.
  """
  
  def __init__(self, bodies: np.ndarray, gravity: np.ndarray = np.array([0, -9.81, 0], dtype=np.float32),
               dt: float = 0.016, restitution: float = 0.1, use_differentiable: bool = True):
    """Initialize physics engine with JIT compilation.
    
    Args:
      bodies: Initial state array of shape (N, NUM_PROPERTIES)
      gravity: Gravity acceleration vector [x, y, z] in m/s²
      dt: Fixed timestep in seconds (default 60 Hz)
      restitution: Coefficient of restitution for collisions (0-1)
      use_differentiable: Use differentiable broadphase (always True for JIT)
    """
    self.bodies = Tensor(bodies.astype(np.float32))
    self.gravity = Tensor(gravity.astype(np.float32))
    self.dt = dt
    self.restitution = restitution
    self.use_differentiable = True  # Always use differentiable for JIT
    
    # JIT compile the N-step simulation function with bound parameters
    self.jitted_n_step = TinyJit(lambda bodies, num_steps: _n_step_simulation(bodies, self.dt, self.gravity, num_steps, self.restitution))
    
    # Also keep single-step for debugging
    self.jitted_step = TinyJit(self._physics_step)
    
  def _physics_step(self, bodies: Tensor) -> Tensor:
    """Single physics step as a pure tensor function.
    
    This delegates to the static version for consistency.
    
    Args:
      bodies: State tensor of shape (N, NUM_PROPERTIES)
      
    Returns:
      Updated state tensor
    """
    return _physics_step_static(bodies, self.dt, self.gravity, self.restitution)
  
  def run_simulation(self, num_steps: int) -> None:
    """Run N-step simulation using the JIT-compiled function.
    
    This executes the entire simulation as a single fused computation graph,
    maximizing performance by avoiding Python interpreter overhead.
    
    Args:
      num_steps: Number of simulation steps to run
    """
    self.bodies = self.jitted_n_step(self.bodies, num_steps)
  
  def step(self, dt: float | None = None) -> Tensor:
    """Execute one physics simulation step.
    
    Args:
      dt: Optional timestep override (uses self.dt if not provided)
    """
    if dt is not None and dt != self.dt:
      # If timestep changes, we need to update and recompile
      self.dt = dt
      self.jitted_step = TinyJit(self._physics_step)
    
    # Run the JIT-compiled physics step
    self.bodies = self.jitted_step(self.bodies)
    return self.bodies  # Return for compatibility with tests
  
  def get_state(self) -> np.ndarray:
    """Get current state of all bodies as numpy array."""
    return self.bodies.numpy()
  
  def set_bodies(self, bodies: np.ndarray) -> None:
    """Update the bodies state tensor."""
    self.bodies = Tensor(bodies.astype(np.float32))

# For backward compatibility
PhysicsEngine = TensorPhysicsEngine