"""Physics engine core that orchestrates the simulation pipeline.

This implementation is fully JIT-compilable using TinyJit, with all operations
expressed as pure tensor computations without Python loops or branches.
"""
import numpy as np
from tinygrad import Tensor, TinyJit
from .types import BodySchema, ExecutionMode
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
               dt: float = 0.016, restitution: float = 0.1, use_differentiable: bool = True,
               execution_mode: ExecutionMode = ExecutionMode.PURE):
    """Initialize physics engine with JIT compilation.
    
    Args:
      bodies: Initial state array of shape (N, NUM_PROPERTIES)
      gravity: Gravity acceleration vector [x, y, z] in m/s²
      dt: Fixed timestep in seconds (default 60 Hz)
      restitution: Coefficient of restitution for collisions (0-1)
      use_differentiable: Use differentiable broadphase (always True for JIT)
      execution_mode: Backend execution mode (PURE, C, or WGPU)
    """
    self.bodies = Tensor(bodies.astype(np.float32))
    self.gravity = Tensor(gravity.astype(np.float32))
    self.dt = dt
    self.restitution = restitution
    self.use_differentiable = True  # Always use differentiable for JIT
    self.execution_mode = execution_mode
    
    # Initialize based on execution mode
    if execution_mode == ExecutionMode.PURE:
      # Pure tensor implementation - use existing JIT compilation
      self.jitted_n_step = TinyJit(lambda bodies, num_steps: _n_step_simulation(bodies, self.dt, self.gravity, num_steps, self.restitution))
      self.jitted_step = TinyJit(self._physics_step)
    elif execution_mode == ExecutionMode.C:
      # C backend - initialize with custom ops
      self._init_c_backend()
    elif execution_mode == ExecutionMode.WGPU:
      # WebGPU backend - placeholder for future implementation
      self._init_wgpu_backend()
    else:
      raise ValueError(f"Unsupported execution mode: {execution_mode}")
  
  def _init_c_backend(self):
    """Initialize C backend with custom operations."""
    import sys
    from pathlib import Path
    
    # Add custom_ops to path
    custom_ops_path = Path(__file__).parent.parent / "custom_ops"
    sys.path.append(str(custom_ops_path))
    
    try:
      from custom_ops.python.extension import enable_physics_on_device, physics_enabled
      from custom_ops.python.patterns import physics_step as c_physics_step
      
      # Check if the C library is compiled
      lib_path = custom_ops_path / "build" / ("libphysics.dylib" if sys.platform == "darwin" else "libphysics.so")
      if not lib_path.exists():
        raise RuntimeError("C physics library not found. Please compile it first by running: cd custom_ops/src && make")
      
      # Enable physics operations on CPU
      enable_physics_on_device("CPU")
      
      # Store C-specific functions
      self._c_physics_step = c_physics_step
      self._physics_enabled = physics_enabled
      
      # For C backend, override the step method behavior
      self.use_c_backend = True
      
      # Create JIT-compiled versions that use C ops
      self.jitted_step = TinyJit(self._physics_step_c)
      self.jitted_n_step = None  # C backend doesn't support N-step yet
    except ImportError as e:
      raise RuntimeError(f"Failed to import C backend: {e}")
  
  def _init_wgpu_backend(self):
    """Initialize WebGPU backend - placeholder for future implementation."""
    raise NotImplementedError("WebGPU backend is not yet implemented")
    
  def _physics_step(self, bodies: Tensor) -> Tensor:
    """Single physics step as a pure tensor function.
    
    This delegates to the static version for consistency.
    
    Args:
      bodies: State tensor of shape (N, NUM_PROPERTIES)
      
    Returns:
      Updated state tensor
    """
    return _physics_step_static(bodies, self.dt, self.gravity, self.restitution)
  
  def _physics_step_c(self, bodies: Tensor) -> Tensor:
    """Single physics step using C custom operations.
    
    Args:
      bodies: State tensor of shape (N, NUM_PROPERTIES)
      
    Returns:
      Updated state tensor
    """
    with self._physics_enabled("CPU"):
      return self._c_physics_step(bodies, self.dt)
  
  def run_simulation(self, num_steps: int) -> None:
    """Run N-step simulation using the JIT-compiled function.
    
    This executes the entire simulation as a single fused computation graph,
    maximizing performance by avoiding Python interpreter overhead.
    
    Args:
      num_steps: Number of simulation steps to run
    """
    if self.execution_mode == ExecutionMode.PURE and self.jitted_n_step is not None:
      # Use optimized N-step for pure mode
      self.bodies = self.jitted_n_step(self.bodies, num_steps)
    else:
      # Fall back to single-step iteration for other modes
      for _ in range(num_steps):
        self.step()
  
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