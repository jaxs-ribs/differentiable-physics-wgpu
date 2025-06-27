"""
Physics Tensor Operations
High-level tensor operations that get compiled to physics custom ops
"""
from pathlib import Path
import sys
import ctypes
import numpy as np

# Add parent directory to path to import tinygrad
sys.path.append(str(Path(__file__).parent.parent))

from tinygrad import Tensor
from tinygrad.uop.ops import UOp, Ops
from tinygrad.dtype import dtypes
from physics_extension import physics_enabled

class PhysicsTensor(Tensor):
    """
    Extended Tensor class with physics operations
    This demonstrates how physics ops could be integrated
    """
    
    @staticmethod
    def physics_step(bodies: Tensor, dt: float) -> Tensor:
        """
        Perform a physics simulation step
        
        Args:
            bodies: Tensor of shape [N, 8] containing rigid body data
            dt: Time step
            
        Returns:
            Updated bodies tensor
        """
        # Create a custom UOp that represents the physics step
        # In a real implementation, this would be integrated with the tensor's UOp graph
        
        # For demonstration, we'll use ctypes to call the C function directly
        from physics_patterns import get_physics_lib
        lib = get_physics_lib()
        
        # Convert tensor to numpy for ctypes
        bodies_np = bodies.numpy()
        n_bodies = bodies_np.shape[0]
        output_np = np.zeros_like(bodies_np)
        
        # Call C function
        lib.physics_step(
            bodies_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int32(n_bodies),
            ctypes.c_float(dt),
            output_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        
        return Tensor(output_np, device=bodies.device, dtype=bodies.dtype)
    
    def integrate(self, dt: float) -> 'PhysicsTensor':
        """Integrate physics bodies by one time step"""
        return PhysicsTensor.physics_step(self, dt)

def create_physics_world(n_bodies: int = 10, device: str = "CPU") -> PhysicsTensor:
    """
    Create a physics world with random bodies
    
    Returns:
        PhysicsTensor of shape [n_bodies, 8]
    """
    # Create bodies with interesting initial conditions
    bodies_data = []
    
    # Create a grid of bodies
    grid_size = int(np.sqrt(n_bodies))
    spacing = 2.0
    
    for i in range(n_bodies):
        row = i // grid_size
        col = i % grid_size
        
        # Position in a grid pattern, elevated
        pos_x = (col - grid_size/2) * spacing
        pos_y = 10.0 + row * spacing  # Start high up
        pos_z = 0.0
        
        # Small random velocities
        vel_x = np.random.uniform(-0.5, 0.5)
        vel_y = 0.0
        vel_z = np.random.uniform(-0.5, 0.5)
        
        # Varied masses and sizes
        mass = np.random.uniform(1.0, 3.0)
        radius = 0.5 + 0.2 * np.random.randn()
        
        bodies_data.append([pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, mass, radius])
    
    # Create tensor and convert to PhysicsTensor
    tensor_data = Tensor(bodies_data, device=device, dtype=dtypes.float32)
    return PhysicsTensor(tensor_data.data, device=device, dtype=dtypes.float32)

def visualize_positions(bodies: Tensor, step: int):
    """Simple text visualization of body positions"""
    positions = bodies[:, :3].numpy()
    
    print(f"\nStep {step}:")
    print("  Body | X      Y      Z")
    print("  -----|------------------")
    for i, pos in enumerate(positions[:5]):  # Show first 5 bodies
        print(f"  {i:4d} | {pos[0]:6.2f} {pos[1]:6.2f} {pos[2]:6.2f}")
    if len(positions) > 5:
        print(f"  ... ({len(positions)-5} more bodies)")

def run_physics_simulation():
    """Run a complete physics simulation"""
    print("Physics Simulation with TinyGrad Custom Ops")
    print("=" * 50)
    
    # Enable physics operations
    with physics_enabled("CPU"):
        # Create physics world
        bodies = create_physics_world(n_bodies=20)
        dt = 0.016  # 60 FPS
        
        print(f"Created {bodies.shape[0]} rigid bodies")
        print(f"Simulation timestep: {dt}s ({1/dt:.1f} FPS)")
        
        # Initial state
        visualize_positions(bodies, 0)
        
        # Run simulation
        n_steps = 100
        for step in range(1, n_steps + 1):
            # Perform physics step
            bodies = bodies.integrate(dt)
            
            # Visualize every 20 steps
            if step % 20 == 0:
                visualize_positions(bodies, step)
        
        # Final statistics
        final_positions = bodies[:, 1].numpy()  # Y positions
        print(f"\nFinal Y positions: min={final_positions.min():.2f}, "
              f"max={final_positions.max():.2f}, mean={final_positions.mean():.2f}")

def benchmark_physics():
    """Benchmark physics operations"""
    import time
    
    print("\nPhysics Performance Benchmark")
    print("=" * 50)
    
    sizes = [10, 100, 1000]
    
    with physics_enabled("CPU"):
        for n_bodies in sizes:
            bodies = create_physics_world(n_bodies)
            dt = 0.016
            
            # Warmup
            for _ in range(5):
                bodies = bodies.integrate(dt)
            
            # Benchmark
            start = time.time()
            n_iterations = 100
            for _ in range(n_iterations):
                bodies = bodies.integrate(dt)
            elapsed = time.time() - start
            
            steps_per_second = n_iterations / elapsed
            bodies_per_second = n_bodies * steps_per_second
            
            print(f"Bodies: {n_bodies:4d} | "
                  f"Steps/sec: {steps_per_second:7.1f} | "
                  f"Bodies/sec: {bodies_per_second:10.0f}")

if __name__ == "__main__":
    # Check if physics library exists
    lib_path = Path(__file__).parent / ("libphysics.dylib" if sys.platform == "darwin" else "libphysics.so")
    if not lib_path.exists():
        print(f"ERROR: Physics library not found at {lib_path}")
        print("Please run 'make' in the physics_core directory first")
        sys.exit(1)
    
    run_physics_simulation()
    benchmark_physics()