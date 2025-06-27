"""
Physics Demo using TinyGrad with Custom Physics Operations
Demonstrates how to use physics operations on standard TinyGrad devices
"""
import numpy as np
from pathlib import Path
import sys

# Add path to import tinygrad and custom_ops
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from tinygrad import Tensor, Device
from tinygrad.helpers import DEBUG
from custom_ops.python.extension import enable_physics_on_device, physics_enabled
from custom_ops.python.patterns import physics_step

def create_test_bodies(n_bodies: int = 10):
    """
    Create test rigid bodies with random initial conditions
    
    Returns:
        Tensor of shape [n_bodies, 8] with format:
        [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, mass, radius]
    """
    # Create bodies with random positions and velocities
    bodies_data = []
    
    for i in range(n_bodies):
        # Position (spread out in 3D space)
        pos_x = np.random.uniform(-10, 10)
        pos_y = np.random.uniform(0, 20)  # Start above ground
        pos_z = np.random.uniform(-10, 10)
        
        # Velocity (small random velocities)
        vel_x = np.random.uniform(-2, 2)
        vel_y = np.random.uniform(-1, 1)
        vel_z = np.random.uniform(-2, 2)
        
        # Physical properties
        mass = np.random.uniform(0.5, 2.0)
        radius = np.random.uniform(0.3, 1.0)
        
        bodies_data.append([pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, mass, radius])
    
    return Tensor(bodies_data, dtype="float32")

def simulate_physics_basic():
    """Basic physics simulation without custom ops (for comparison)"""
    print("=== Basic Physics Simulation (Pure TinyGrad) ===")
    
    # Create test bodies
    bodies = create_test_bodies(5)
    dt = 0.016  # 60 FPS timestep
    
    print(f"Initial bodies shape: {bodies.shape}")
    print(f"Initial positions (Y): {bodies[:, 1].numpy()}")
    
    # Simple gravity integration without custom ops
    gravity = Tensor([-0.0, -9.81, 0.0], dtype="float32")
    
    for step in range(10):
        # Extract components
        positions = bodies[:, :3]
        velocities = bodies[:, 3:6]
        masses = bodies[:, 6:7]
        
        # Apply gravity to velocities
        velocities = velocities + gravity.unsqueeze(0) * dt
        
        # Update positions
        positions = positions + velocities * dt
        
        # Reconstruct bodies tensor
        bodies = positions.cat(velocities, bodies[:, 6:], dim=1)
        
        if step % 3 == 0:
            print(f"Step {step}: Y positions: {positions[:, 1].numpy()}")

def simulate_physics_custom():
    """Physics simulation using custom ops"""
    print("\n=== Physics Simulation with Custom Ops ===")
    
    # Enable physics on CPU device
    enable_physics_on_device("CPU")
    
    try:
        # Create test bodies
        bodies = create_test_bodies(5)
        dt = 0.016  # 60 FPS timestep
        
        print(f"Initial bodies shape: {bodies.shape}")
        print(f"Initial positions (Y): {bodies[:, 1].numpy()}")
        
        # NOTE: In a real implementation, we would integrate this with Tensor operations
        # For now, this demonstrates the concept
        
        # The physics_step function would be called like this:
        # bodies = physics_step(bodies, dt)
        
        # Since we can't fully integrate without modifying TinyGrad core,
        # we'll demonstrate the pattern matching and device extension
        
        print("\nPhysics operations enabled on CPU device")
        print("Pattern matcher ready to recognize physics operations")
        print("Custom C functions loaded and ready")
        
    finally:
        # Clean up
        from custom_ops.python.extension import disable_physics_on_device
        disable_physics_on_device("CPU")

def demonstrate_context_manager():
    """Demonstrate using the context manager for physics operations"""
    print("\n=== Physics with Context Manager ===")
    
    bodies = create_test_bodies(3)
    
    # Use context manager to temporarily enable physics
    with physics_enabled("CPU"):
        print("Physics operations enabled within context")
        # Perform physics operations here
        # bodies = physics_step(bodies, 0.016)
    
    print("Physics operations disabled after context")

def main():
    """Run all demonstrations"""
    print("TinyGrad Physics Extension Demo")
    print("=" * 50)
    
    # Check if physics library is compiled
    from pathlib import Path
    lib_path = Path(__file__).parent.parent / "build" / ("libphysics.dylib" if sys.platform == "darwin" else "libphysics.so")
    if not lib_path.exists():
        print(f"ERROR: Physics library not found at {lib_path}")
        print("Please run 'make' in the physics_core directory first")
        return
    
    # Run demonstrations
    simulate_physics_basic()
    simulate_physics_custom()
    demonstrate_context_manager()
    
    print("\n" + "=" * 50)
    print("Demo completed!")
    print("\nNext steps for full integration:")
    print("1. Modify TinyGrad's Tensor class to support physics operations")
    print("2. Implement pattern recognition for actual physics computations")
    print("3. Add support for more complex physics operations")
    print("4. Optimize for GPU execution")

if __name__ == "__main__":
    main()