"""
Benchmark physics operations performance
"""
import time
import sys
from pathlib import Path
import numpy as np

# Add paths to import modules
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from tinygrad import Tensor
from custom_ops.python.extension import physics_enabled
from custom_ops.python.tensor_ops import PhysicsTensor, create_physics_world

def benchmark_physics():
    """Benchmark physics operations"""
    print("Physics Performance Benchmark")
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

def benchmark_comparison():
    """Compare performance of different approaches"""
    print("\nPerformance Comparison")
    print("=" * 50)
    
    n_bodies = 100
    dt = 0.016
    n_iterations = 100
    
    # Pure TinyGrad approach
    bodies = Tensor.randn(n_bodies, 8) 
    start = time.time()
    for _ in range(n_iterations):
        # Simple gravity integration
        bodies = bodies.numpy()
        bodies[:, 4] -= 9.81 * dt  # Apply gravity to Y velocity
        bodies[:, :3] += bodies[:, 3:6] * dt  # Update positions
        bodies = Tensor(bodies)
    pure_time = time.time() - start
    
    # Custom op approach
    with physics_enabled("CPU"):
        bodies = create_physics_world(n_bodies)
        start = time.time()
        for _ in range(n_iterations):
            bodies = bodies.integrate(dt)
        custom_time = time.time() - start
    
    print(f"Pure TinyGrad: {n_iterations/pure_time:.1f} steps/sec")
    print(f"Custom Op:     {n_iterations/custom_time:.1f} steps/sec")
    print(f"Speedup:       {pure_time/custom_time:.2f}x")

def main():
    # Check if physics library exists
    lib_path = Path(__file__).parent.parent / "build" / ("libphysics.dylib" if sys.platform == "darwin" else "libphysics.so")
    if not lib_path.exists():
        print(f"ERROR: Physics library not found at {lib_path}")
        print("Please run 'make' in the custom_ops/src directory first")
        sys.exit(1)
    
    benchmark_physics()
    benchmark_comparison()

if __name__ == "__main__":
    main()