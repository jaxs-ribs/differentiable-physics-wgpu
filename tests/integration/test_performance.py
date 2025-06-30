"""Test performance characteristics of the physics engine.

WHAT: Benchmarks the physics engine to ensure it meets minimum 
      performance requirements and validates JIT compilation speedup.

WHY: Performance is critical for real-time physics:
     - Must achieve 60+ steps/second for real-time simulation
     - JIT compilation should provide significant speedup
     - Performance regressions break user applications
     - Helps identify optimization opportunities

HOW: - Creates scenes with multiple bodies to stress the engine
     - Times simulation steps and calculates steps/second
     - Compares first run (includes JIT compilation) vs subsequent runs
     - Uses time.time() for wall-clock timing
     - CI mode uses fewer iterations to keep tests fast
"""
import time
import os
import numpy as np
import pytest
from physics.engine import TensorPhysicsEngine
from physics.types import BodySchema, ShapeType, create_body_array

def test_simulation_performance():
    """Verify simulation runs at acceptable speed."""
    # Create a scene with multiple bodies
    num_bodies = 10
    bodies_list = []
    
    for i in range(num_bodies):
        body = create_body_array(
            position=np.array([i * 2.0, 5.0, 0], dtype=np.float32),
            velocity=np.array([0, -1.0, 0], dtype=np.float32),
            orientation=np.array([1, 0, 0, 0], dtype=np.float32),
            angular_vel=np.zeros(3, dtype=np.float32),
            mass=1.0,
            inertia=np.eye(3, dtype=np.float32),
            shape_type=ShapeType.SPHERE,
            shape_params=np.array([0.5, 0, 0], dtype=np.float32)
        )
        bodies_list.append(body)
    
    bodies = np.stack(bodies_list)
    engine = TensorPhysicsEngine(bodies, dt=0.016)
    
    # Time 100 steps
    start_time = time.time()
    steps = 100 if os.environ.get('CI') == 'true' else 1000
    
    for _ in range(steps):
        engine.step()
    
    elapsed = time.time() - start_time
    steps_per_second = steps / elapsed
    
    print(f"Performance: {steps_per_second:.1f} steps/second")
    
    # Should achieve at least 60 steps/second (real-time for 60Hz)
    assert steps_per_second > 60, f"Too slow: {steps_per_second:.1f} steps/s"

def test_jit_speedup():
    """Verify JIT compilation provides speedup."""
    os.environ['JIT'] = '1'
    
    # Simple 2-body scene
    bodies = np.zeros((2, BodySchema.NUM_PROPERTIES), dtype=np.float32)
    bodies[0, BodySchema.QUAT_W] = 1.0
    bodies[0, BodySchema.SHAPE_TYPE] = ShapeType.SPHERE
    bodies[0, BodySchema.INV_MASS] = 1.0
    bodies[1, BodySchema.QUAT_W] = 1.0
    bodies[1, BodySchema.SHAPE_TYPE] = ShapeType.SPHERE
    bodies[1, BodySchema.INV_MASS] = 1.0
    
    engine = TensorPhysicsEngine(bodies, dt=0.016)
    
    # First run (includes compilation)
    start = time.time()
    engine.run_simulation(10)
    first_run_time = time.time() - start
    
    # Subsequent runs (using compiled code)
    times = []
    for _ in range(5):
        engine = TensorPhysicsEngine(bodies, dt=0.016)
        start = time.time()
        engine.run_simulation(10)
        times.append(time.time() - start)
    
    avg_compiled_time = np.mean(times)
    
    # Compiled runs should be faster
    print(f"First run: {first_run_time:.4f}s, Avg compiled: {avg_compiled_time:.4f}s")
    
    # In CI, just check it runs without error
    if os.environ.get('CI') != 'true':
        assert avg_compiled_time < first_run_time * 0.8, "JIT should provide speedup"

if __name__ == "__main__":
    test_simulation_performance()
    test_jit_speedup()
    print("Performance tests passed!")