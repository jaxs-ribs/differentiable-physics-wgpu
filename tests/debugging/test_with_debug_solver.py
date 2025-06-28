#!/usr/bin/env python3
"""Test with debug solver."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import debug solver BEFORE engine
import physics.solver_debug

from physics.engine import TensorPhysicsEngine
from physics.types import create_body_array, ShapeType

def test_with_debug():
    """Test with debug output."""
    print("\n=== Test with Debug Solver ===")
    
    bodies = []
    
    # Ground
    bodies.append(create_body_array(
        position=np.array([0., 0., 0.], dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1e8,
        inertia=np.eye(3, dtype=np.float32) * 1e8,
        shape_type=ShapeType.BOX,
        shape_params=np.array([10., 0.5, 10.], dtype=np.float32)
    ))
    
    # Ball
    bodies.append(create_body_array(
        position=np.array([0., 1.5, 0.], dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1.0,
        inertia=np.eye(3, dtype=np.float32) * 0.1,
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([0.5, 0., 0.], dtype=np.float32)
    ))
    
    engine = TensorPhysicsEngine(np.stack(bodies), dt=0.001)
    
    print("Running until collision...")
    
    for step in range(500):
        state = engine.get_state()
        vy = state[1, 4]
        
        if step % 100 == 0:
            print(f"  Step {step}: vy={vy:.3f}")
        
        engine.step()
        
        # Check for bounce
        state_after = engine.get_state()
        vy_after = state_after[1, 4]
        
        if vy < 0 and vy_after > 0:
            print(f"\nBounce at step {step}!")
            print(f"  Velocity changed from {vy:.3f} to {vy_after:.3f}")
            break

if __name__ == "__main__":
    test_with_debug()