#!/usr/bin/env python3
"""Debug version of bounce test."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.engine import TensorPhysicsEngine
from physics.types import create_body_array, ShapeType

# Run without JIT for easier debugging
os.environ['JIT'] = '0'

def test_bounce_debug():
    """Debug the bounce issue."""
    print("\n=== Bounce Debug Test ===")
    
    bodies = []
    
    # Ground at y=-2
    bodies.append(create_body_array(
        position=np.array([0., -2., 0.], dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1e8,
        inertia=np.eye(3, dtype=np.float32) * 1e8,
        shape_type=ShapeType.BOX,
        shape_params=np.array([10., 0.5, 10.], dtype=np.float32)
    ))
    
    # Ball at y=5
    bodies.append(create_body_array(
        position=np.array([0., 5.0, 0.], dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1.0,
        inertia=np.eye(3, dtype=np.float32) * 0.1,
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([0.5, 0., 0.], dtype=np.float32)
    ))
    
    engine = TensorPhysicsEngine(np.stack(bodies), dt=0.001)
    
    prev_vy = 0
    for step in range(1500):
        if step % 100 == 0:
            state = engine.get_state()
            y = state[1, 1]
            vy = state[1, 4]
            print(f"Step {step}: y={y:.3f}, vy={vy:.3f}")
        
        # Get state before step
        state_before = engine.get_state()
        vy_before = state_before[1, 4]
        
        # Do step
        engine.step()
        
        # Get state after step
        state_after = engine.get_state()
        vy_after = state_after[1, 4]
        
        # Check for large velocity change
        if abs(vy_after - vy_before) > 5.0:
            print(f"\nLARGE VELOCITY CHANGE at step {step}!")
            print(f"  Before: vy={vy_before:.6f}")
            print(f"  After: vy={vy_after:.6f}")
            print(f"  Change: {vy_after - vy_before:.6f}")
            print(f"  Position: y={state_before[1,1]:.6f}")
            break

if __name__ == "__main__":
    test_bounce_debug()