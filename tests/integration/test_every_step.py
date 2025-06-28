#!/usr/bin/env python3
"""Print every step around collision."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['JIT'] = '0'

from physics.engine import TensorPhysicsEngine
from physics.types import create_body_array, ShapeType

def test_every_step():
    """Print every single step."""
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
    
    # Ball
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
    
    print("Looking for collision around step 1100...")
    
    for step in range(1110):
        state = engine.get_state()
        y = state[1, 1]
        vy = state[1, 4]
        
        if step >= 1100:
            print(f"Step {step}: y={y:.6f}, vy={vy:.6f}")
        
        engine.step()
        
        if step >= 1100:
            state_after = engine.get_state()
            y_after = state_after[1, 1]
            vy_after = state_after[1, 4]
            if abs(vy_after - vy) > 1.0:
                print(f"  -> After step: y={y_after:.6f}, vy={vy_after:.6f}")
                print(f"  -> LARGE CHANGE: Δvy = {vy_after - vy:.6f}")
                break

if __name__ == "__main__":
    test_every_step()