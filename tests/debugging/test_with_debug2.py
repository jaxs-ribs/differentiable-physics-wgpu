#!/usr/bin/env python3
"""Test with debug solver 2."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['JIT'] = '0'

# Import debug solver BEFORE engine
import physics.solver_debug2

from physics.engine import TensorPhysicsEngine
from physics.types import create_body_array, ShapeType, BodySchema

def test_debug2():
    """Test with debug2."""
    bodies = []
    
    # Static ground box at y=-2
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
    
    print("Running until bounce...")
    prev_vy = 0
    for step in range(2000):
        state = engine.get_state()
        vy = state[1, 4]
        
        if prev_vy < 0 and vy > 0:
            print(f"Bounce detected at step {step}")
            break
        prev_vy = vy
        engine.step()

if __name__ == "__main__":
    test_debug2()