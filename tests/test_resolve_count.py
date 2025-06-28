#!/usr/bin/env python3
"""Count how many times resolve_collisions is called."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['JIT'] = '0'

# Counter
resolve_count = 0
step_count = 0

# Patch resolve_collisions
import physics.solver
_original_resolve = physics.solver.resolve_collisions

def resolve_collisions_counter(*args, **kwargs):
    global resolve_count
    resolve_count += 1
    return _original_resolve(*args, **kwargs)

physics.solver.resolve_collisions = resolve_collisions_counter

# Patch _physics_step_static
import physics.engine
_original_step = physics.engine._physics_step_static

def step_counter(*args, **kwargs):
    global step_count
    step_count += 1
    return _original_step(*args, **kwargs)

physics.engine._physics_step_static = step_counter

from physics.engine import TensorPhysicsEngine
from physics.types import create_body_array, ShapeType

def test_count():
    """Count resolve calls."""
    global resolve_count, step_count
    
    bodies = []
    
    # Ground
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
    
    print("Running simulation...")
    resolve_count = 0
    step_count = 0
    
    prev_vy = 0
    for i in range(1200):
        state = engine.get_state()
        vy = state[1, 4]
        
        if prev_vy < 0 and vy > 0:
            print(f"\nBounce at step {i}!")
            print(f"Total physics steps: {step_count}")
            print(f"Total resolve calls: {resolve_count}")
            print(f"Ratio: {resolve_count / step_count:.2f} resolves per step")
            break
        
        prev_vy = vy
        engine.step()

if __name__ == "__main__":
    test_count()