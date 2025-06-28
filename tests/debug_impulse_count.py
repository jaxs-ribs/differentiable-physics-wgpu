#!/usr/bin/env python3
"""Debug how many times impulses are applied."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.engine import TensorPhysicsEngine
from physics.types import create_body_array, ShapeType

# Add counter
collision_count = 0

# Monkey patch
import physics.solver as solver
original_resolve = solver.resolve_collisions

def counting_resolve(bodies, pair_indices, contact_normals, contact_depths, 
                    contact_points, contact_mask, restitution=0.1):
    global collision_count
    num_contacts = contact_mask.sum().numpy()
    if num_contacts > 0:
        collision_count += 1
        print(f"Collision #{collision_count}: {num_contacts} contacts")
    return original_resolve(bodies, pair_indices, contact_normals, contact_depths,
                          contact_points, contact_mask, restitution)

solver.resolve_collisions = counting_resolve

def test_collision_counting():
    """Count collisions."""
    global collision_count
    collision_count = 0
    
    print("\n=== Collision Counting Test ===")
    
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
    
    # Ball approaching ground
    bodies.append(create_body_array(
        position=np.array([0., -0.5, 0.], dtype=np.float32),
        velocity=np.array([0., -5., 0.], dtype=np.float32),
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1.0,
        inertia=np.eye(3, dtype=np.float32) * 0.1,
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([0.5, 0., 0.], dtype=np.float32)
    ))
    
    engine = TensorPhysicsEngine(np.stack(bodies), dt=0.001)
    
    print("Running simulation...")
    
    prev_vy = -5.0
    for step in range(200):
        state = engine.get_state()
        y = state[1, 1]
        vy = state[1, 4]
        
        # Detect bounce
        if prev_vy < 0 and vy > 0:
            print(f"\nBounce detected at step {step}:")
            print(f"  Position: y={y:.3f}")
            print(f"  Velocity: {prev_vy:.3f} -> {vy:.3f}")
            print(f"  Total collisions so far: {collision_count}")
            break
        
        prev_vy = vy
        engine.step()
    
    print(f"\nTotal collision resolutions: {collision_count}")

if __name__ == "__main__":
    test_collision_counting()