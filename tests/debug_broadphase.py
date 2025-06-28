#!/usr/bin/env python3
"""Debug broadphase collision detection."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.engine import TensorPhysicsEngine
from physics.types import create_body_array, ShapeType
from physics.broadphase_tensor import differentiable_broadphase

def test_broadphase():
    """Test broadphase detection."""
    print("\n=== Broadphase Debug ===")
    
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
    
    # Ball close to ground
    bodies.append(create_body_array(
        position=np.array([0., -1.0, 0.], dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1.0,
        inertia=np.eye(3, dtype=np.float32) * 0.1,
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([0.5, 0., 0.], dtype=np.float32)
    ))
    
    engine = TensorPhysicsEngine(np.stack(bodies), dt=0.001)
    
    # Check broadphase
    pair_indices, collision_mask = differentiable_broadphase(engine.bodies)
    
    print(f"Number of bodies: {len(bodies)}")
    print(f"Pair indices shape: {pair_indices.shape}")
    print(f"Collision mask shape: {collision_mask.shape}")
    
    pairs = pair_indices.numpy()
    mask = collision_mask.numpy()
    
    print(f"\nAll possible pairs:")
    for i, (a, b) in enumerate(pairs):
        print(f"  Pair {i}: bodies {a} and {b}, active: {mask[i]}")
    
    num_active = mask.sum()
    print(f"\nActive collision pairs: {num_active}")
    
    if num_active > 0:
        active_pairs = pairs[mask]
        print("Active pairs:")
        for pair in active_pairs:
            print(f"  Bodies {pair[0]} and {pair[1]}")

if __name__ == "__main__":
    test_broadphase()