#!/usr/bin/env python3
"""Simple collision test with manual calculation."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['JIT'] = '0'

from physics.types import create_body_array, ShapeType, BodySchema
from physics.broadphase_tensor import differentiable_broadphase
from physics.narrowphase import narrowphase
from physics.solver import resolve_collisions
from tinygrad import Tensor

def test_simple():
    """Test simple collision."""
    print("\n=== Simple Collision Test ===")
    
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
    
    # Ball just above contact
    bodies.append(create_body_array(
        position=np.array([0., -1.01, 0.], dtype=np.float32),
        velocity=np.array([0., -10.85, 0.], dtype=np.float32),
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1.0,
        inertia=np.eye(3, dtype=np.float32) * 0.1,
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([0.5, 0., 0.], dtype=np.float32)
    ))
    
    bodies_t = Tensor(np.stack(bodies))
    
    print("Initial:")
    print(f"  Ball y: {bodies[1][1]:.3f}")
    print(f"  Ball vy: {bodies[1][4]:.3f}")
    print(f"  Ground top: -1.5")
    print(f"  Contact expected at: -1.0")
    
    # Get collision
    pair_indices, collision_mask = differentiable_broadphase(bodies_t)
    contact_normals, contact_depths, contact_points, contact_mask, pair_indices = narrowphase(
        bodies_t, pair_indices, collision_mask
    )
    
    print(f"\nCollisions found: {contact_mask.sum().numpy()}")
    
    if contact_mask.sum().numpy() > 0:
        idx = 0
        pair = pair_indices.numpy()[idx]
        normal = contact_normals.numpy()[idx]
        depth = contact_depths.numpy()[idx]
        point = contact_points.numpy()[idx]
        
        print(f"\nContact details:")
        print(f"  Pair: {pair}")
        print(f"  Normal: {normal}")
        print(f"  Depth: {depth:.3f}")
        print(f"  Point: {point}")
        
        # Apply collision
        bodies_after = resolve_collisions(
            bodies_t, pair_indices, contact_normals, contact_depths,
            contact_points, contact_mask, restitution=0.1
        )
        
        vel_before = bodies_t.numpy()[1, 3:6]
        vel_after = bodies_after.numpy()[1, 3:6]
        
        print(f"\nVelocity change:")
        print(f"  Before: {vel_before}")
        print(f"  After: {vel_after}")
        print(f"  Expected Y: {-10.85 * 0.1:.3f}")
        print(f"  Actual Y: {vel_after[1]:.3f}")
        print(f"  Factor: {vel_after[1] / (10.85 * 0.1):.2f}x")

if __name__ == "__main__":
    test_simple()