#!/usr/bin/env python3
"""Check if we get multiple contacts for same collision."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['JIT'] = '0'

from physics.types import create_body_array, ShapeType
from physics.broadphase_tensor import differentiable_broadphase
from physics.narrowphase import narrowphase
from physics.solver import resolve_collisions
from tinygrad import Tensor

def test_double_contact():
    """Check for duplicate contacts."""
    bodies = []
    
    # Ground box at y=-2
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
    
    # Sphere in contact
    bodies.append(create_body_array(
        position=np.array([0., -1.005, 0.], dtype=np.float32),
        velocity=np.array([0., -10.85, 0.], dtype=np.float32),
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1.0,
        inertia=np.eye(3, dtype=np.float32) * 0.1,
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([0.5, 0., 0.], dtype=np.float32)
    ))
    
    bodies_t = Tensor(np.stack(bodies))
    
    # Get contacts
    pair_indices, collision_mask = differentiable_broadphase(bodies_t)
    print(f"Broadphase pairs: {pair_indices.numpy()}")
    print(f"Broadphase mask: {collision_mask.numpy()}")
    
    contact_normals, contact_depths, contact_points, contact_mask, pair_indices_out = narrowphase(
        bodies_t, pair_indices, collision_mask
    )
    
    print(f"\nNarrowphase:")
    print(f"Contact mask: {contact_mask.numpy()}")
    print(f"Number of contacts: {contact_mask.sum().numpy()}")
    
    # Print all contacts
    for i in range(len(contact_mask.numpy())):
        if contact_mask.numpy()[i]:
            print(f"\nContact {i}:")
            print(f"  Pair: {pair_indices_out.numpy()[i]}")
            print(f"  Normal: {contact_normals.numpy()[i]}")
            print(f"  Depth: {contact_depths.numpy()[i]}")
            print(f"  Point: {contact_points.numpy()[i]}")
    
    # Now apply collision resolution and see what happens
    print("\n--- Applying collision resolution ---")
    
    # Store initial velocities
    vel_before = bodies_t.numpy()[1, 3:6].copy()
    
    bodies_after = resolve_collisions(
        bodies_t, pair_indices_out, contact_normals, contact_depths,
        contact_points, contact_mask, restitution=0.1
    )
    
    vel_after = bodies_after.numpy()[1, 3:6]
    
    print(f"\nBall velocity:")
    print(f"  Before: {vel_before}")
    print(f"  After: {vel_after}")
    print(f"  Change: {vel_after - vel_before}")
    print(f"  Expected Y change: {11.935}")
    print(f"  Actual Y change: {vel_after[1] - vel_before[1]}")
    print(f"  Ratio: {(vel_after[1] - vel_before[1]) / 11.935:.2f}x")

if __name__ == "__main__":
    test_double_contact()