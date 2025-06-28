#!/usr/bin/env python3
"""Debug normal vector magnitude."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.engine import TensorPhysicsEngine
from physics.types import create_body_array, ShapeType
from physics.broadphase_tensor import differentiable_broadphase
from physics.narrowphase import narrowphase
from tinygrad import Tensor

def test_normal_magnitude():
    """Check if normals are normalized."""
    print("\n=== Normal Magnitude Debug ===")
    
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
    
    # Ball overlapping
    bodies.append(create_body_array(
        position=np.array([0., -1.2, 0.], dtype=np.float32),
        velocity=np.array([0., -5., 0.], dtype=np.float32),
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1.0,
        inertia=np.eye(3, dtype=np.float32) * 0.1,
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([0.5, 0., 0.], dtype=np.float32)
    ))
    
    # Add more bodies at different positions
    for i in range(3):
        bodies.append(create_body_array(
            position=np.array([i+1, -1.3 + i*0.1, 0.], dtype=np.float32),
            velocity=np.zeros(3, dtype=np.float32),
            orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
            angular_vel=np.zeros(3, dtype=np.float32),
            mass=1.0,
            inertia=np.eye(3, dtype=np.float32) * 0.1,
            shape_type=ShapeType.SPHERE,
            shape_params=np.array([0.3, 0., 0.], dtype=np.float32)
        ))
    
    bodies_t = Tensor(np.stack(bodies))
    
    # Get collisions
    pair_indices, collision_mask = differentiable_broadphase(bodies_t)
    contact_normals, contact_depths, contact_points, contact_mask, pair_indices_out = narrowphase(
        bodies_t, pair_indices, collision_mask
    )
    
    normals = contact_normals.numpy()
    mask = contact_mask.numpy()
    
    print(f"Total contacts: {mask.sum()}")
    print("\nChecking normal magnitudes:")
    
    for i in range(len(normals)):
        if mask[i]:
            magnitude = np.linalg.norm(normals[i])
            print(f"  Contact {i}: normal={normals[i]}, magnitude={magnitude:.6f}")
            if abs(magnitude - 1.0) > 0.001:
                print(f"    WARNING: Normal not normalized!")

if __name__ == "__main__":
    test_normal_magnitude()