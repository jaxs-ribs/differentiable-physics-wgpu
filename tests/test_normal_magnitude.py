#!/usr/bin/env python3
"""Check normal magnitude in collision."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['JIT'] = '0'

from physics.engine import _physics_step_static
from physics.types import create_body_array, ShapeType
from physics.broadphase_tensor import differentiable_broadphase
from physics.narrowphase import narrowphase
from tinygrad import Tensor

def test_normal_mag():
    """Check normal magnitude."""
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
    
    # Ball in contact
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
    
    # Get collision
    pair_indices, collision_mask = differentiable_broadphase(bodies_t)
    contact_normals, contact_depths, contact_points, contact_mask, pair_indices = narrowphase(
        bodies_t, pair_indices, collision_mask
    )
    
    if contact_mask.sum().numpy() > 0:
        for i in range(len(contact_mask.numpy())):
            if contact_mask.numpy()[i]:
                normal = contact_normals.numpy()[i]
                magnitude = np.linalg.norm(normal)
                print(f"Contact {i}:")
                print(f"  Normal: {normal}")
                print(f"  Magnitude: {magnitude}")
                if abs(magnitude - 1.0) > 0.001:
                    print(f"  ERROR: Normal not unit length!")

if __name__ == "__main__":
    test_normal_mag()