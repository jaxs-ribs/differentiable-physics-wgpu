#!/usr/bin/env python3
"""Test normal direction convention."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['JIT'] = '0'

from physics.types import create_body_array, ShapeType, BodySchema
from physics.broadphase_tensor import differentiable_broadphase
from physics.narrowphase import narrowphase
from tinygrad import Tensor

def test_normal_convention():
    """Test what normal direction narrowphase produces."""
    print("\n=== Testing Normal Convention ===")
    
    bodies = []
    
    # Ground at y=0 (body 0)
    bodies.append(create_body_array(
        position=np.array([0., 0., 0.], dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1e8,
        inertia=np.eye(3, dtype=np.float32) * 1e8,
        shape_type=ShapeType.BOX,
        shape_params=np.array([10., 0.5, 10.], dtype=np.float32)
    ))
    
    # Ball touching ground (body 1)
    bodies.append(create_body_array(
        position=np.array([0., 0.51, 0.], dtype=np.float32),
        velocity=np.array([0., -1.0, 0.], dtype=np.float32),
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1.0,
        inertia=np.eye(3, dtype=np.float32) * 0.1,
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([0.5, 0., 0.], dtype=np.float32)
    ))
    
    bodies_t = Tensor(np.stack(bodies))
    
    # Run collision detection
    pair_indices, collision_mask = differentiable_broadphase(bodies_t)
    print(f"\nBroadphase pairs: {pair_indices.numpy()}")
    
    contact_normals, contact_depths, contact_points, contact_mask, pair_indices = narrowphase(
        bodies_t, pair_indices, collision_mask
    )
    
    if contact_mask.sum().numpy() > 0:
        idx = contact_mask.numpy().nonzero()[0][0]
        pair = pair_indices.numpy()[idx]
        normal = contact_normals.numpy()[idx]
        
        print(f"\nContact found:")
        print(f"  Pair: body {pair[0]} (ground) and body {pair[1]} (ball)")
        print(f"  Normal: {normal}")
        print(f"  Normal direction: {'upward' if normal[1] > 0 else 'downward'}")
        
        # According to solver comments, normal should point from B to A
        # If pair is (0, 1), then A=0 (ground), B=1 (ball)
        # So normal should point from ball to ground (downward)
        print(f"\n  Expected: Normal from ball to ground (downward)")
        print(f"  Actual: Normal pointing {'downward ✓' if normal[1] < 0 else 'upward ❌'}")

if __name__ == "__main__":
    test_normal_convention()