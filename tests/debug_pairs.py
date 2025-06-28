#!/usr/bin/env python3
"""Debug collision pairs."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.engine import TensorPhysicsEngine
from physics.types import create_body_array, ShapeType
from physics.broadphase_tensor import differentiable_broadphase
from physics.narrowphase import narrowphase
from tinygrad import Tensor

def test_pairs():
    """Test collision pair generation."""
    print("\n=== Collision Pairs Debug ===")
    
    bodies = []
    
    # Ground (box)
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
    
    # Ball (sphere) near ground
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
    
    bodies_t = Tensor(np.stack(bodies))
    
    # Broadphase
    print("=== BROADPHASE ===")
    pair_indices, collision_mask = differentiable_broadphase(bodies_t)
    
    pairs_bp = pair_indices.numpy()
    mask_bp = collision_mask.numpy()
    
    print(f"Broadphase pairs: {pairs_bp.shape}")
    for i, (a, b) in enumerate(pairs_bp):
        if mask_bp[i]:
            print(f"  Pair {i}: ({a}, {b}) - ACTIVE")
        else:
            print(f"  Pair {i}: ({a}, {b}) - inactive")
    
    # Narrowphase
    print("\n=== NARROWPHASE ===")
    contact_normals, contact_depths, contact_points, contact_mask, pair_indices_out = narrowphase(
        bodies_t, pair_indices, collision_mask
    )
    
    pairs_np = pair_indices_out.numpy()
    mask_np = contact_mask.numpy()
    normals_np = contact_normals.numpy()
    depths_np = contact_depths.numpy()
    
    print(f"Narrowphase output pairs: {pairs_np.shape}")
    num_contacts = mask_np.sum()
    print(f"Active contacts: {num_contacts}")
    
    for i in range(len(pairs_np)):
        if mask_np[i]:
            print(f"  Contact {i}: bodies ({pairs_np[i,0]}, {pairs_np[i,1]})")
            print(f"    Normal: {normals_np[i]}")
            print(f"    Depth: {depths_np[i]:.4f}")

if __name__ == "__main__":
    test_pairs()