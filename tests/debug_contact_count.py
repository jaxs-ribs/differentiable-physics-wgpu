#!/usr/bin/env python3
"""Debug contact count per collision."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['JIT'] = '0'  # Disable JIT

from physics.engine import _physics_step_static
from physics.types import create_body_array, ShapeType
from tinygrad import Tensor

def test_contact_count():
    """Test contact count."""
    print("\n=== Contact Count Debug ===")
    
    bodies = []
    
    # Ground
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
    
    # Ball close to collision
    bodies.append(create_body_array(
        position=np.array([0., 1.05, 0.], dtype=np.float32),
        velocity=np.array([0., -4.0, 0.], dtype=np.float32),
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1.0,
        inertia=np.eye(3, dtype=np.float32) * 0.1,
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([0.5, 0., 0.], dtype=np.float32)
    ))
    
    bodies_t = Tensor(np.stack(bodies))
    gravity = Tensor([0., -9.81, 0.])
    
    print("Initial state:")
    print(f"  Ball: y={bodies_t[1,1].numpy():.3f}, vy={bodies_t[1,4].numpy():.3f}")
    
    # Manually do broadphase and narrowphase
    from physics.broadphase_tensor import differentiable_broadphase
    from physics.narrowphase import narrowphase
    
    # Step 1: Broadphase
    pair_indices, collision_mask = differentiable_broadphase(bodies_t)
    print(f"\nBroadphase:")
    print(f"  Pairs shape: {pair_indices.shape}")
    print(f"  Active pairs: {collision_mask.sum().numpy()}")
    
    # Step 2: Narrowphase
    contact_normals, contact_depths, contact_points, contact_mask, pair_indices_out = narrowphase(
        bodies_t, pair_indices, collision_mask
    )
    
    print(f"\nNarrowphase:")
    print(f"  Output shape: {contact_mask.shape}")
    print(f"  Active contacts: {contact_mask.sum().numpy()}")
    
    # Print all contacts
    mask_np = contact_mask.numpy()
    normals_np = contact_normals.numpy()
    pairs_np = pair_indices_out.numpy()
    
    print(f"\nAll contacts:")
    for i in range(len(mask_np)):
        if mask_np[i]:
            print(f"  Contact {i}: pair ({pairs_np[i,0]}, {pairs_np[i,1]}), normal={normals_np[i]}")
    
    # Now step and see what changes
    print(f"\n--- Running one physics step ---")
    bodies_after = _physics_step_static(bodies_t, 0.01, gravity)
    
    print(f"After step:")
    print(f"  Ball: y={bodies_after[1,1].numpy():.3f}, vy={bodies_after[1,4].numpy():.3f}")
    
    # Calculate expected
    print(f"\nExpected:")
    print(f"  vy after collision: {0.1 * 4.0:.3f}")
    print(f"  vy after gravity: {0.4 - 9.81*0.01:.3f}")

if __name__ == "__main__":
    test_contact_count()