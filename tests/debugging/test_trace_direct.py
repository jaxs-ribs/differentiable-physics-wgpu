#!/usr/bin/env python3
"""Direct test of collision resolution."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['JIT'] = '0'

from physics.engine import _physics_step_static
from physics.types import create_body_array, ShapeType, BodySchema
from physics.solver import resolve_collisions
from tinygrad import Tensor

def test_direct():
    """Test collision resolution directly."""
    print("\n=== Direct Collision Test ===")
    
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
    
    # Ball about to collide
    bodies.append(create_body_array(
        position=np.array([0., 0.52, 0.], dtype=np.float32),
        velocity=np.array([0., -3.0, 0.], dtype=np.float32),
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1.0,
        inertia=np.eye(3, dtype=np.float32) * 0.1,
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([0.5, 0., 0.], dtype=np.float32)
    ))
    
    bodies_t = Tensor(np.stack(bodies))
    
    print("Before:")
    print(f"  Ball pos: {bodies_t[1,BodySchema.POS_Y].numpy():.3f}")
    print(f"  Ball vel: {bodies_t[1,BodySchema.VEL_Y].numpy():.3f}")
    
    # Manually do collision detection
    from physics.broadphase_tensor import differentiable_broadphase
    from physics.narrowphase import narrowphase
    
    # Broadphase
    pair_indices, collision_mask = differentiable_broadphase(bodies_t)
    print(f"\nBroadphase: {collision_mask.sum().numpy()} potential collisions")
    
    # Narrowphase
    contact_normals, contact_depths, contact_points, contact_mask, pair_indices = narrowphase(
        bodies_t, pair_indices, collision_mask
    )
    print(f"Narrowphase: {contact_mask.sum().numpy()} actual collisions")
    
    if contact_mask.sum().numpy() > 0:
        # Print contact info
        idx = contact_mask.numpy().nonzero()[0][0]
        print(f"\nContact {idx}:")
        print(f"  Pair: {pair_indices.numpy()[idx]}")
        print(f"  Normal: {contact_normals.numpy()[idx]}")
        print(f"  Depth: {contact_depths.numpy()[idx]:.3f}")
        print(f"  Point: {contact_points.numpy()[idx]}")
        
        # Get velocities before
        vel_before = bodies_t[1,BodySchema.VEL_Y].numpy()
        
        # Apply collision resolution
        bodies_after = resolve_collisions(
            bodies_t, pair_indices, contact_normals, contact_depths,
            contact_points, contact_mask, restitution=0.1
        )
        
        vel_after = bodies_after[1,BodySchema.VEL_Y].numpy()
        
        print(f"\nVelocity change:")
        print(f"  Before: {vel_before:.3f}")
        print(f"  After: {vel_after:.3f}")
        print(f"  Expected (e=0.1): {-vel_before * 0.1:.3f}")
        print(f"  Ratio: {vel_after / (-vel_before * 0.1):.2f}x")

if __name__ == "__main__":
    test_direct()