#!/usr/bin/env python3
"""Test penetration depth during collision."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['JIT'] = '0'

from physics.engine import _physics_step_static
from physics.types import create_body_array, ShapeType, BodySchema
from physics.broadphase_tensor import differentiable_broadphase
from physics.narrowphase import narrowphase
from tinygrad import Tensor

def test_penetration():
    """Test penetration during fast collision."""
    print("\n=== Testing Penetration Depth ===")
    
    bodies = []
    
    # Ground at y=0
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
    
    # Ball with high velocity
    bodies.append(create_body_array(
        position=np.array([0., 1.1, 0.], dtype=np.float32),
        velocity=np.array([0., -10.0, 0.], dtype=np.float32),  # Very fast
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
    
    # Step with different timesteps
    for dt in [0.1, 0.01, 0.001]:
        print(f"\nTimestep dt={dt}:")
        
        # Calculate where ball will be after integration
        y_after_gravity = bodies[1][1] + bodies[1][4] * dt - 0.5 * 9.81 * dt**2
        print(f"  Ball position after dt: {y_after_gravity:.3f}")
        
        # Check collision
        test_bodies = bodies_t.clone()
        pair_indices, collision_mask = differentiable_broadphase(test_bodies)
        contact_normals, contact_depths, contact_points, contact_mask, _ = narrowphase(
            test_bodies, pair_indices, collision_mask
        )
        
        if contact_mask.sum().numpy() > 0:
            idx = contact_mask.numpy().nonzero()[0][0]
            depth = contact_depths.numpy()[idx]
            print(f"  Penetration depth: {depth:.3f}")
            
            # Do one physics step
            bodies_after = _physics_step_static(test_bodies, dt, gravity)
            vy_after = bodies_after[1,4].numpy()
            print(f"  Velocity after step: {vy_after:.3f}")
            print(f"  Expected with e=0.1: {10.0 * 0.1:.3f}")
            print(f"  Ratio: {vy_after / (10.0 * 0.1):.2f}x")

if __name__ == "__main__":
    test_penetration()