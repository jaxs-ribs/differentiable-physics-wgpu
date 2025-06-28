#!/usr/bin/env python3
"""Trace exactly one physics step."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['JIT'] = '0'

from physics.engine import _physics_step_static
from physics.types import create_body_array, ShapeType, BodySchema
from tinygrad import Tensor

# Patch _physics_step_static to add logging
from physics.broadphase_tensor import differentiable_broadphase
from physics.narrowphase import narrowphase
from physics.solver import resolve_collisions
from physics.integration import integrate

def _physics_step_traced(bodies, dt, gravity):
    """Traced version of physics step."""
    print("\n=== PHYSICS STEP START ===")
    print(f"dt = {dt}")
    
    # Print initial state
    ball_y = bodies.numpy()[1, BodySchema.POS_Y]
    ball_vy = bodies.numpy()[1, BodySchema.VEL_Y]
    print(f"Initial: ball y={ball_y:.6f}, vy={ball_vy:.6f}")
    
    # 1. Broadphase
    pair_indices, collision_mask = differentiable_broadphase(bodies)
    print(f"\nBroadphase: {collision_mask.sum().numpy()} potential collisions")
    
    # 2. Narrowphase
    contact_normals, contact_depths, contact_points, contact_mask, pair_indices = narrowphase(
        bodies, pair_indices, collision_mask
    )
    print(f"Narrowphase: {contact_mask.sum().numpy()} actual collisions")
    
    if contact_mask.sum().numpy() > 0:
        idx = 0
        normal = contact_normals.numpy()[idx]
        depth = contact_depths.numpy()[idx]
        print(f"  Contact: normal={normal}, depth={depth:.6f}")
    
    # 3. Collision resolution
    bodies_before_collision = bodies.numpy().copy()
    bodies = resolve_collisions(
        bodies, pair_indices, contact_normals, contact_depths, 
        contact_points, contact_mask, restitution=0.1
    )
    bodies_after_collision = bodies.numpy().copy()
    
    ball_vy_after_collision = bodies_after_collision[1, BodySchema.VEL_Y]
    if contact_mask.sum().numpy() > 0:
        print(f"\nAfter collision: ball vy={ball_vy_after_collision:.6f}")
        print(f"  Change: {ball_vy:.6f} → {ball_vy_after_collision:.6f}")
    
    # 4. Integration
    bodies = integrate(bodies, dt, gravity)
    bodies_after_integration = bodies.numpy()
    
    ball_y_final = bodies_after_integration[1, BodySchema.POS_Y]
    ball_vy_final = bodies_after_integration[1, BodySchema.VEL_Y]
    
    print(f"\nAfter integration:")
    print(f"  Position: {ball_y:.6f} → {ball_y_final:.6f}")
    print(f"  Velocity: {ball_vy_after_collision:.6f} → {ball_vy_final:.6f}")
    
    print("\n=== PHYSICS STEP END ===")
    
    return Tensor(bodies_after_integration)

def test_one_step():
    """Test one step at the critical moment."""
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
    
    # Ball at critical position (just in contact)
    # Contact point should be at y = -1.0
    bodies.append(create_body_array(
        position=np.array([0., -1.002, 0.], dtype=np.float32),
        velocity=np.array([0., -10.85, 0.], dtype=np.float32),
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1.0,
        inertia=np.eye(3, dtype=np.float32) * 0.1,
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([0.5, 0., 0.], dtype=np.float32)
    ))
    
    bodies_t = Tensor(np.stack(bodies))
    gravity = Tensor([0., -9.81, 0.])
    
    print("Setup: Ball at y=-1.002, vy=-10.85")
    print("Expected: collision at y=-1.0, bounce to vy=+1.085")
    
    # Do one step
    bodies_after = _physics_step_traced(bodies_t, 0.001, gravity)

if __name__ == "__main__":
    test_one_step()