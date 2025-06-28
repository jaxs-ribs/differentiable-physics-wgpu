#!/usr/bin/env python3
"""Trace through one timestep in detail."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.engine import TensorPhysicsEngine
from physics.types import create_body_array, ShapeType
from tinygrad import Tensor

# Disable JIT for debugging
os.environ['JIT'] = '0'

def trace_step():
    """Trace one physics step."""
    print("\n=== Tracing One Step ===")
    
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
    
    # Ball just before collision
    bodies.append(create_body_array(
        position=np.array([0., -1.05, 0.], dtype=np.float32),  # Overlapping
        velocity=np.array([0., -6.0, 0.], dtype=np.float32),  # Falling
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1.0,
        inertia=np.eye(3, dtype=np.float32) * 0.1,
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([0.5, 0., 0.], dtype=np.float32)
    ))
    
    engine = TensorPhysicsEngine(np.stack(bodies), dt=0.01)
    
    print("Initial state:")
    state = engine.get_state()
    print(f"  Ball: y={state[1,1]:.3f}, vy={state[1,4]:.3f}")
    
    # Manually step through
    from physics.broadphase_tensor import differentiable_broadphase
    from physics.narrowphase import narrowphase
    from physics.solver import resolve_collisions
    from physics.integration import integrate
    
    bodies_t = engine.bodies
    
    # 1. Broadphase
    pair_indices, collision_mask = differentiable_broadphase(bodies_t)
    print(f"\nBroadphase: {collision_mask.sum().numpy()} potential collisions")
    
    # 2. Narrowphase
    contact_normals, contact_depths, contact_points, contact_mask, pair_indices = narrowphase(
        bodies_t, pair_indices, collision_mask
    )
    num_contacts = contact_mask.sum().numpy()
    print(f"Narrowphase: {num_contacts} actual contacts")
    
    if num_contacts > 0:
        normals = contact_normals.numpy()
        depths = contact_depths.numpy()
        mask = contact_mask.numpy()
        idx = mask.nonzero()[0][0]
        print(f"  Normal: {normals[idx]}")
        print(f"  Depth: {depths[idx]:.4f}")
    
    # 3. Before collision resolution
    print(f"\nBefore collision resolution:")
    print(f"  Ball velocity: {bodies_t[1, 4].numpy():.3f}")
    
    # Collision resolution
    bodies_t = resolve_collisions(
        bodies_t, pair_indices, contact_normals, contact_depths,
        contact_points, contact_mask, restitution=0.1
    )
    
    print(f"After collision resolution:")
    print(f"  Ball velocity: {bodies_t[1, 4].numpy():.3f}")
    
    # 4. Integration
    print(f"\nBefore integration:")
    print(f"  Ball: y={bodies_t[1,1].numpy():.3f}, vy={bodies_t[1,4].numpy():.3f}")
    
    bodies_t = integrate(bodies_t, engine.dt, engine.gravity)
    
    print(f"After integration:")
    print(f"  Ball: y={bodies_t[1,1].numpy():.3f}, vy={bodies_t[1,4].numpy():.3f}")
    
    # Calculate what we expect
    print(f"\nExpected after collision:")
    print(f"  vy should be: {0.1 * 6.0:.3f} (restitution * impact_velocity)")
    print(f"  After gravity: {0.6 - 9.81 * 0.01:.3f}")

if __name__ == "__main__":
    trace_step()