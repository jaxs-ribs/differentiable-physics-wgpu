#!/usr/bin/env python3
"""Test without JIT to debug."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Disable JIT before importing engine
import os
os.environ['JIT'] = '0'

from physics.engine import TensorPhysicsEngine
from physics.types import create_body_array, ShapeType

# Add debug to solver
import physics.solver as solver
original_resolve = solver.resolve_collisions

def debug_resolve(bodies, pair_indices, contact_normals, contact_depths, 
                 contact_points, contact_mask, restitution=0.1):
    num_contacts = contact_mask.sum().numpy()
    if num_contacts > 0:
        print(f"\n=== RESOLVE COLLISIONS ===")
        print(f"Contacts: {num_contacts}")
        print(f"Restitution: {restitution}")
        
        # Get first contact
        mask_np = contact_mask.numpy()
        if mask_np.any():
            idx = mask_np.nonzero()[0][0]
            pair = pair_indices.numpy()[idx]
            normal = contact_normals.numpy()[idx]
        
            print(f"First contact: bodies {pair[0]} and {pair[1]}")
            print(f"Normal: {normal}")
    
    result = original_resolve(bodies, pair_indices, contact_normals, contact_depths,
                            contact_points, contact_mask, restitution)
    
    if num_contacts > 0:
        # Check velocity change
        old_vy = bodies[1, 4].numpy()
        new_vy = result[1, 4].numpy()
        print(f"Ball velocity: {old_vy:.3f} -> {new_vy:.3f}")
    
    return result

solver.resolve_collisions = debug_resolve

def test_no_jit():
    """Test without JIT."""
    print("\n=== No JIT Test ===")
    
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
    
    # Ball
    bodies.append(create_body_array(
        position=np.array([0., -0.5, 0.], dtype=np.float32),
        velocity=np.array([0., -5., 0.], dtype=np.float32),
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1.0,
        inertia=np.eye(3, dtype=np.float32) * 0.1,
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([0.5, 0., 0.], dtype=np.float32)
    ))
    
    # Create engine without JIT
    engine = TensorPhysicsEngine(np.stack(bodies), dt=0.001)
    
    # Manually call step without JIT
    from physics.integration import integrate
    from physics.broadphase_tensor import differentiable_broadphase
    from physics.narrowphase import narrowphase
    
    print("Stepping manually...")
    
    for step in range(100):
        state = engine.get_state()
        vy = state[1, 4]
        
        if step % 20 == 0:
            print(f"Step {step}: vy={vy:.3f}")
        
        # Manual step
        bodies_t = engine.bodies
        
        # Broadphase
        pair_indices, collision_mask = differentiable_broadphase(bodies_t)
        
        # Narrowphase
        contact_normals, contact_depths, contact_points, contact_mask, pair_indices = narrowphase(
            bodies_t, pair_indices, collision_mask
        )
        
        # Resolve
        bodies_t = solver.resolve_collisions(
            bodies_t, pair_indices, contact_normals, contact_depths,
            contact_points, contact_mask, restitution=0.1
        )
        
        # Integrate
        bodies_t = integrate(bodies_t, engine.dt, engine.gravity)
        
        engine.bodies = bodies_t
        
        if vy > 0:
            print(f"\nBounce at step {step}, vy={vy:.3f}")
            break

if __name__ == "__main__":
    test_no_jit()