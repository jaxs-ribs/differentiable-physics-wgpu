#!/usr/bin/env python3
"""Test if multiple collisions are happening in one step."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['JIT'] = '0'

from physics.engine import TensorPhysicsEngine, _physics_step_static
from physics.types import create_body_array, ShapeType, BodySchema
from physics.broadphase_tensor import differentiable_broadphase
from physics.narrowphase import narrowphase
from physics.solver import resolve_collisions
from physics.integration import integrate
from tinygrad import Tensor

# Patch _physics_step_static to count collisions
_original_step = _physics_step_static

def _physics_step_debug(bodies, dt, gravity):
    """Debug version that prints collision info."""
    # 1. Broadphase
    pair_indices, collision_mask = differentiable_broadphase(bodies)
    
    # 2. Narrowphase
    contact_normals, contact_depths, contact_points, contact_mask, pair_indices = narrowphase(
        bodies, pair_indices, collision_mask
    )
    
    # Check for collisions
    num_contacts = contact_mask.sum().numpy()
    if num_contacts > 0:
        print(f"\nStep has {num_contacts} contacts!")
        for i in range(len(contact_mask.numpy())):
            if contact_mask.numpy()[i]:
                pair = pair_indices.numpy()[i]
                depth = contact_depths.numpy()[i]
                normal = contact_normals.numpy()[i]
                print(f"  Contact {i}: bodies ({pair[0]},{pair[1]}), depth={depth:.3f}, normal={normal}")
                
                # Check velocities
                bodies_np = bodies.numpy()
                vel_a = bodies_np[pair[0], BodySchema.VEL_Y]
                vel_b = bodies_np[pair[1], BodySchema.VEL_Y]
                print(f"    Velocities: A={vel_a:.3f}, B={vel_b:.3f}")
    
    # Continue with normal processing
    bodies_before = bodies.numpy().copy()
    bodies = resolve_collisions(
        bodies, pair_indices, contact_normals, contact_depths, 
        contact_points, contact_mask, restitution=0.1
    )
    
    # Check velocity after collision
    if num_contacts > 0:
        bodies_after = bodies.numpy()
        for i in range(len(contact_mask.numpy())):
            if contact_mask.numpy()[i]:
                pair = pair_indices.numpy()[i]
                vel_a_before = bodies_before[pair[0], BodySchema.VEL_Y]
                vel_b_before = bodies_before[pair[1], BodySchema.VEL_Y]
                vel_a_after = bodies_after[pair[0], BodySchema.VEL_Y]
                vel_b_after = bodies_after[pair[1], BodySchema.VEL_Y]
                print(f"  After collision: A: {vel_a_before:.3f}->{vel_a_after:.3f}, B: {vel_b_before:.3f}->{vel_b_after:.3f}")
    
    # Apply gravity and integrate
    bodies = integrate(bodies, dt, gravity)
    
    return bodies

# Monkey patch
import physics.engine
physics.engine._physics_step_static = _physics_step_debug

def test_multiple():
    """Test for multiple collisions."""
    print("\n=== Testing for Multiple Collisions ===")
    
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
    
    # Ball
    bodies.append(create_body_array(
        position=np.array([0., 2.0, 0.], dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1.0,
        inertia=np.eye(3, dtype=np.float32) * 0.1,
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([0.5, 0., 0.], dtype=np.float32)
    ))
    
    engine = TensorPhysicsEngine(np.stack(bodies), dt=0.001)  # Smaller timestep
    
    print("Simulating...")
    for step in range(2000):
        state = engine.get_state()
        y = state[1, BodySchema.POS_Y]
        vy = state[1, BodySchema.VEL_Y]
        
        if step % 200 == 0:
            print(f"Step {step}: y={y:.3f}, vy={vy:.3f}")
        
        engine.step()
        
        # Check for velocity reversal
        state_after = engine.get_state()
        vy_after = state_after[1, BodySchema.VEL_Y]
        
        if vy < 0 and vy_after > 0:
            print(f"\nBounce detected at step {step}!")
            print(f"Velocity changed from {vy:.3f} to {vy_after:.3f}")
            break

if __name__ == "__main__":
    test_multiple()