#!/usr/bin/env python3
"""Debug script to investigate solver NaN issue."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from physics.engine import TensorPhysicsEngine
from physics.types import BodySchema, ShapeType
from physics.broadphase_tensor import differentiable_broadphase
from physics.narrowphase import narrowphase
from physics.solver import resolve_collisions
from tinygrad import Tensor

def create_test_scene():
    """Create simple scene with falling sphere."""
    bodies = np.zeros((2, BodySchema.NUM_PROPERTIES), dtype=np.float32)
    
    # Static ground
    bodies[0, BodySchema.POS_Y] = -5.0
    bodies[0, BodySchema.QUAT_W] = 1.0
    bodies[0, BodySchema.SHAPE_TYPE] = ShapeType.BOX
    bodies[0, BodySchema.SHAPE_PARAM_1:BodySchema.SHAPE_PARAM_3+1] = [10.0, 0.5, 10.0]
    bodies[0, BodySchema.INV_MASS] = 0.0
    
    # Sphere at position that will have collision
    bodies[1, BodySchema.POS_X:BodySchema.POS_Z+1] = [0.0, 1.0, 0.0]
    bodies[1, BodySchema.VEL_X:BodySchema.VEL_Z+1] = [0.0, 1.0, 0.0]
    bodies[1, BodySchema.QUAT_W] = 1.0
    bodies[1, BodySchema.SHAPE_TYPE] = ShapeType.SPHERE
    bodies[1, BodySchema.SHAPE_PARAM_1] = 1.0
    bodies[1, BodySchema.INV_MASS] = 1.0
    bodies[1, BodySchema.INV_INERTIA_XX] = 2.5
    bodies[1, BodySchema.INV_INERTIA_YY] = 2.5
    bodies[1, BodySchema.INV_INERTIA_ZZ] = 2.5
    
    return bodies

def debug_solver():
    """Debug the solver at the problematic state."""
    bodies = create_test_scene()
    bodies_tensor = Tensor(bodies)
    
    print("Initial state:")
    print(f"Body 0 pos: {bodies[0, BodySchema.POS_X:BodySchema.POS_Z+1]}")
    print(f"Body 1 pos: {bodies[1, BodySchema.POS_X:BodySchema.POS_Z+1]}")
    print(f"Body 1 vel: {bodies[1, BodySchema.VEL_X:BodySchema.VEL_Z+1]}")
    
    # Run collision detection
    pair_indices, collision_mask = differentiable_broadphase(bodies_tensor)
    print(f"\nBroadphase: {collision_mask.sum().numpy()} potential collisions")
    
    contact_normals, contact_depths, contact_points, contact_mask, pair_indices_out = narrowphase(
        bodies_tensor, pair_indices, collision_mask
    )
    print(f"Narrowphase: {contact_mask.sum().numpy()} actual contacts")
    
    if contact_mask.sum().numpy() > 0:
        print(f"\nContact details:")
        print(f"Pair indices: {pair_indices_out.numpy()}")
        print(f"Contact normals: {contact_normals.numpy()}")
        print(f"Contact depths: {contact_depths.numpy()}")
        print(f"Contact points: {contact_points.numpy()}")
        print(f"Contact mask: {contact_mask.numpy()}")
    
    # Apply collision resolution
    print("\nApplying collision resolution...")
    bodies_after = resolve_collisions(
        bodies_tensor, pair_indices_out, contact_normals, contact_depths, 
        contact_points, contact_mask, restitution=0.1
    )
    
    # Check results
    pos_after = bodies_after[1, BodySchema.POS_X:BodySchema.POS_Z+1].numpy()
    vel_after = bodies_after[1, BodySchema.VEL_X:BodySchema.VEL_Z+1].numpy()
    
    print(f"\nAfter collision resolution:")
    print(f"Body 1 pos: {pos_after}")
    print(f"Body 1 vel: {vel_after}")
    
    # Check for NaN
    if np.any(np.isnan(pos_after)) or np.any(np.isnan(vel_after)):
        print("\n!!! NaN DETECTED !!!")
        
        # Check all body properties
        print("\nAll body 1 properties:")
        for i in range(BodySchema.NUM_PROPERTIES):
            val = bodies_after[1, i].numpy()
            print(f"  Property {i}: {val}")

if __name__ == "__main__":
    debug_solver()