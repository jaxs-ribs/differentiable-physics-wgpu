#!/usr/bin/env python3
"""Trace a single step in detail."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from physics.types import BodySchema, ShapeType
from physics.integration import integrate
from physics.broadphase_tensor import differentiable_broadphase
from physics.narrowphase import narrowphase
from physics.solver import resolve_collisions
from tinygrad import Tensor

def create_problematic_state():
    """Create the exact state from step 2."""
    bodies = np.zeros((2, BodySchema.NUM_PROPERTIES), dtype=np.float32)
    
    # Static ground
    bodies[0, BodySchema.POS_Y] = -5.0
    bodies[0, BodySchema.QUAT_W] = 1.0
    bodies[0, BodySchema.SHAPE_TYPE] = ShapeType.BOX
    bodies[0, BodySchema.SHAPE_PARAM_1:BodySchema.SHAPE_PARAM_3+1] = [10.0, 0.5, 10.0]
    bodies[0, BodySchema.INV_MASS] = 0.0
    
    # Sphere at problematic state
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

def trace_step():
    """Trace through the physics step."""
    bodies = create_problematic_state()
    bodies_tensor = Tensor(bodies)
    gravity = Tensor([0.0, -9.81, 0.0])
    dt = 0.016
    
    print("Initial state:")
    print(f"Body 0: pos={bodies[0, BodySchema.POS_X:BodySchema.POS_Z+1]}, shape=BOX, size=[10, 0.5, 10]")
    print(f"Body 1: pos={bodies[1, BodySchema.POS_X:BodySchema.POS_Z+1]}, vel={bodies[1, BodySchema.VEL_X:BodySchema.VEL_Z+1]}, shape=SPHERE, radius=1.0")
    
    # 1. Broadphase
    print("\n1. BROADPHASE")
    pair_indices, collision_mask = differentiable_broadphase(bodies_tensor)
    print(f"Pairs: {pair_indices.shape[0]}")
    print(f"Collision mask sum: {collision_mask.sum().numpy()}")
    if collision_mask.sum().numpy() > 0:
        print(f"Colliding pairs: {pair_indices[collision_mask].numpy()}")
    
    # 2. Narrowphase
    print("\n2. NARROWPHASE")
    contact_normals, contact_depths, contact_points, contact_mask, pair_indices_out = narrowphase(
        bodies_tensor, pair_indices, collision_mask
    )
    print(f"Contact mask sum: {contact_mask.sum().numpy()}")
    if contact_mask.sum().numpy() > 0:
        print(f"Contact normal: {contact_normals[contact_mask].numpy()}")
        print(f"Contact depth: {contact_depths[contact_mask].numpy()}")
        print(f"Contact point: {contact_points[contact_mask].numpy()}")
    
    # 3. Collision resolution
    print("\n3. COLLISION RESOLUTION")
    bodies_after_collision = resolve_collisions(
        bodies_tensor, pair_indices_out, contact_normals, contact_depths, 
        contact_points, contact_mask, restitution=0.1
    )
    
    pos_after_col = bodies_after_collision[1, BodySchema.POS_X:BodySchema.POS_Z+1].numpy()
    vel_after_col = bodies_after_collision[1, BodySchema.VEL_X:BodySchema.VEL_Z+1].numpy()
    print(f"After collision: pos={pos_after_col}, vel={vel_after_col}")
    
    if np.any(np.isnan(pos_after_col)):
        print("!!! NaN detected after collision !!!")
        # Print all values
        print("\nBody 1 full state:")
        for i in range(27):
            print(f"  Property {i}: {bodies_after_collision[1, i].numpy()}")
        return
    
    # 4. Integration
    print("\n4. INTEGRATION")
    bodies_final = integrate(bodies_after_collision, dt, gravity)
    
    pos_final = bodies_final[1, BodySchema.POS_X:BodySchema.POS_Z+1].numpy()
    vel_final = bodies_final[1, BodySchema.VEL_X:BodySchema.VEL_Z+1].numpy()
    print(f"After integration: pos={pos_final}, vel={vel_final}")
    
    if np.all(pos_final == 1.0):
        print("\n!!! Position became all 1s !!!")
        # Check what happened
        print("Checking integration computation...")
        
        # Get components
        pos_before = bodies_after_collision[:, BodySchema.POS_X:BodySchema.POS_Z+1]
        vel_before = bodies_after_collision[:, BodySchema.VEL_X:BodySchema.VEL_Z+1]
        inv_mass = bodies_after_collision[:, BodySchema.INV_MASS:BodySchema.INV_MASS+1]
        
        print(f"pos_before: {pos_before.numpy()}")
        print(f"vel_before: {vel_before.numpy()}")
        print(f"inv_mass: {inv_mass.numpy()}")
        
        # Check if NaN propagated
        if np.any(np.isnan(pos_before.numpy())) or np.any(np.isnan(vel_before.numpy())):
            print("NaN values found in input to integration!")

if __name__ == "__main__":
    trace_step()