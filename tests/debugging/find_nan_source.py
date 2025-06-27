#!/usr/bin/env python3
"""Find the source of NaN values in the simulation."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from physics.engine import TensorPhysicsEngine
from physics.types import BodySchema, ShapeType
from physics.broadphase_tensor import differentiable_broadphase
from physics.narrowphase import narrowphase
from physics.solver import resolve_collisions
from physics.integration import integrate
from tinygrad import Tensor

def trace_nan_source():
    """Trace where NaN first appears."""
    print("Tracing NaN source through simulation steps...")
    
    # Create initial scene
    bodies = np.zeros((2, BodySchema.NUM_PROPERTIES), dtype=np.float32)
    
    # Static ground
    bodies[0, BodySchema.POS_Y] = -5.0
    bodies[0, BodySchema.QUAT_W] = 1.0
    bodies[0, BodySchema.SHAPE_TYPE] = ShapeType.BOX
    bodies[0, BodySchema.SHAPE_PARAM_1:BodySchema.SHAPE_PARAM_3+1] = [10.0, 0.5, 10.0]
    bodies[0, BodySchema.INV_MASS] = 0.0
    
    # Falling sphere
    bodies[1, BodySchema.POS_X:BodySchema.POS_Z+1] = [0.0, 5.0, 0.0]
    bodies[1, BodySchema.QUAT_W] = 1.0
    bodies[1, BodySchema.SHAPE_TYPE] = ShapeType.SPHERE
    bodies[1, BodySchema.SHAPE_PARAM_1] = 1.0
    bodies[1, BodySchema.INV_MASS] = 1.0
    bodies[1, BodySchema.INV_INERTIA_XX] = 2.5
    bodies[1, BodySchema.INV_INERTIA_YY] = 2.5
    bodies[1, BodySchema.INV_INERTIA_ZZ] = 2.5
    
    bodies_tensor = Tensor(bodies)
    gravity = Tensor([0.0, -9.81, 0.0])
    dt = 0.016
    
    # Run simulation steps manually
    for step in range(4):
        print(f"\n--- Step {step} ---")
        
        # Check current state
        pos = bodies_tensor[1, BodySchema.POS_X:BodySchema.POS_Z+1].numpy()
        vel = bodies_tensor[1, BodySchema.VEL_X:BodySchema.VEL_Z+1].numpy()
        print(f"State: pos={pos}, vel={vel}")
        
        if np.any(np.isnan(pos)) or np.any(np.isnan(vel)):
            print("!!! NaN already present in state !!!")
            break
        
        # 1. Broadphase
        pair_indices, collision_mask = differentiable_broadphase(bodies_tensor)
        n_collisions = collision_mask.sum().numpy()
        print(f"Broadphase: {n_collisions} potential collisions")
        
        # 2. Narrowphase
        contact_normals, contact_depths, contact_points, contact_mask, pair_indices_out = narrowphase(
            bodies_tensor, pair_indices, collision_mask
        )
        n_contacts = contact_mask.sum().numpy()
        print(f"Narrowphase: {n_contacts} contacts")
        
        if n_contacts > 0:
            print("Contact details:")
            # Print first contact
            print(f"  Normal: {contact_normals[0].numpy()}")
            print(f"  Depth: {contact_depths[0].numpy()}")
            print(f"  Point: {contact_points[0].numpy()}")
        
        # 3. Collision resolution
        bodies_after_collision = resolve_collisions(
            bodies_tensor, pair_indices_out, contact_normals, contact_depths, 
            contact_points, contact_mask, restitution=0.1
        )
        
        pos_after_col = bodies_after_collision[1, BodySchema.POS_X:BodySchema.POS_Z+1].numpy()
        vel_after_col = bodies_after_collision[1, BodySchema.VEL_X:BodySchema.VEL_Z+1].numpy()
        print(f"After collision: pos={pos_after_col}, vel={vel_after_col}")
        
        if np.any(np.isnan(pos_after_col)) or np.any(np.isnan(vel_after_col)):
            print("!!! NaN appeared in collision resolution !!!")
            
            # Debug collision inputs
            print("\nDebugging collision inputs:")
            print(f"Number of contacts: {n_contacts}")
            if n_contacts > 0:
                # Check body properties
                indices_a = pair_indices_out[:, 0].numpy()
                indices_b = pair_indices_out[:, 1].numpy()
                print(f"Pair indices: {indices_a}, {indices_b}")
                
                # Check masses
                inv_mass_0 = bodies_tensor[0, BodySchema.INV_MASS].numpy()
                inv_mass_1 = bodies_tensor[1, BodySchema.INV_MASS].numpy()
                print(f"Inverse masses: {inv_mass_0}, {inv_mass_1}")
                
                # Check if division by zero in solver
                if inv_mass_0 + inv_mass_1 == 0:
                    print("!!! Both bodies are static (sum of inv_mass = 0) !!!")
            break
        
        # 4. Integration
        bodies_tensor = integrate(bodies_after_collision, dt, gravity)
        
        pos_after_int = bodies_tensor[1, BodySchema.POS_X:BodySchema.POS_Z+1].numpy()
        vel_after_int = bodies_tensor[1, BodySchema.VEL_X:BodySchema.VEL_Z+1].numpy()
        print(f"After integration: pos={pos_after_int}, vel={vel_after_int}")
        
        if np.all(pos_after_int == 1.0):
            print("!!! Position corruption to all 1s detected !!!")
            print("This happens when NaN is passed through Tensor.where()")
            break

if __name__ == "__main__":
    trace_nan_source()