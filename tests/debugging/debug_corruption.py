#!/usr/bin/env python3
"""Debug script to trace the position corruption issue."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from physics.engine import TensorPhysicsEngine, _physics_step_static
from physics.types import BodySchema, ShapeType
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
    
    # Falling sphere
    bodies[1, BodySchema.POS_X:BodySchema.POS_Z+1] = [0.0, 5.0, 0.0]
    bodies[1, BodySchema.QUAT_W] = 1.0
    bodies[1, BodySchema.SHAPE_TYPE] = ShapeType.SPHERE
    bodies[1, BodySchema.SHAPE_PARAM_1] = 1.0
    bodies[1, BodySchema.INV_MASS] = 1.0
    bodies[1, BodySchema.INV_INERTIA_XX] = 2.5
    bodies[1, BodySchema.INV_INERTIA_YY] = 2.5
    bodies[1, BodySchema.INV_INERTIA_ZZ] = 2.5
    
    return bodies

def trace_physics_step(bodies_tensor, dt, gravity, step_num):
    """Trace through a single physics step to find corruption."""
    print(f"\n--- Step {step_num} ---")
    
    # Print initial state
    pos_before = bodies_tensor[1, BodySchema.POS_X:BodySchema.POS_Z+1].numpy()
    vel_before = bodies_tensor[1, BodySchema.VEL_X:BodySchema.VEL_Z+1].numpy()
    print(f"Before: pos={pos_before}, vel={vel_before}")
    
    # Manually run through physics step components
    from physics.integration import integrate
    from physics.broadphase_tensor import differentiable_broadphase
    from physics.narrowphase import narrowphase
    from physics.solver import resolve_collisions
    
    # 1. Broad phase
    pair_indices, collision_mask = differentiable_broadphase(bodies_tensor)
    print(f"Broad phase: {pair_indices.shape[0]} pairs, {collision_mask.sum().numpy()} potential collisions")
    
    # 2. Narrow phase
    contact_normals, contact_depths, contact_points, contact_mask, pair_indices_out = narrowphase(
        bodies_tensor, pair_indices, collision_mask
    )
    num_contacts = contact_mask.sum().numpy()
    print(f"Narrow phase: {num_contacts} actual contacts")
    
    # 3. Collision resolution
    bodies_after_collision = resolve_collisions(
        bodies_tensor, pair_indices_out, contact_normals, contact_depths, 
        contact_points, contact_mask, restitution=0.1
    )
    pos_after_collision = bodies_after_collision[1, BodySchema.POS_X:BodySchema.POS_Z+1].numpy()
    vel_after_collision = bodies_after_collision[1, BodySchema.VEL_X:BodySchema.VEL_Z+1].numpy()
    print(f"After collision: pos={pos_after_collision}, vel={vel_after_collision}")
    
    # 4. Integration (motion)
    bodies_integrated = integrate(bodies_after_collision, dt, gravity)
    pos_after_int = bodies_integrated[1, BodySchema.POS_X:BodySchema.POS_Z+1].numpy()
    vel_after_int = bodies_integrated[1, BodySchema.VEL_X:BodySchema.VEL_Z+1].numpy()
    print(f"After integration: pos={pos_after_int}, vel={vel_after_int}")
    
    # Check for corruption
    if np.all(pos_after_int == 1.0):
        print("!!! CORRUPTION DETECTED !!!")
        
        # Debug where corruption happened
        if np.all(pos_after_collision == 1.0):
            print("Corruption happened in collision resolution")
        else:
            print("Corruption happened in integration")
        
        # Check body properties that might be corrupted
        for prop_name in ['INV_MASS', 'SHAPE_TYPE', 'SHAPE_PARAM_1']:
            prop_idx = getattr(BodySchema, prop_name)
            val = bodies_integrated[1, prop_idx].numpy()
            print(f"Body 1 {prop_name}: {val}")
    
    return bodies_integrated

def main():
    """Debug the position corruption."""
    print("Debugging Position Corruption")
    print("=" * 60)
    
    bodies = create_test_scene()
    bodies_tensor = Tensor(bodies)
    gravity = Tensor([0.0, -9.81, 0.0])
    dt = 0.016
    
    # Run 5 steps with detailed tracing
    current_bodies = bodies_tensor
    for i in range(5):
        current_bodies = trace_physics_step(current_bodies, dt, gravity, i)
        
        # Stop if corruption detected
        pos = current_bodies[1, BodySchema.POS_X:BodySchema.POS_Z+1].numpy()
        if np.all(pos == 1.0):
            print("\nStopping due to corruption")
            break

if __name__ == "__main__":
    main()