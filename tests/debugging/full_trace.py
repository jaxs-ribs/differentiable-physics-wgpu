#!/usr/bin/env python3
"""Full trace of simulation to find where NaN/corruption occurs."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from physics.engine import TensorPhysicsEngine
from physics.types import BodySchema, ShapeType
from physics.engine import _physics_step_static
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

def trace_with_engine():
    """Trace using the actual engine."""
    print("Tracing with TensorPhysicsEngine...")
    
    bodies = create_test_scene()
    engine = TensorPhysicsEngine(bodies, dt=0.016)
    
    # Get internal tensor state
    bodies_tensor = engine.bodies
    gravity = engine.gravity
    dt = engine.dt
    
    for i in range(5):
        print(f"\n--- Step {i} ---")
        
        # Get current state
        pos = bodies_tensor[1, BodySchema.POS_X:BodySchema.POS_Z+1].numpy()
        vel = bodies_tensor[1, BodySchema.VEL_X:BodySchema.VEL_Z+1].numpy()
        print(f"Before: pos={pos}, vel={vel}")
        
        # Manually call the static physics step
        bodies_tensor = _physics_step_static(bodies_tensor, dt, gravity)
        
        # Check result
        pos_after = bodies_tensor[1, BodySchema.POS_X:BodySchema.POS_Z+1].numpy()
        vel_after = bodies_tensor[1, BodySchema.VEL_X:BodySchema.VEL_Z+1].numpy()
        print(f"After: pos={pos_after}, vel={vel_after}")
        
        # Check for corruption
        if np.any(np.isnan(pos_after)) or np.any(np.isnan(vel_after)):
            print("!!! NaN detected !!!")
            break
        elif np.all(pos_after == 1.0):
            print("!!! Position corruption to all 1s !!!")
            break
    
    # Update engine state for comparison
    engine.bodies = bodies_tensor

def trace_jit_vs_non_jit():
    """Compare JIT and non-JIT execution."""
    print("\n\nComparing JIT vs non-JIT...")
    
    # Non-JIT
    bodies1 = create_test_scene()
    bodies_tensor1 = Tensor(bodies1)
    gravity = Tensor([0.0, -9.81, 0.0])
    dt = 0.016
    
    print("\nNon-JIT execution:")
    for i in range(3):
        pos = bodies_tensor1[1, BodySchema.POS_X:BodySchema.POS_Z+1].numpy()
        print(f"Step {i}: pos={pos}")
        bodies_tensor1 = _physics_step_static(bodies_tensor1, dt, gravity)
    
    # JIT
    bodies2 = create_test_scene()
    engine = TensorPhysicsEngine(bodies2, dt=0.016)
    
    print("\nJIT execution:")
    for i in range(3):
        state = engine.get_state()
        pos = state[1, BodySchema.POS_X:BodySchema.POS_Z+1]
        print(f"Step {i}: pos={pos}")
        if i < 2:
            engine.step()

if __name__ == "__main__":
    trace_with_engine()
    trace_jit_vs_non_jit()