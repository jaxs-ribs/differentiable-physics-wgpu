#!/usr/bin/env python3
"""Test fix for integration NaN issue."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from physics.engine import TensorPhysicsEngine
from physics.types import BodySchema, ShapeType
from physics.integration import integrate
from tinygrad import Tensor

def test_integration_fix():
    """Test a potential fix for the integration issue."""
    print("Testing integration fix...")
    
    # Create problematic state
    bodies = np.zeros((2, BodySchema.NUM_PROPERTIES), dtype=np.float32)
    
    # Static ground  
    bodies[0, BodySchema.POS_Y] = -5.0
    bodies[0, BodySchema.QUAT_W] = 1.0
    bodies[0, BodySchema.INV_MASS] = 0.0
    
    # Dynamic body with NaN position (simulating corruption)
    bodies[1, BodySchema.POS_X:BodySchema.POS_Z+1] = [float('nan'), float('nan'), float('nan')]
    bodies[1, BodySchema.VEL_X:BodySchema.VEL_Z+1] = [10.0, 20.0, 30.0]
    bodies[1, BodySchema.QUAT_W] = 1.0
    bodies[1, BodySchema.INV_MASS] = 1.0
    
    bodies_tensor = Tensor(bodies)
    gravity = Tensor([0.0, -9.81, 0.0])
    dt = 0.016
    
    print(f"Before: pos={bodies[1, BodySchema.POS_X:BodySchema.POS_Z+1]}")
    
    # Run integration
    bodies_after = integrate(bodies_tensor, dt, gravity)
    pos_after = bodies_after[1, BodySchema.POS_X:BodySchema.POS_Z+1].numpy()
    
    print(f"After: pos={pos_after}")
    
    if np.all(pos_after == 1.0):
        print("!!! BUG CONFIRMED: NaN became all 1s !!!")
        print("\nThis is a TinyGrad bug in Tensor.where() with NaN values.")
        print("When condition is True and value is NaN, it returns 1.0 instead of NaN.")
        
        # Test workaround
        print("\nTesting workaround...")
        
        # Instead of using where, we can use masking differently
        pos = bodies_tensor[:, BodySchema.POS_X:BodySchema.POS_Z+1]
        vel = bodies_tensor[:, BodySchema.VEL_X:BodySchema.VEL_Z+1]
        inv_mass = bodies_tensor[:, BodySchema.INV_MASS:BodySchema.INV_MASS+1]
        
        is_dynamic = (inv_mass > 1e-7).reshape(-1, 1)
        new_vel = vel + gravity * dt * is_dynamic
        new_pos = pos + new_vel * dt * is_dynamic
        
        print(f"Workaround result: {new_pos[1].numpy()}")

if __name__ == "__main__":
    test_integration_fix()