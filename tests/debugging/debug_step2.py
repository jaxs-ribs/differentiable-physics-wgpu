#!/usr/bin/env python3
"""Debug script to reproduce exact step 2 state where NaN occurs."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from physics.engine import TensorPhysicsEngine
from physics.types import BodySchema, ShapeType
from physics.integration import integrate
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

def test_integration_with_specific_state():
    """Test integration with the specific state that causes NaN."""
    print("Testing integration with problematic state...")
    
    # Create a state similar to what we see at step 2
    bodies = create_test_scene()
    
    # Set the problematic state from step 2
    bodies[1, BodySchema.POS_X:BodySchema.POS_Z+1] = [0.0, 1.0, 0.0]
    bodies[1, BodySchema.VEL_X:BodySchema.VEL_Z+1] = [0.0, 1.0, 0.0]
    
    bodies_tensor = Tensor(bodies)
    gravity = Tensor([0.0, -9.81, 0.0])
    dt = 0.016
    
    print(f"Before integration:")
    print(f"  pos: {bodies[1, BodySchema.POS_X:BodySchema.POS_Z+1]}")
    print(f"  vel: {bodies[1, BodySchema.VEL_X:BodySchema.VEL_Z+1]}")
    
    # Test integration
    bodies_after = integrate(bodies_tensor, dt, gravity)
    
    pos_after = bodies_after[1, BodySchema.POS_X:BodySchema.POS_Z+1].numpy()
    vel_after = bodies_after[1, BodySchema.VEL_X:BodySchema.VEL_Z+1].numpy()
    
    print(f"\nAfter integration:")
    print(f"  pos: {pos_after}")
    print(f"  vel: {vel_after}")
    
    if np.any(pos_after == 1.0) and np.all(pos_after == pos_after[0]):
        print("\n!!! Position became all 1s !!!")
    
    # Let's check the integration function more carefully
    print("\nChecking intermediate values in integration:")
    
    # Extract values
    pos = bodies_tensor[:, BodySchema.POS_X:BodySchema.POS_Z+1]
    vel = bodies_tensor[:, BodySchema.VEL_X:BodySchema.VEL_Z+1]
    inv_mass = bodies_tensor[:, BodySchema.INV_MASS:BodySchema.INV_MASS+1]
    
    print(f"inv_mass: {inv_mass.numpy()}")
    
    # Check dynamic mask
    is_dynamic = (inv_mass > 1e-7).reshape(-1, 1)
    print(f"is_dynamic: {is_dynamic.numpy()}")
    
    # Compute new velocity
    new_vel = vel + gravity.reshape(1, 3) * dt * is_dynamic
    print(f"new_vel: {new_vel.numpy()}")
    
    # Compute new position
    new_pos = pos + new_vel * dt
    print(f"new_pos: {new_pos.numpy()}")

def run_full_simulation():
    """Run simulation and print state at each step."""
    print("\nRunning full simulation to step 2...")
    
    bodies = create_test_scene()
    engine = TensorPhysicsEngine(bodies, dt=0.016)
    
    for i in range(3):
        state = engine.get_state()
        pos = state[1, BodySchema.POS_X:BodySchema.POS_Z+1]
        vel = state[1, BodySchema.VEL_X:BodySchema.VEL_Z+1]
        print(f"\nStep {i}: pos={pos}, vel={vel}")
        
        if i < 2:
            engine.step()
    
    # Now set up exact state from step 2
    print("\n" + "="*60)
    print("Setting up exact step 2 state...")
    
    bodies2 = create_test_scene()
    bodies2[1, BodySchema.POS_X:BodySchema.POS_Z+1] = [0.0, 1.0, 0.0]
    bodies2[1, BodySchema.VEL_X:BodySchema.VEL_Z+1] = [0.0, 1.0, 0.0]
    
    engine2 = TensorPhysicsEngine(bodies2, dt=0.016)
    print("Before step:")
    state = engine2.get_state()
    print(f"  pos: {state[1, BodySchema.POS_X:BodySchema.POS_Z+1]}")
    print(f"  vel: {state[1, BodySchema.VEL_X:BodySchema.VEL_Z+1]}")
    
    engine2.step()
    
    print("After step:")
    state = engine2.get_state()
    print(f"  pos: {state[1, BodySchema.POS_X:BodySchema.POS_Z+1]}")
    print(f"  vel: {state[1, BodySchema.VEL_X:BodySchema.VEL_Z+1]}")

if __name__ == "__main__":
    test_integration_with_specific_state()
    run_full_simulation()