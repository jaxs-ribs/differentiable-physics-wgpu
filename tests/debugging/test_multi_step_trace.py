#!/usr/bin/env python3
"""Trace multiple physics steps around collision."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['JIT'] = '0'

from physics.engine import TensorPhysicsEngine
from physics.types import create_body_array, ShapeType, BodySchema

def test_multi_step():
    """Test multiple steps around collision time."""
    bodies = []
    
    # Ground at y=-2
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
    
    # Ball falling from specific height to get -10.85 m/s at contact
    # v² = u² + 2as, so u² = v² - 2as
    # At contact (y=-1.0), we want v=-10.85
    # So initial height: h = -1.0 + v²/(2g) = -1.0 + 10.85²/(2*9.81) ≈ 5.0
    bodies.append(create_body_array(
        position=np.array([0., 5.0, 0.], dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1.0,
        inertia=np.eye(3, dtype=np.float32) * 0.1,
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([0.5, 0., 0.], dtype=np.float32)
    ))
    
    engine = TensorPhysicsEngine(np.stack(bodies), dt=0.001)
    
    print("Simulating fall and bounce...")
    print("Contact expected at y=-1.0")
    
    bounce_step = None
    velocities = []
    
    for step in range(1200):
        state = engine.get_state()
        y = state[1, BodySchema.POS_Y]
        vy = state[1, BodySchema.VEL_Y]
        
        # Store velocities around contact
        if y < 0:
            velocities.append((step, y, vy))
        
        # Detect bounce
        if len(velocities) > 1 and velocities[-2][2] < 0 and velocities[-1][2] > 0:
            bounce_step = step
            print(f"\nBounce detected between steps {step-1} and {step}")
            
            # Print several steps before and after
            print("\nVelocity history:")
            for i in range(max(0, len(velocities)-10), min(len(velocities), len(velocities)+5)):
                if i < len(velocities):
                    s, pos, vel = velocities[i]
                    marker = " <-- BOUNCE" if s == bounce_step else ""
                    print(f"  Step {s}: y={pos:.6f}, vy={vel:.6f}{marker}")
            
            break
        
        if step > 1110 and step < 1120:
            print(f"Step {step}: y={y:.6f}, vy={vy:.6f}")
        
        engine.step()
    
    if bounce_step:
        # Calculate velocity ratio
        vel_before = None
        vel_after = None
        for s, pos, vel in velocities:
            if s == bounce_step - 1:
                vel_before = vel
            if s == bounce_step:
                vel_after = vel
        
        if vel_before and vel_after:
            ratio = abs(vel_after / vel_before)
            print(f"\nVelocity ratio: {vel_before:.3f} → {vel_after:.3f} = {ratio:.3f}")
            print(f"Expected ratio: 0.1")
            print(f"ERROR FACTOR: {ratio / 0.1:.2f}x")

if __name__ == "__main__":
    test_multi_step()