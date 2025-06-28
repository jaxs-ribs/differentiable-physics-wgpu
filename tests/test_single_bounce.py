#!/usr/bin/env python3
"""Test a single bounce carefully."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['JIT'] = '0'

from physics.engine import TensorPhysicsEngine
from physics.types import create_body_array, ShapeType, BodySchema

def test_single_bounce():
    """Test one bounce in detail."""
    print("\n=== Single Bounce Test ===")
    
    bodies = []
    
    # Ground at y=0
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
    
    # Ball falling from height
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
    
    engine = TensorPhysicsEngine(np.stack(bodies), dt=0.001)
    
    # Track velocity just before and after bounce
    bounce_detected = False
    vel_before_bounce = None
    vel_after_bounce = None
    
    print(f"Ball starts at y=2.0m")
    print(f"Expected impact velocity: sqrt(2*g*h) = sqrt(2*9.81*1.5) ≈ -5.42 m/s")
    print(f"With e=0.1, expected bounce velocity: +0.542 m/s")
    print(f"Expected max height after bounce: v²/(2g) = 0.542²/(2*9.81) ≈ 0.015m above contact")
    
    prev_vy = 0
    for step in range(3000):
        state = engine.get_state()
        y = state[1, BodySchema.POS_Y]
        vy = state[1, BodySchema.VEL_Y]
        
        if step % 200 == 0:
            print(f"  Step {step}: y={y:.3f}, vy={vy:.3f}")
            
        # Emergency exit if something goes wrong
        if y > 10 or y < -10:
            print(f"ERROR: Ball escaped! y={y}")
            break
        
        # Detect bounce
        if prev_vy < 0 and vy > 0 and not bounce_detected:
            vel_before_bounce = prev_vy
            vel_after_bounce = vy
            bounce_detected = True
            print(f"\nBounce detected at step {step}!")
            print(f"  Velocity: {vel_before_bounce:.3f} → {vel_after_bounce:.3f}")
            print(f"  Ratio: {abs(vel_after_bounce/vel_before_bounce):.3f}")
            print(f"  Expected ratio: 0.1")
        
        prev_vy = vy
        engine.step()
        
        # Track peak after bounce
        if bounce_detected and vy < 0 and prev_vy >= 0:
            print(f"\nPeak reached at y={y:.3f}")
            print(f"  Height above ground: {y - 0.5:.3f}")
            print(f"  Height above contact point (1.0): {y - 1.0:.3f}")
            break

if __name__ == "__main__":
    test_single_bounce()