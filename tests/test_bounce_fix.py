#!/usr/bin/env python3
"""Test bounce physics after attempted fixes."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.engine import TensorPhysicsEngine
from physics.types import create_body_array, ShapeType

def test_bounce_fix():
    """Test if bounce physics is fixed."""
    print("\n=== Testing Bounce Fix ===")
    
    # Create scene
    bodies = []
    
    # Static ground
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
    
    # Falling ball from known height
    drop_height = 1.0  # 1m above contact
    ball_radius = 0.5
    contact_y = -2.0 + 0.5 + ball_radius
    
    bodies.append(create_body_array(
        position=np.array([0., contact_y + drop_height, 0.], dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1.0,
        inertia=np.eye(3, dtype=np.float32) * (2.0/5.0) * 1.0 * (ball_radius**2),
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([ball_radius, 0., 0.], dtype=np.float32)
    ))
    
    engine = TensorPhysicsEngine(np.stack(bodies), dt=0.001)
    
    # Calculate expected physics
    gravity = 9.81
    restitution = 0.1  # From engine
    impact_velocity = np.sqrt(2 * gravity * drop_height)
    bounce_velocity = restitution * impact_velocity
    expected_bounce_height = (bounce_velocity**2) / (2 * gravity)
    
    print(f"Drop height: {drop_height}m")
    print(f"Expected impact velocity: {-impact_velocity:.3f} m/s")
    print(f"Expected bounce velocity: {bounce_velocity:.3f} m/s")
    print(f"Expected bounce height: {expected_bounce_height:.3f}m")
    
    # Run simulation
    max_height_after_bounce = -float('inf')
    has_bounced = False
    
    for step in range(3000):
        state = engine.get_state()
        ball_y = state[1, 1]
        ball_vy = state[1, 4]
        
        if not has_bounced and ball_vy > 0 and ball_y < contact_y + drop_height/2:
            has_bounced = True
            print(f"\nBounce detected at step {step}:")
            print(f"  Position: {ball_y:.3f}m")
            print(f"  Velocity: {ball_vy:.3f} m/s")
        
        if has_bounced:
            max_height_after_bounce = max(max_height_after_bounce, ball_y)
            if ball_vy < 0:  # Started falling again
                break
        
        engine.step()
    
    actual_bounce_height = max_height_after_bounce - contact_y
    
    print(f"\nResults:")
    print(f"  Actual bounce height: {actual_bounce_height:.3f}m")
    print(f"  Expected bounce height: {expected_bounce_height:.3f}m")
    print(f"  Error: {abs(actual_bounce_height - expected_bounce_height):.3f}m")
    
    # Check if reasonable (within 20% of expected)
    tolerance = 0.2
    relative_error = abs(actual_bounce_height - expected_bounce_height) / expected_bounce_height
    
    if relative_error < tolerance:
        print(f"\n✅ PASS: Bounce height within {tolerance*100}% of expected")
        return True
    else:
        print(f"\n❌ FAIL: Bounce height error is {relative_error*100:.1f}% (> {tolerance*100}%)")
        return False

if __name__ == "__main__":
    success = test_bounce_fix()
    sys.exit(0 if success else 1)