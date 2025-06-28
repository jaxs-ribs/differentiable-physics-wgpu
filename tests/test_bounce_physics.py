#!/usr/bin/env python3
"""Test correct bouncing behavior with restitution."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.engine import TensorPhysicsEngine
from physics.types import create_body_array, ShapeType

def test_bounce_restitution():
    """Test that a ball dropped from height h bounces to approximately e^2 * h."""
    print("\n=== Testing Bounce Physics ===")
    
    # Create a simple scene: ground and falling ball
    bodies = []
    
    # Static ground box at y=-2
    bodies.append(create_body_array(
        position=np.array([0., -2., 0.], dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1e8,  # Very large mass = static
        inertia=np.eye(3, dtype=np.float32) * 1e8,
        shape_type=ShapeType.BOX,
        shape_params=np.array([10., 0.5, 10.], dtype=np.float32)
    ))
    
    # Falling sphere starting at y=5
    initial_height = 5.0
    ball_radius = 0.5
    bodies.append(create_body_array(
        position=np.array([0., initial_height, 0.], dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1.0,
        inertia=np.eye(3, dtype=np.float32) * (2.0/5.0) * 1.0 * (ball_radius**2),
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([ball_radius, 0., 0.], dtype=np.float32)
    ))
    
    # Create engine with known restitution
    restitution = 0.1  # Very low bounce
    engine = TensorPhysicsEngine(np.stack(bodies), dt=0.001)  # Small timestep for accuracy
    
    # Debug: Check which body is which
    print(f"\nBody 0 (ground): shape_type={bodies[0][23]}, y={bodies[0][1]}")
    print(f"Body 1 (ball): shape_type={bodies[1][23]}, y={bodies[1][1]}")
    
    # Calculate expected physics
    gravity = 9.81
    fall_distance = initial_height - (-2.0 + 0.5 + ball_radius)  # Account for ground height and radii
    impact_velocity = np.sqrt(2 * gravity * fall_distance)
    bounce_velocity = restitution * impact_velocity
    expected_bounce_height = (bounce_velocity**2) / (2 * gravity)
    
    print(f"Initial ball height: {initial_height}m")
    print(f"Fall distance: {fall_distance:.3f}m")
    print(f"Expected impact velocity: {impact_velocity:.3f} m/s")
    print(f"Expected bounce velocity: {bounce_velocity:.3f} m/s")
    print(f"Expected bounce height: {expected_bounce_height:.3f}m")
    
    # Run simulation until ball bounces and reaches peak
    max_height_after_bounce = -float('inf')
    has_bounced = False
    previous_y = initial_height
    going_up = False
    
    for step in range(10000):  # 10 seconds at 0.001s timestep
        engine.step()
        state = engine.get_state()
        ball_y = state[1, 0]  # Y position of ball
        ball_vy = state[1, 4]  # Y velocity of ball
        
        # Detect bounce (velocity changes from negative to positive)
        if not has_bounced and ball_vy > 0 and previous_y < initial_height - 1.0:
            has_bounced = True
            going_up = True
            print(f"\nBounce detected at step {step}, y={ball_y:.3f}, vy={ball_vy:.3f}")
        
        # Track maximum height after bounce
        if has_bounced and going_up:
            if ball_y > max_height_after_bounce:
                max_height_after_bounce = ball_y
            # Detect when ball starts falling again
            if ball_vy < 0:
                going_up = False
                break
        
        previous_y = ball_y
        
        if step % 1000 == 0:
            print(f"Step {step}: ball at y={ball_y:.3f}, vy={ball_vy:.3f}")
    
    # Calculate actual bounce height
    ground_top = -2.0 + 0.5  # Ground position + half thickness
    contact_point = ground_top + ball_radius
    actual_bounce_height = max_height_after_bounce - contact_point
    
    print(f"\nActual max height after bounce: {max_height_after_bounce:.3f}m")
    print(f"Actual bounce height: {actual_bounce_height:.3f}m")
    print(f"Expected bounce height: {expected_bounce_height:.3f}m")
    print(f"Error: {abs(actual_bounce_height - expected_bounce_height):.3f}m")
    
    # Check if bounce height is reasonable (within 50% of expected)
    tolerance = 0.5
    if abs(actual_bounce_height - expected_bounce_height) > expected_bounce_height * tolerance:
        print("❌ FAIL: Bounce height is not within tolerance!")
        print(f"   Expected: {expected_bounce_height:.3f}m ± {expected_bounce_height * tolerance:.3f}m")
        print(f"   Actual: {actual_bounce_height:.3f}m")
        return False
    else:
        print("✅ PASS: Bounce height is within tolerance!")
        return True

if __name__ == "__main__":
    success = test_bounce_restitution()
    sys.exit(0 if success else 1)