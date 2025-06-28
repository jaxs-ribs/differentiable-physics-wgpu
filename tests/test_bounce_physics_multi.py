#!/usr/bin/env python3
"""Test that bounces don't increase in height (conservation of energy)."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.engine import TensorPhysicsEngine
from physics.types import create_body_array, ShapeType

def test_multiple_bounces():
    """Test that consecutive bounces don't gain energy."""
    print("\n=== Testing Multiple Bounce Physics ===")
    
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
    restitution = 0.8  # Higher restitution to see multiple bounces
    engine = TensorPhysicsEngine(np.stack(bodies), dt=0.001)  # Small timestep for accuracy
    
    # Track bounce heights
    bounce_heights = []
    previous_y = initial_height
    previous_vy = 0
    going_down = True
    local_max = initial_height
    
    ground_top = -2.0 + 0.5  # Ground position + half thickness
    contact_point = ground_top + ball_radius
    
    print(f"Initial ball height: {initial_height}m")
    print(f"Contact point: {contact_point:.3f}m")
    print(f"Restitution: {restitution}")
    
    # Run simulation for several bounces
    for step in range(20000):  # 20 seconds at 0.001s timestep
        engine.step()
        state = engine.get_state()
        ball_y = state[1, 1]  # Y position of ball
        ball_vy = state[1, 4]  # Y velocity of ball
        
        # Detect direction changes
        if going_down and ball_vy > 0.1:  # Started going up (bounce)
            going_down = False
            local_max = ball_y
            
        elif not going_down and ball_vy < -0.1:  # Started going down (peak)
            going_down = True
            # Record the bounce height
            bounce_height = local_max - contact_point
            if bounce_height > 0.05:  # Ignore tiny bounces
                bounce_heights.append(bounce_height)
                print(f"Bounce {len(bounce_heights)}: height = {bounce_height:.3f}m at step {step}")
            
            # Check if we have enough bounces
            if len(bounce_heights) >= 5:
                break
        
        # Update tracking
        if not going_down and ball_y > local_max:
            local_max = ball_y
        
        previous_y = ball_y
        previous_vy = ball_vy
        
        if step % 2000 == 0:
            print(f"  Step {step}: ball at y={ball_y:.3f}, vy={ball_vy:.3f}")
    
    # Analyze bounce heights
    print("\n=== Bounce Height Analysis ===")
    print("Bounce heights:", [f"{h:.3f}" for h in bounce_heights])
    
    if len(bounce_heights) < 3:
        print("❌ FAIL: Not enough bounces detected!")
        return False
    
    # Check if bounces are decreasing (allowing small numerical errors)
    increasing_bounces = []
    tolerance = 0.02  # 2cm tolerance for numerical errors
    
    for i in range(1, len(bounce_heights)):
        if bounce_heights[i] > bounce_heights[i-1] + tolerance:
            increasing_bounces.append((i, bounce_heights[i-1], bounce_heights[i]))
    
    # Calculate expected bounce height ratios
    expected_ratio = restitution ** 2
    print(f"\nExpected bounce ratio (e²): {expected_ratio:.3f}")
    
    actual_ratios = []
    for i in range(1, len(bounce_heights)):
        if bounce_heights[i-1] > 0:
            ratio = bounce_heights[i] / bounce_heights[i-1]
            actual_ratios.append(ratio)
            print(f"Bounce {i} to {i+1} ratio: {ratio:.3f}")
    
    if increasing_bounces:
        print("\n❌ FAIL: Ball is gaining energy!")
        for bounce_num, prev_height, curr_height in increasing_bounces:
            print(f"   Bounce {bounce_num}: {prev_height:.3f}m → {curr_height:.3f}m (INCREASED by {curr_height - prev_height:.3f}m)")
        return False
    
    # Check if ratios are reasonable (not wildly different from expected)
    bad_ratios = []
    for i, ratio in enumerate(actual_ratios):
        if ratio > expected_ratio * 1.5:  # 50% higher than expected is bad
            bad_ratios.append((i+1, ratio))
    
    if bad_ratios:
        print("\n❌ FAIL: Bounce ratios are too high!")
        for bounce_num, ratio in bad_ratios:
            print(f"   Bounce {bounce_num} to {bounce_num+1}: ratio = {ratio:.3f} (expected ≤ {expected_ratio * 1.5:.3f})")
        return False
    
    print("\n✅ PASS: Ball bounces are decreasing as expected!")
    return True

if __name__ == "__main__":
    success = test_multiple_bounces()
    sys.exit(0 if success else 1)