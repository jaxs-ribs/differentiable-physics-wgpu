#!/usr/bin/env python3
"""Simple bounce test with actual restitution."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.engine import TensorPhysicsEngine
from physics.types import create_body_array, ShapeType

def test_simple_bounce():
    """Test a simple bounce with restitution 0.1."""
    print("\n=== Testing Simple Bounce ===")
    
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
    
    # Falling sphere starting at y=2
    initial_height = 2.0
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
    
    # Create engine (uses restitution=0.1)
    engine = TensorPhysicsEngine(np.stack(bodies), dt=0.001)
    
    ground_top = -2.0 + 0.5
    contact_point = ground_top + ball_radius
    
    print(f"Initial height: {initial_height}m")
    print(f"Contact point: {contact_point}m")
    print(f"Drop height: {initial_height - contact_point}m")
    print(f"Restitution: 0.1 (from engine)")
    
    # Track max heights
    max_heights = [initial_height]
    current_max = initial_height
    going_down = True
    
    for step in range(10000):
        engine.step()
        state = engine.get_state()
        ball_y = state[1, 1]  # Y position of ball
        ball_vy = state[1, 4]  # Y velocity of ball
        
        # Track direction changes
        if going_down and ball_vy > 0.1:
            going_down = False
            current_max = ball_y
        elif not going_down and ball_vy < -0.1:
            going_down = True
            if current_max - contact_point > 0.01:  # Significant bounce
                max_heights.append(current_max)
                print(f"Bounce {len(max_heights)-1}: max height = {current_max:.3f}m")
                if len(max_heights) >= 4:
                    break
        
        if not going_down and ball_y > current_max:
            current_max = ball_y
    
    print("\nBounce heights above contact:")
    for i, h in enumerate(max_heights):
        print(f"  {i}: {h - contact_point:.3f}m")
    
    # Check if heights are increasing
    increasing = False
    for i in range(1, len(max_heights)):
        if max_heights[i] > max_heights[i-1] + 0.01:
            increasing = True
            print(f"\n❌ Height increased from {max_heights[i-1]:.3f} to {max_heights[i]:.3f}")
    
    if not increasing:
        print("\n✅ Ball heights are decreasing as expected")
    
    return not increasing

if __name__ == "__main__":
    success = test_simple_bounce()
    sys.exit(0 if success else 1)