#!/usr/bin/env python3
"""Basic bounce test to isolate the issue."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.engine import TensorPhysicsEngine
from physics.types import create_body_array, ShapeType

def test_basic_bounce():
    """Test basic bouncing."""
    print("\n=== Basic Bounce Test ===")
    
    # Create scene
    bodies = []
    
    # Ground at y=0 for simplicity
    bodies.append(create_body_array(
        position=np.array([0., 0., 0.], dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1e8,
        inertia=np.eye(3, dtype=np.float32) * 1e8,
        shape_type=ShapeType.BOX,
        shape_params=np.array([10., 0.5, 10.], dtype=np.float32)  # Top at y=0.5
    ))
    
    # Ball at y=2 (1m above ground top)
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
    
    engine = TensorPhysicsEngine(np.stack(bodies), dt=0.001)
    
    # Calculate expected values
    ground_top = 0.5
    contact_point = ground_top + ball_radius  # y=1.0
    drop_height = initial_height - contact_point  # 1.0m
    
    gravity = 9.81
    time_to_fall = np.sqrt(2 * drop_height / gravity)  # ~0.45s
    impact_velocity = np.sqrt(2 * gravity * drop_height)  # ~4.43 m/s
    bounce_velocity = 0.1 * impact_velocity  # ~0.44 m/s
    
    print(f"Setup:")
    print(f"  Ball starts at y={initial_height}")
    print(f"  Contact point at y={contact_point}")
    print(f"  Drop height: {drop_height}m")
    print(f"  Time to fall: {time_to_fall:.3f}s")
    print(f"  Expected impact velocity: {-impact_velocity:.3f} m/s")
    print(f"  Expected bounce velocity: {bounce_velocity:.3f} m/s")
    
    # Run simulation
    print(f"\nRunning simulation...")
    
    for step in range(1000):  # 1 second
        state = engine.get_state()
        ball_y = state[1, 1]
        ball_vy = state[1, 4]
        
        if step % 100 == 0:
            print(f"  t={step*0.001:.1f}s: y={ball_y:.3f}, vy={ball_vy:.3f}")
        
        # Check for bounce
        if step > 400 and ball_vy > 0:
            print(f"\nBounce detected at t={step*0.001:.3f}s")
            print(f"  Position: y={ball_y:.3f}")
            print(f"  Velocity: vy={ball_vy:.3f}")
            
            # Run a bit more to find max height
            max_y = ball_y
            for _ in range(200):
                engine.step()
                state = engine.get_state()
                if state[1, 1] > max_y:
                    max_y = state[1, 1]
            
            bounce_height = max_y - contact_point
            print(f"  Max height after bounce: {max_y:.3f}")
            print(f"  Bounce height: {bounce_height:.3f}m")
            print(f"  Expected: {(bounce_velocity**2)/(2*gravity):.3f}m")
            break
        
        engine.step()

if __name__ == "__main__":
    test_basic_bounce()