#!/usr/bin/env python3
"""Debug bounce physics with a falling ball."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.engine import TensorPhysicsEngine
from physics.types import create_body_array, ShapeType

def debug_falling_bounce():
    """Debug a ball falling and bouncing."""
    print("\n=== Debugging Falling Ball Bounce ===")
    
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
    
    # Create engine
    restitution = 0.8
    engine = TensorPhysicsEngine(np.stack(bodies), dt=0.001)
    
    ground_top = -2.0 + 0.5
    contact_point = ground_top + ball_radius
    
    print(f"Initial height: {initial_height}m")
    print(f"Contact point: {contact_point}m")
    print(f"Drop height: {initial_height - contact_point}m")
    print(f"Restitution: {restitution}")
    
    # Track the motion
    max_height = initial_height
    min_height = initial_height
    has_bounced = False
    bounce_velocity = None
    
    for step in range(5000):
        engine.step()
        state = engine.get_state()
        ball_y = state[1, 1]  # Y position of ball
        ball_vy = state[1, 4]  # Y velocity of ball
        
        # Track extremes
        if ball_y > max_height:
            max_height = ball_y
        if ball_y < min_height:
            min_height = ball_y
        
        # Detect bounce
        if not has_bounced and ball_vy > 0 and ball_y < initial_height - 0.5:
            has_bounced = True
            bounce_velocity = ball_vy
            print(f"\nBounce at step {step}:")
            print(f"  Position: {ball_y:.3f}m")
            print(f"  Velocity: {ball_vy:.3f} m/s")
        
        # Stop after reaching peak after bounce
        if has_bounced and ball_vy < 0:
            print(f"\nPeak after bounce at step {step}:")
            print(f"  Max height: {max_height:.3f}m")
            print(f"  Bounce height: {max_height - contact_point:.3f}m")
            break
        
        if step % 500 == 0:
            print(f"Step {step}: y={ball_y:.3f}, vy={ball_vy:.3f}")
    
    # Calculate physics
    gravity = 9.81
    drop_height = initial_height - contact_point
    impact_velocity = np.sqrt(2 * gravity * drop_height)
    expected_bounce_velocity = restitution * impact_velocity
    expected_bounce_height = (expected_bounce_velocity**2) / (2 * gravity)
    
    print(f"\nPhysics analysis:")
    print(f"  Drop height: {drop_height:.3f}m")
    print(f"  Expected impact velocity: {-impact_velocity:.3f} m/s")
    print(f"  Expected bounce velocity: {expected_bounce_velocity:.3f} m/s")
    print(f"  Expected bounce height: {expected_bounce_height:.3f}m")
    print(f"  Actual bounce height: {max_height - contact_point:.3f}m")

if __name__ == "__main__":
    debug_falling_bounce()