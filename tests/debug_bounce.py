#!/usr/bin/env python3
"""Debug the bounce physics issue."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.engine import TensorPhysicsEngine
from physics.types import create_body_array, ShapeType

def debug_single_bounce():
    """Debug a single bounce to understand the physics."""
    print("\n=== Debugging Single Bounce ===")
    
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
    
    # Falling sphere with known velocity (just before impact)
    ball_radius = 0.5
    impact_velocity = -10.0  # m/s downward
    contact_y = -2.0 + 0.5 + ball_radius  # Just touching
    
    bodies.append(create_body_array(
        position=np.array([0., contact_y + 0.001, 0.], dtype=np.float32),  # Just above contact
        velocity=np.array([0., impact_velocity, 0.], dtype=np.float32),
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1.0,
        inertia=np.eye(3, dtype=np.float32) * (2.0/5.0) * 1.0 * (ball_radius**2),
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([ball_radius, 0., 0.], dtype=np.float32)
    ))
    
    # Create engine with known restitution
    restitution = 0.8
    engine = TensorPhysicsEngine(np.stack(bodies), dt=0.001)
    
    print(f"Initial conditions:")
    print(f"  Ball position: y={contact_y + 0.001:.3f}")
    print(f"  Ball velocity: vy={impact_velocity:.3f} m/s")
    print(f"  Restitution: {restitution}")
    print(f"  Expected bounce velocity: {-restitution * impact_velocity:.3f} m/s")
    
    # Run just a few steps to see the bounce
    velocities = []
    positions = []
    
    for step in range(10):
        engine.step()
        state = engine.get_state()
        ball_y = state[1, 1]  # Y position of ball
        ball_vy = state[1, 4]  # Y velocity of ball
        
        velocities.append(ball_vy)
        positions.append(ball_y)
        
        print(f"Step {step}: y={ball_y:.5f}, vy={ball_vy:.5f}")
        
        # Stop after bounce
        if ball_vy > 0 and step > 0:
            print(f"\nBounce detected!")
            print(f"  Velocity before: {impact_velocity:.3f} m/s")
            print(f"  Velocity after: {ball_vy:.3f} m/s")
            print(f"  Expected: {-restitution * impact_velocity:.3f} m/s")
            print(f"  Ratio: {ball_vy / (-impact_velocity):.3f}")
            print(f"  Expected ratio: {restitution:.3f}")
            break
    
    return velocities, positions

if __name__ == "__main__":
    debug_single_bounce()