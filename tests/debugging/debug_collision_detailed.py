#!/usr/bin/env python3
"""Debug collision detection in detail."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.engine import TensorPhysicsEngine
from physics.types import create_body_array, ShapeType

def debug_collision_step_by_step():
    """Debug collision detection step by step."""
    print("\n=== Debugging Collision Step by Step ===")
    
    # Create a simple scene: ground and ball just above it
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
    
    # Ball just above ground, falling
    ball_radius = 0.5
    contact_y = -2.0 + 0.5 + ball_radius  # Just touching
    
    bodies.append(create_body_array(
        position=np.array([0., contact_y + 0.1, 0.], dtype=np.float32),  # 10cm above contact
        velocity=np.array([0., -5.0, 0.], dtype=np.float32),  # Falling at 5 m/s
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1.0,
        inertia=np.eye(3, dtype=np.float32) * (2.0/5.0) * 1.0 * (ball_radius**2),
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([ball_radius, 0., 0.], dtype=np.float32)
    ))
    
    # Create engine
    engine = TensorPhysicsEngine(np.stack(bodies), dt=0.01)  # Larger timestep to see changes
    
    print(f"Initial setup:")
    print(f"  Ground: y=-2, top at y=-1.5")
    print(f"  Ball: radius={ball_radius}, initial y={contact_y + 0.1:.3f}")
    print(f"  Contact should occur at y={contact_y:.3f}")
    
    # Run a few steps
    for step in range(5):
        print(f"\n--- Step {step} ---")
        state = engine.get_state()
        
        ball_y = state[1, 1]
        ball_vy = state[1, 4]
        ground_y = state[0, 1]
        
        print(f"Ball: y={ball_y:.4f}, vy={ball_vy:.4f}")
        print(f"Ground: y={ground_y:.4f}")
        print(f"Distance to contact: {ball_y - contact_y:.4f}")
        
        # Step the engine
        engine.step()
        
        # Check after step
        state_after = engine.get_state()
        ball_y_after = state_after[1, 1]
        ball_vy_after = state_after[1, 4]
        
        print(f"After step: y={ball_y_after:.4f}, vy={ball_vy_after:.4f}")
        print(f"Change: Δy={ball_y_after - ball_y:.4f}, Δvy={ball_vy_after - ball_vy:.4f}")

if __name__ == "__main__":
    debug_collision_step_by_step()