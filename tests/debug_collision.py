#!/usr/bin/env python3
"""Debug collision resolution to understand the bounce issue."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.engine import TensorPhysicsEngine
from physics.types import create_body_array, ShapeType
from tinygrad import Tensor

def debug_single_collision():
    """Debug a single collision step."""
    print("\n=== Debugging Single Collision ===")
    
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
    
    # Ball just about to hit ground
    ball_radius = 0.5
    # Position ball so it's intersecting the ground
    ball_y = -2.0 + 0.5 + ball_radius - 0.1  # 0.1m penetration
    bodies.append(create_body_array(
        position=np.array([0., ball_y, 0.], dtype=np.float32),
        velocity=np.array([0., -10.0, 0.], dtype=np.float32),  # Moving down at 10 m/s
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1.0,
        inertia=np.eye(3, dtype=np.float32) * (2.0/5.0) * 1.0 * (ball_radius**2),
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([ball_radius, 0., 0.], dtype=np.float32)
    ))
    
    print(f"Initial state:")
    print(f"  Ground: y={bodies[0][1]:.3f}, vy={bodies[0][4]:.3f}")
    print(f"  Ball: y={bodies[1][1]:.3f}, vy={bodies[1][4]:.3f}")
    
    # Create engine
    engine = TensorPhysicsEngine(np.stack(bodies), dt=0.001)
    
    # Step once
    engine.step()
    
    # Check result
    state = engine.get_state()
    print(f"\nAfter one step:")
    print(f"  Ground: y={state[0,1]:.3f}, vy={state[0,4]:.3f}")
    print(f"  Ball: y={state[1,1]:.3f}, vy={state[1,4]:.3f}")
    
    # Expected behavior with restitution = 0.1:
    # Ball velocity should change from -10 m/s to about +1 m/s (10% restitution)
    expected_vy = 10.0 * 0.1
    actual_vy = state[1,4]
    
    print(f"\nExpected ball vy after collision: {expected_vy:.3f} m/s")
    print(f"Actual ball vy after collision: {actual_vy:.3f} m/s")
    
    if abs(actual_vy - expected_vy) > 5.0:
        print("❌ FAIL: Velocity change is way off!")
    else:
        print("✅ PASS: Velocity change is reasonable")

if __name__ == "__main__":
    debug_single_collision()