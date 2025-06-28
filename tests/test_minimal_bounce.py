#!/usr/bin/env python3
"""Minimal test to isolate bounce issue."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.engine import TensorPhysicsEngine
from physics.types import create_body_array, ShapeType

def test_minimal_bounce():
    """Minimal bounce test."""
    print("\n=== Minimal Bounce Test ===")
    
    # Just two bodies
    bodies = []
    
    # Ground
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
    
    # Ball
    bodies.append(create_body_array(
        position=np.array([0., 0., 0.], dtype=np.float32),
        velocity=np.array([0., -10., 0.], dtype=np.float32),
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1.0,
        inertia=np.eye(3, dtype=np.float32) * 0.1,
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([0.5, 0., 0.], dtype=np.float32)
    ))
    
    engine = TensorPhysicsEngine(np.stack(bodies), dt=0.001)
    
    print("Initial: ball at y=0, vy=-10")
    print("Expected: after bounce, vy=+1.0 (restitution=0.1)")
    
    # Track velocity
    max_upward_velocity = 0
    for step in range(1000):
        state = engine.get_state()
        vy = state[1, 4]
        
        if vy > max_upward_velocity:
            max_upward_velocity = vy
            print(f"Step {step}: New max upward velocity: {vy:.3f} m/s")
        
        engine.step()
        
        if vy > 0 and step > 500:
            break
    
    print(f"\nFinal max upward velocity: {max_upward_velocity:.3f} m/s")
    print(f"Expected: 1.0 m/s")
    print(f"Ratio: {max_upward_velocity / 1.0:.2f}x")
    
    return abs(max_upward_velocity - 1.0) < 0.1

if __name__ == "__main__":
    success = test_minimal_bounce()
    sys.exit(0 if success else 1)