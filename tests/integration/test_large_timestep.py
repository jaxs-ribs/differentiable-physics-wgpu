#!/usr/bin/env python3
"""Test with large timestep like main.py uses."""

import sys
import os

# Add parent directories to path to find test_setup
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_setup import setup_test_paths
setup_test_paths()

import numpy as np

from physics.engine import TensorPhysicsEngine
from physics.types import create_body_array, ShapeType

def test_large_dt():
    """Test with dt=0.016 like main.py."""
    print("\n=== Testing with large timestep (dt=0.016) ===")
    
    bodies = []
    
    # Ground at y=-2
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
    
    # Ball at y=5
    bodies.append(create_body_array(
        position=np.array([0., 5.0, 0.], dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1.0,
        inertia=np.eye(3, dtype=np.float32) * (2.0/5.0) * 1.0 * (0.5**2),
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([0.5, 0., 0.], dtype=np.float32)
    ))
    
    # Test with different timesteps
    for dt in [0.001, 0.016]:
        print(f"\n--- dt = {dt} ---")
        engine = TensorPhysicsEngine(np.stack(bodies), dt=dt)
        
        # Find collision
        prev_vy = 0
        for step in range(int(2.0 / dt)):  # Run for 2 seconds
            state = engine.get_state()
            y = state[1, 1]
            vy = state[1, 4]
            
            if prev_vy < 0 and vy > 0:
                print(f"Bounce at step {step}:")
                print(f"  Position: y={y:.3f}")
                print(f"  Velocity: {prev_vy:.3f} → {vy:.3f}")
                print(f"  Ratio: {abs(vy/prev_vy):.3f}")
                
                # Calculate expected
                impact_speed = abs(prev_vy)
                expected_bounce = 0.1 * impact_speed
                print(f"  Expected bounce velocity: {expected_bounce:.3f}")
                print(f"  ERROR FACTOR: {vy / expected_bounce:.2f}x")
                break
            
            prev_vy = vy
            engine.step()

if __name__ == "__main__":
    test_large_dt()