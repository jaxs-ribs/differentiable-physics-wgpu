#!/usr/bin/env python3
"""Replicate the exact issue from test_bounce_physics_multi.py."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.engine import TensorPhysicsEngine
from physics.types import create_body_array, ShapeType, BodySchema

def test_replicate():
    """Replicate the exact scenario."""
    print("\n=== Replicating Bounce Issue ===")
    
    bodies = []
    
    # Static ground box at y=-2 (same as failing test)
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
    
    # Falling sphere starting at y=5 (same as failing test)
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
    
    # Create engine (e=0.1 is hardcoded)
    engine = TensorPhysicsEngine(np.stack(bodies), dt=0.001)
    
    ground_top = -2.0 + 0.5  # Ground position + half thickness
    contact_point = ground_top + ball_radius  # y = -1.0
    
    print(f"Initial ball height: {initial_height}m")
    print(f"Drop height: {initial_height - contact_point} = {initial_height - contact_point}m")
    print(f"Contact point: {contact_point}m")
    print(f"Expected impact velocity: -sqrt(2*g*h) = -sqrt(2*9.81*{initial_height - contact_point}) = {-np.sqrt(2*9.81*(initial_height - contact_point)):.3f} m/s")
    print(f"Engine restitution: 0.1 (hardcoded)")
    print(f"Expected bounce velocity: {np.sqrt(2*9.81*(initial_height - contact_point)) * 0.1:.3f} m/s")
    print(f"Expected max height after bounce: {contact_point + (np.sqrt(2*9.81*(initial_height - contact_point)) * 0.1)**2 / (2*9.81):.3f}m")
    
    # Track first bounce
    prev_y = initial_height
    prev_vy = 0
    bounce_found = False
    max_after_bounce = contact_point
    
    print("\nSimulating...")
    for step in range(10000):
        state = engine.get_state()
        y = state[1, BodySchema.POS_Y]
        vy = state[1, BodySchema.VEL_Y]
        
        if step % 1000 == 0:
            print(f"  Step {step}: y={y:.3f}, vy={vy:.3f}")
        
        # Detect bounce
        if prev_vy < 0 and vy > 0 and not bounce_found:
            bounce_found = True
            print(f"\nBounce at step {step}!")
            print(f"  Position: {y:.3f}")
            print(f"  Velocity: {prev_vy:.3f} → {vy:.3f}")
            print(f"  Expected: {prev_vy:.3f} → {-prev_vy * 0.1:.3f}")
            print(f"  Ratio: {abs(vy/prev_vy):.3f} (expected 0.1)")
        
        # Track max height after bounce
        if bounce_found and y > max_after_bounce:
            max_after_bounce = y
        
        # Stop after reaching peak
        if bounce_found and vy < 0 and prev_vy >= 0:
            print(f"\nPeak after bounce: {max_after_bounce:.3f}m")
            print(f"Expected peak: ~{contact_point + 0.01 * (initial_height - contact_point):.3f}m")
            print(f"Height ratio: {(max_after_bounce - contact_point) / (initial_height - contact_point):.3f}")
            print(f"Expected ratio: 0.01 (e²)")
            break
        
        prev_y = y
        prev_vy = vy
        engine.step()

if __name__ == "__main__":
    test_replicate()