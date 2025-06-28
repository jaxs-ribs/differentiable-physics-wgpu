#!/usr/bin/env python3
"""Debug where energy is being added."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.engine import TensorPhysicsEngine
from physics.types import create_body_array, ShapeType

def debug_energy_gain():
    """Track energy through a bounce."""
    print("\n=== Debugging Energy Gain ===")
    
    # Create scene
    bodies = []
    
    # Static ground
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
    
    # Falling ball
    ball_radius = 0.5
    mass = 1.0
    bodies.append(create_body_array(
        position=np.array([0., 1.0, 0.], dtype=np.float32),  # 1m high
        velocity=np.zeros(3, dtype=np.float32),
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=mass,
        inertia=np.eye(3, dtype=np.float32) * (2.0/5.0) * mass * (ball_radius**2),
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([ball_radius, 0., 0.], dtype=np.float32)
    ))
    
    engine = TensorPhysicsEngine(np.stack(bodies), dt=0.001)
    
    contact_y = -2.0 + 0.5 + ball_radius  # -1.0
    gravity = 9.81
    
    def calculate_energy(y, vy):
        """Calculate total mechanical energy."""
        ke = 0.5 * mass * vy**2
        pe = mass * gravity * (y - contact_y)  # PE relative to contact point
        return ke + pe, ke, pe
    
    print(f"Contact point: {contact_y}m")
    print(f"Gravity: {gravity} m/s²")
    print(f"Mass: {mass} kg")
    print(f"Restitution: 0.1")
    
    # Track energy
    prev_y = 1.0
    prev_vy = 0.0
    has_bounced = False
    energy_before_bounce = None
    energy_after_bounce = None
    
    for step in range(2000):
        state = engine.get_state()
        y = state[1, 1]
        vy = state[1, 4]
        
        total, ke, pe = calculate_energy(y, vy)
        
        # Detect bounce
        if not has_bounced and prev_vy < 0 and vy > 0 and y < 0:
            has_bounced = True
            energy_before_bounce = calculate_energy(prev_y, prev_vy)
            energy_after_bounce = calculate_energy(y, vy)
            
            print(f"\n=== BOUNCE DETECTED at step {step} ===")
            print(f"Before: y={prev_y:.4f}, vy={prev_vy:.4f}")
            print(f"  Energy: Total={energy_before_bounce[0]:.3f}, KE={energy_before_bounce[1]:.3f}, PE={energy_before_bounce[2]:.3f}")
            print(f"After: y={y:.4f}, vy={vy:.4f}")
            print(f"  Energy: Total={energy_after_bounce[0]:.3f}, KE={energy_after_bounce[1]:.3f}, PE={energy_after_bounce[2]:.3f}")
            print(f"Energy gain: {energy_after_bounce[0] - energy_before_bounce[0]:.3f} J")
            
            # Expected energy after bounce
            impact_velocity = np.sqrt(2 * gravity * 2.0)  # Dropped from 2m above contact
            bounce_velocity = 0.1 * impact_velocity  # restitution = 0.1
            expected_ke = 0.5 * mass * bounce_velocity**2
            print(f"\nExpected after bounce:")
            print(f"  Impact velocity: {-impact_velocity:.3f} m/s")
            print(f"  Bounce velocity: {bounce_velocity:.3f} m/s")
            print(f"  Expected KE: {expected_ke:.3f} J")
            break
        
        if step % 200 == 0:
            print(f"Step {step}: y={y:.3f}, vy={vy:.3f}, E={total:.3f}")
        
        prev_y = y
        prev_vy = vy
        engine.step()

if __name__ == "__main__":
    debug_energy_gain()