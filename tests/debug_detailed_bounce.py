#!/usr/bin/env python3
"""Debug bounce in extreme detail."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.engine import TensorPhysicsEngine
from physics.types import create_body_array, ShapeType

# Monkey patch the solver to add debug output
import physics.solver as solver
original_resolve = solver.resolve_collisions

def debug_resolve_collisions(bodies, contact_normals, contact_depths, 
                                  contact_points, contact_mask, restitution=0.1):
    """Wrapped version with debug output."""
    # Get collision info
    num_contacts = contact_mask.sum().numpy()
    if num_contacts > 0:
        print(f"\n=== COLLISION RESOLUTION ===")
        print(f"Number of contacts: {num_contacts}")
        
        # Get the first contact
        idx = contact_mask.numpy().nonzero()[0][0]
        normal = contact_normals[idx].numpy()
        depth = contact_depths[idx].numpy()
        point = contact_points[idx].numpy()
        
        print(f"Contact 0:")
        print(f"  Normal: {normal} (magnitude: {np.linalg.norm(normal):.3f})")
        print(f"  Depth: {depth:.4f}")
        print(f"  Point: {point}")
        
        # Get velocities
        indices = [(0, 1)]  # Assuming ground and ball
        for a, b in indices:
            vel_a = bodies[a, 3:6].numpy()
            vel_b = bodies[b, 3:6].numpy()
            v_rel = vel_a - vel_b
            v_rel_n = np.dot(v_rel, normal)
            
            print(f"\nPair ({a}, {b}):")
            print(f"  Vel A: {vel_a}")
            print(f"  Vel B: {vel_b}")
            print(f"  V_rel: {v_rel}")
            print(f"  V_rel_normal: {v_rel_n:.3f}")
            
            if v_rel_n < 0:
                print(f"  -> Approaching (need impulse)")
                j = -(1 + restitution) * v_rel_n  # Simplified for debug
                print(f"  -> Impulse magnitude: {j:.3f}")
            else:
                print(f"  -> Separating (skip)")
    
    return original_resolve(bodies, contact_normals, contact_depths, 
                          contact_points, contact_mask, restitution)

solver.resolve_collisions = debug_resolve_collisions

def test_detailed_bounce():
    """Test with detailed debug output."""
    print("\n=== Detailed Bounce Debug ===")
    
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
    
    # Ball just before impact
    ball_radius = 0.5
    contact_y = -2.0 + 0.5 + ball_radius
    
    bodies.append(create_body_array(
        position=np.array([0., contact_y + 0.01, 0.], dtype=np.float32),
        velocity=np.array([0., -5.0, 0.], dtype=np.float32),
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1.0,
        inertia=np.eye(3, dtype=np.float32) * (2.0/5.0) * 1.0 * (ball_radius**2),
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([ball_radius, 0., 0.], dtype=np.float32)
    ))
    
    engine = TensorPhysicsEngine(np.stack(bodies), dt=0.01)
    
    print(f"Initial state:")
    print(f"  Ball: y={contact_y + 0.01:.3f}, vy=-5.0")
    print(f"  Expected bounce velocity: {0.1 * 5.0:.3f} m/s")
    
    # Run just a few steps
    for step in range(3):
        print(f"\n--- Step {step} ---")
        state = engine.get_state()
        ball_y = state[1, 1]
        ball_vy = state[1, 4]
        print(f"Ball: y={ball_y:.4f}, vy={ball_vy:.4f}")
        
        engine.step()
        
        state_after = engine.get_state()
        ball_y_after = state_after[1, 1]
        ball_vy_after = state_after[1, 4]
        print(f"After step: y={ball_y_after:.4f}, vy={ball_vy_after:.4f}")

if __name__ == "__main__":
    test_detailed_bounce()