#!/usr/bin/env python3
"""Debug if double collisions are happening."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.engine import TensorPhysicsEngine
from physics.types import create_body_array, ShapeType
from tinygrad.tensor import Tensor

def debug_collision_count():
    """Check if multiple collisions are being detected per contact."""
    print("\n=== Debugging Collision Count ===")
    
    # Create a simple scene
    bodies = []
    
    # Static ground box at y=-2
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
    
    # Ball just above contact
    ball_radius = 0.5
    contact_y = -2.0 + 0.5 + ball_radius
    
    bodies.append(create_body_array(
        position=np.array([0., contact_y + 0.01, 0.], dtype=np.float32),  # 1cm above contact
        velocity=np.array([0., -1.0, 0.], dtype=np.float32),  # Slow fall
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1.0,
        inertia=np.eye(3, dtype=np.float32) * (2.0/5.0) * 1.0 * (ball_radius**2),
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([ball_radius, 0., 0.], dtype=np.float32)
    ))
    
    # Create engine
    engine = TensorPhysicsEngine(np.stack(bodies), dt=0.001)
    
    # Manually step through collision detection
    from physics.broadphase_tensor import detect_potential_collisions_tensor
    from physics.narrowphase import handle_collisions_mixed_shapes_tensor
    
    print("Initial state:")
    state = engine.get_state()
    print(f"  Ball: y={state[1, 1]:.4f}, vy={state[1, 4]:.4f}")
    
    for step in range(5):
        # Get current state
        bodies_tensor = engine.bodies
        
        # Broadphase
        collision_pairs, collision_mask = detect_potential_collisions_tensor(
            bodies_tensor[:, :3],  # positions
            Tensor.ones(2, 1),  # All bodies active
            threshold=10.0
        )
        
        num_potential = collision_mask.sum().numpy()
        print(f"\nStep {step}: {num_potential} potential collisions")
        
        if num_potential > 0:
            # Get collision details
            pair_indices = collision_pairs[collision_mask].numpy()
            for i, (a, b) in enumerate(pair_indices):
                print(f"  Pair {i}: bodies {a} and {b}")
        
        # Step the engine
        engine.step()
        state = engine.get_state()
        print(f"  After step: y={state[1, 1]:.4f}, vy={state[1, 4]:.4f}")

if __name__ == "__main__":
    debug_collision_count()