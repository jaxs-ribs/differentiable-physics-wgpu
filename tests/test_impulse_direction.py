#!/usr/bin/env python3
"""Test impulse direction calculation."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['JIT'] = '0'

from physics.types import create_body_array, ShapeType, BodySchema
from physics.solver import resolve_collisions
from tinygrad import Tensor

def test_impulse_direction():
    """Test the impulse calculation in detail."""
    print("\n=== Testing Impulse Direction ===")
    
    # Simple test case: ball hitting ground
    # Ground is body 0, ball is body 1
    # Normal should point from ground (B) to ball (A), i.e., upward [0, 1, 0]
    
    bodies = []
    
    # Ground at y=0
    bodies.append(create_body_array(
        position=np.array([0., 0., 0.], dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1e8,
        inertia=np.eye(3, dtype=np.float32) * 1e8,
        shape_type=ShapeType.BOX,
        shape_params=np.array([10., 0.5, 10.], dtype=np.float32)
    ))
    
    # Ball touching ground, moving down
    bodies.append(create_body_array(
        position=np.array([0., 1.0, 0.], dtype=np.float32),
        velocity=np.array([0., -1.0, 0.], dtype=np.float32),
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1.0,
        inertia=np.eye(3, dtype=np.float32) * 0.1,
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([0.5, 0., 0.], dtype=np.float32)
    ))
    
    bodies_t = Tensor(np.stack(bodies))
    
    # Create manual contact data
    # Pair indices: (ground=0, ball=1)
    pair_indices = Tensor([[0, 1]], dtype='int32')
    
    # Normal should point from ground to ball (upward)
    contact_normals = Tensor([[0., 1., 0.]])
    
    # Small penetration depth
    contact_depths = Tensor([0.01])
    
    # Contact point at ground surface
    contact_points = Tensor([[0., 0.5, 0.]])
    
    # Active contact
    contact_mask = Tensor([True])
    
    print("Setup:")
    print(f"  Ball velocity: {bodies[1][BodySchema.VEL_Y]}")
    print(f"  Contact normal: {contact_normals.numpy()[0]}")
    print(f"  Expected impulse direction: upward (positive Y)")
    
    # Test different restitution values
    for e in [0.0, 0.5, 1.0]:
        print(f"\nTesting with e={e}:")
        
        # Apply collision resolution
        bodies_after = resolve_collisions(
            bodies_t, pair_indices, contact_normals, contact_depths,
            contact_points, contact_mask, restitution=e
        )
        
        vel_before = bodies_t[1, BodySchema.VEL_Y].numpy()
        vel_after = bodies_after[1, BodySchema.VEL_Y].numpy()
        
        print(f"  Velocity: {vel_before:.3f} → {vel_after:.3f}")
        print(f"  Expected: {vel_before:.3f} → {-e * vel_before:.3f}")
        
        if abs(vel_after - (-e * vel_before)) > 0.001:
            print(f"  ❌ ERROR: Mismatch!")
        else:
            print(f"  ✓ Correct")
    
    # Now test with normal pointing wrong way
    print("\n\nTesting with REVERSED normal:")
    contact_normals_reversed = Tensor([[0., -1., 0.]])
    
    for e in [0.1]:
        bodies_after = resolve_collisions(
            bodies_t, pair_indices, contact_normals_reversed, contact_depths,
            contact_points, contact_mask, restitution=e
        )
        
        vel_before = bodies_t[1, BodySchema.VEL_Y].numpy()
        vel_after = bodies_after[1, BodySchema.VEL_Y].numpy()
        
        print(f"  Velocity with reversed normal: {vel_before:.3f} → {vel_after:.3f}")

if __name__ == "__main__":
    test_impulse_direction()