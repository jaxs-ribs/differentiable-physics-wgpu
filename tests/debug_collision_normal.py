#!/usr/bin/env python3
"""Debug collision normal direction."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.engine import TensorPhysicsEngine
from physics.types import create_body_array, ShapeType
from physics.narrowphase import detect_sphere_box_collisions_tensor
from tinygrad.tensor import Tensor

def debug_collision_normal():
    """Debug the collision normal direction for sphere-box."""
    print("\n=== Debugging Collision Normal ===")
    
    # Create a sphere above a box
    # Box at origin (A)
    box_pos = Tensor([0., 0., 0.])
    box_quat = Tensor([1., 0., 0., 0.])
    box_params = Tensor([10., 0.5, 10.])  # Half extents
    
    # Sphere above box (B)
    sphere_pos = Tensor([0., 1.0, 0.])  # 1m above box center
    sphere_quat = Tensor([1., 0., 0., 0.])
    sphere_params = Tensor([0.5, 0., 0.])  # Radius 0.5
    
    # Detect collision
    # For sphere-box, box is A and sphere is B
    normals, depths, points, mask, indices = detect_sphere_box_collisions_tensor(
        box_pos.unsqueeze(0), box_quat.unsqueeze(0), box_params.unsqueeze(0),
        sphere_pos.unsqueeze(0), sphere_quat.unsqueeze(0), sphere_params.unsqueeze(0),
        Tensor([[0, 0]]), Tensor([True])
    )
    
    print(f"Box (A) position: {box_pos.numpy()}")
    print(f"Sphere (B) position: {sphere_pos.numpy()}")
    print(f"Collision mask: {mask.numpy()}")
    print(f"Contact normal: {normals.numpy()}")
    print(f"Contact depth: {depths.numpy()}")
    print(f"Contact point: {points.numpy()}")
    
    # The normal should point from B to A (from sphere to box)
    # Since sphere is above box, normal should point downward (negative Y)
    expected_normal = np.array([0., -1., 0.])
    actual_normal = normals.numpy()[0]
    
    print(f"\nExpected normal (B to A): {expected_normal}")
    print(f"Actual normal: {actual_normal}")
    
    # Now let's check the impulse calculation
    # If sphere is falling down with velocity [0, -10, 0]
    # And box is static with velocity [0, 0, 0]
    sphere_vel = np.array([0., -10., 0.])
    box_vel = np.array([0., 0., 0.])
    
    v_rel = box_vel - sphere_vel  # v_a - v_b
    print(f"\nVelocities:")
    print(f"  Box (A): {box_vel}")
    print(f"  Sphere (B): {sphere_vel}")
    print(f"  v_rel = v_a - v_b = {v_rel}")
    
    # Project onto normal
    v_rel_normal = np.dot(v_rel, actual_normal)
    print(f"  v_rel_normal = {v_rel_normal:.3f}")
    print(f"  This is {'< 0 (approaching)' if v_rel_normal < 0 else '>= 0 (separating)'}")

if __name__ == "__main__":
    debug_collision_normal()