#!/usr/bin/env python3
"""Test pair index ordering."""

import sys
import os

# Add parent directories to path to find test_setup
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_setup import setup_test_paths
setup_test_paths()

import numpy as np

os.environ['JIT'] = '0'

from physics.types import create_body_array, ShapeType, BodySchema
from physics.broadphase_tensor import differentiable_broadphase
from physics.narrowphase import narrowphase
from physics.solver import resolve_collisions
from tinygrad import Tensor

def test_index_order():
    """Test how collision pairs are indexed."""
    print("\n=== Testing Index Order in Collision ===")
    
    bodies = []
    
    # Test 1: Ground first, ball second
    print("\nTest 1: Ground (0), Ball (1)")
    
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
    
    # Ball
    bodies.append(create_body_array(
        position=np.array([0., 0.51, 0.], dtype=np.float32),
        velocity=np.array([0., -1.0, 0.], dtype=np.float32),
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1.0,
        inertia=np.eye(3, dtype=np.float32) * 0.1,
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([0.5, 0., 0.], dtype=np.float32)
    ))
    
    bodies_t = Tensor(np.stack(bodies))
    
    # Get collision
    pair_indices, collision_mask = differentiable_broadphase(bodies_t)
    contact_normals, contact_depths, contact_points, contact_mask, pair_indices = narrowphase(
        bodies_t, pair_indices, collision_mask
    )
    
    if contact_mask.sum().numpy() > 0:
        idx = 0
        pair = pair_indices.numpy()[idx]
        normal = contact_normals.numpy()[idx]
        
        print(f"  Pair: ({pair[0]}, {pair[1]})")
        print(f"  indices_a = {pair[0]} (ground)")
        print(f"  indices_b = {pair[1]} (ball)")
        print(f"  Normal: {normal}")
        
        # In solver:
        # v_rel = v_a - v_b = v_ground - v_ball = [0,0,0] - [0,-1,0] = [0,1,0]
        # v_rel·n = [0,1,0]·[0,-1,0] = -1 < 0 (approaching)
        
        v_ground = bodies_t.numpy()[pair[0], 3:6]
        v_ball = bodies_t.numpy()[pair[1], 3:6]
        v_rel = v_ground - v_ball
        v_rel_n = np.dot(v_rel, normal)
        
        print(f"\n  In solver:")
        print(f"  v_rel = v_a - v_b = {v_ground} - {v_ball} = {v_rel}")
        print(f"  v_rel·n = {v_rel_n}")
        print(f"  Approaching: {v_rel_n < 0}")
        
        # Apply collision
        bodies_after = resolve_collisions(
            bodies_t, pair_indices, contact_normals, contact_depths,
            contact_points, contact_mask, restitution=0.1
        )
        
        vel_after = bodies_after.numpy()[1, 3:6]
        print(f"\n  Ball velocity: {v_ball} → {vel_after}")
    
    # Test 2: Ball first, ground second
    print("\n\nTest 2: Ball (0), Ground (1)")
    bodies2 = [bodies[1], bodies[0]]  # Swap order
    bodies2_t = Tensor(np.stack(bodies2))
    
    # Get collision
    pair_indices2, collision_mask2 = differentiable_broadphase(bodies2_t)
    contact_normals2, contact_depths2, contact_points2, contact_mask2, pair_indices2 = narrowphase(
        bodies2_t, pair_indices2, collision_mask2
    )
    
    if contact_mask2.sum().numpy() > 0:
        idx = 0
        pair = pair_indices2.numpy()[idx]
        normal = contact_normals2.numpy()[idx]
        
        print(f"  Pair: ({pair[0]}, {pair[1]})")
        print(f"  indices_a = {pair[0]} (ball)")
        print(f"  indices_b = {pair[1]} (ground)")
        print(f"  Normal: {normal}")
        
        v_ball = bodies2_t.numpy()[pair[0], 3:6]
        v_ground = bodies2_t.numpy()[pair[1], 3:6]
        v_rel = v_ball - v_ground
        v_rel_n = np.dot(v_rel, normal)
        
        print(f"\n  In solver:")
        print(f"  v_rel = v_a - v_b = {v_ball} - {v_ground} = {v_rel}")
        print(f"  v_rel·n = {v_rel_n}")
        print(f"  Approaching: {v_rel_n < 0}")

if __name__ == "__main__":
    test_index_order()