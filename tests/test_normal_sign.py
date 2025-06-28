#!/usr/bin/env python3
"""Test normal sign convention."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['JIT'] = '0'

from physics.types import create_body_array, ShapeType
from physics.broadphase_tensor import differentiable_broadphase
from physics.narrowphase import narrowphase
from tinygrad import Tensor

def test_normal_sign():
    """Test normal sign for box-sphere collision."""
    
    # Test 1: Box above sphere
    print("Test 1: Box above sphere")
    bodies1 = []
    
    # Box at y=0
    bodies1.append(create_body_array(
        position=np.array([0., 0., 0.], dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1.0,
        inertia=np.eye(3, dtype=np.float32),
        shape_type=ShapeType.BOX,
        shape_params=np.array([1., 0.5, 1.], dtype=np.float32)
    ))
    
    # Sphere below at y=-1.4 (touching)
    bodies1.append(create_body_array(
        position=np.array([0., -1.4, 0.], dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1.0,
        inertia=np.eye(3, dtype=np.float32) * 0.1,
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([0.5, 0., 0.], dtype=np.float32)
    ))
    
    bodies_t1 = Tensor(np.stack(bodies1))
    pair_indices1, collision_mask1 = differentiable_broadphase(bodies_t1)
    contact_normals1, _, _, contact_mask1, _ = narrowphase(bodies_t1, pair_indices1, collision_mask1)
    
    if contact_mask1.sum().numpy() > 0:
        normal = contact_normals1.numpy()[0]
        print(f"  Normal: {normal}")
        print(f"  Points: {'downward' if normal[1] < 0 else 'upward'}")
    
    # Test 2: Sphere above box
    print("\nTest 2: Sphere above box")
    bodies2 = []
    
    # Sphere at y=0
    bodies2.append(create_body_array(
        position=np.array([0., 0., 0.], dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1.0,
        inertia=np.eye(3, dtype=np.float32) * 0.1,
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([0.5, 0., 0.], dtype=np.float32)
    ))
    
    # Box below at y=-1.4
    bodies2.append(create_body_array(
        position=np.array([0., -1.4, 0.], dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1.0,
        inertia=np.eye(3, dtype=np.float32),
        shape_type=ShapeType.BOX,
        shape_params=np.array([1., 0.5, 1.], dtype=np.float32)
    ))
    
    bodies_t2 = Tensor(np.stack(bodies2))
    pair_indices2, collision_mask2 = differentiable_broadphase(bodies_t2)
    contact_normals2, _, _, contact_mask2, _ = narrowphase(bodies_t2, pair_indices2, collision_mask2)
    
    if contact_mask2.sum().numpy() > 0:
        normal = contact_normals2.numpy()[0]
        print(f"  Normal: {normal}")
        print(f"  Points: {'downward' if normal[1] < 0 else 'upward'}")
    
    # Test 3: Original failing case
    print("\nTest 3: Original case (ground box at -2, sphere falling)")
    bodies3 = []
    
    # Ground box at y=-2
    bodies3.append(create_body_array(
        position=np.array([0., -2., 0.], dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1e8,
        inertia=np.eye(3, dtype=np.float32) * 1e8,
        shape_type=ShapeType.BOX,
        shape_params=np.array([10., 0.5, 10.], dtype=np.float32)
    ))
    
    # Sphere just touching
    bodies3.append(create_body_array(
        position=np.array([0., -1.01, 0.], dtype=np.float32),
        velocity=np.array([0., -10., 0.], dtype=np.float32),
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1.0,
        inertia=np.eye(3, dtype=np.float32) * 0.1,
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([0.5, 0., 0.], dtype=np.float32)
    ))
    
    bodies_t3 = Tensor(np.stack(bodies3))
    pair_indices3, collision_mask3 = differentiable_broadphase(bodies_t3)
    contact_normals3, _, _, contact_mask3, pair_indices3 = narrowphase(bodies_t3, pair_indices3, collision_mask3)
    
    if contact_mask3.sum().numpy() > 0:
        normal = contact_normals3.numpy()[0]
        pair = pair_indices3.numpy()[0]
        print(f"  Pair: {pair}")
        print(f"  Normal: {normal}")
        print(f"  Points: {'downward (from sphere to box)' if normal[1] < 0 else 'upward (from box to sphere)'}")
        
        # Check relative velocity
        vel_a = bodies_t3.numpy()[pair[0], 3:6]
        vel_b = bodies_t3.numpy()[pair[1], 3:6]
        v_rel = vel_a - vel_b
        v_rel_n = np.dot(v_rel, normal)
        print(f"  v_rel = {v_rel}")
        print(f"  v_rel·n = {v_rel_n}")
        print(f"  Bodies are: {'approaching' if v_rel_n < 0 else 'separating'}")

if __name__ == "__main__":
    test_normal_sign()