#!/usr/bin/env python3
"""Test if narrowphase creates double contacts."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['JIT'] = '0'

from physics.types import create_body_array, ShapeType
from physics.broadphase_tensor import differentiable_broadphase
from physics.narrowphase import narrowphase
from tinygrad import Tensor

def test_narrowphase():
    """Test narrowphase in detail."""
    bodies = []
    
    # Try different orderings
    print("Test 1: Box(0), Sphere(1)")
    
    # Box first
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
    
    # Sphere second
    bodies.append(create_body_array(
        position=np.array([0., -1.01, 0.], dtype=np.float32),
        velocity=np.array([0., -10., 0.], dtype=np.float32),
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1.0,
        inertia=np.eye(3, dtype=np.float32) * 0.1,
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([0.5, 0., 0.], dtype=np.float32)
    ))
    
    bodies_t = Tensor(np.stack(bodies))
    
    # Get broadphase pairs
    pair_indices, collision_mask = differentiable_broadphase(bodies_t)
    print(f"Broadphase found {collision_mask.sum().numpy()} pairs")
    
    # Check what narrowphase does
    contact_normals, contact_depths, contact_points, contact_mask, pair_indices_out = narrowphase(
        bodies_t, pair_indices, collision_mask
    )
    
    num_contacts = contact_mask.sum().numpy()
    print(f"Narrowphase found {num_contacts} contacts")
    
    # Print all contacts
    for i in range(len(contact_mask.numpy())):
        if contact_mask.numpy()[i]:
            print(f"  Contact {i}: pair {pair_indices_out.numpy()[i]}, "
                  f"normal {contact_normals.numpy()[i]}, "
                  f"depth {contact_depths.numpy()[i]:.3f}")
    
    # Now test the other ordering
    print("\n\nTest 2: Sphere(0), Box(1)")
    bodies2 = [bodies[1], bodies[0]]  # Swap order
    bodies_t2 = Tensor(np.stack(bodies2))
    
    pair_indices2, collision_mask2 = differentiable_broadphase(bodies_t2)
    print(f"Broadphase found {collision_mask2.sum().numpy()} pairs")
    
    contact_normals2, contact_depths2, contact_points2, contact_mask2, pair_indices_out2 = narrowphase(
        bodies_t2, pair_indices2, collision_mask2
    )
    
    num_contacts2 = contact_mask2.sum().numpy()
    print(f"Narrowphase found {num_contacts2} contacts")
    
    for i in range(len(contact_mask2.numpy())):
        if contact_mask2.numpy()[i]:
            print(f"  Contact {i}: pair {pair_indices_out2.numpy()[i]}, "
                  f"normal {contact_normals2.numpy()[i]}, "
                  f"depth {contact_depths2.numpy()[i]:.3f}")
    
    # Test with more bodies to see if we get duplicates
    print("\n\nTest 3: Multiple bodies")
    bodies3 = []
    
    # Ground
    bodies3.append(bodies[0])
    
    # Two spheres
    for i in range(2):
        bodies3.append(create_body_array(
            position=np.array([i*2.0, -1.01, 0.], dtype=np.float32),
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
    print(f"Broadphase found {collision_mask3.sum().numpy()} pairs")
    print(f"Pairs: {pair_indices3.numpy()}")
    
    contact_normals3, contact_depths3, contact_points3, contact_mask3, pair_indices_out3 = narrowphase(
        bodies_t3, pair_indices3, collision_mask3
    )
    
    num_contacts3 = contact_mask3.sum().numpy()
    print(f"Narrowphase found {num_contacts3} contacts")

if __name__ == "__main__":
    test_narrowphase()