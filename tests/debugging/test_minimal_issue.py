#!/usr/bin/env python3
"""Minimal test to reproduce the issue."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# CRITICAL: Don't disable JIT to see if that's the issue
# os.environ['JIT'] = '0'

from physics.engine import TensorPhysicsEngine, _physics_step_static
from physics.types import create_body_array, ShapeType, BodySchema
from tinygrad import Tensor

def test_minimal():
    """Minimal test."""
    bodies = []
    
    # Ground
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
    
    # Ball just before collision
    bodies.append(create_body_array(
        position=np.array([0., -1.005, 0.], dtype=np.float32),
        velocity=np.array([0., -10.85, 0.], dtype=np.float32),
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1.0,
        inertia=np.eye(3, dtype=np.float32) * 0.1,
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([0.5, 0., 0.], dtype=np.float32)
    ))
    
    bodies_t = Tensor(np.stack(bodies))
    gravity = Tensor([0., -9.81, 0.])
    
    print("Test 1: Direct call to _physics_step_static")
    bodies_copy = bodies_t.clone()
    
    vel_before = bodies_copy.numpy()[1, 3:6]
    print(f"Before: vel={vel_before}")
    
    bodies_after = _physics_step_static(bodies_copy, 0.001, gravity)
    
    vel_after = bodies_after.numpy()[1, 3:6]
    print(f"After: vel={vel_after}")
    print(f"Change: {vel_after - vel_before}")
    
    # Test 2: Using TensorPhysicsEngine
    print("\n\nTest 2: Using TensorPhysicsEngine")
    engine = TensorPhysicsEngine(np.stack(bodies), dt=0.001)
    
    state = engine.get_state()
    vel_before2 = state[1, 3:6]
    print(f"Before: vel={vel_before2}")
    
    engine.step()
    
    state2 = engine.get_state()
    vel_after2 = state2[1, 3:6]
    print(f"After: vel={vel_after2}")
    print(f"Change: {vel_after2 - vel_before2}")

if __name__ == "__main__":
    test_minimal()