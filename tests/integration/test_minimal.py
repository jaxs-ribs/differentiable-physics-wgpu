#!/usr/bin/env python3
"""Minimal test."""

import sys
import os

# Add parent directories to path to find test_setup
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_setup import setup_test_paths
setup_test_paths()

import numpy as np

os.environ['JIT'] = '0'

from physics.engine import TensorPhysicsEngine
from physics.types import create_body_array, ShapeType, BodySchema

def test_minimal():
    print("Creating bodies...")
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

    # Ball
    bodies.append(create_body_array(
        position=np.array([0., 2.0, 0.], dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1.0,
        inertia=np.eye(3, dtype=np.float32) * 0.1,
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([0.5, 0., 0.], dtype=np.float32)
    ))

    print("Creating engine...")
    engine = TensorPhysicsEngine(np.stack(bodies), dt=0.001)

    print("Running 5 steps...")
    for i in range(5):
        state = engine.get_state()
        print(f"Step {i}: y={state[1,1]:.3f}, vy={state[1,4]:.3f}")
        engine.step()

    print("Done")

if __name__ == "__main__":
    test_minimal()