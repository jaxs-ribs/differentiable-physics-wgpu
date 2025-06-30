#!/usr/bin/env python3
"""Test if angular terms are causing the issue."""

import sys
import os

# Add parent directories to path to find test_setup
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.test_setup import setup_test_paths
setup_test_paths()

import numpy as np

os.environ['JIT'] = '0'

from physics.types import create_body_array, ShapeType, BodySchema
from physics.solver import resolve_collisions, cross_product
from physics.math_utils import get_world_inv_inertia
from tinygrad import Tensor

def test_angular():
    """Test angular term calculation."""
    print("\n=== Testing Angular Terms ===")
    
    bodies = []
    
    # Ground
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
        position=np.array([0., 1.0, 0.], dtype=np.float32),
        velocity=np.array([0., -10.0, 0.], dtype=np.float32),
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1.0,
        inertia=np.eye(3, dtype=np.float32) * 0.1,
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([0.5, 0., 0.], dtype=np.float32)
    ))
    
    bodies_t = Tensor(np.stack(bodies))
    
    # Manual calculation
    inv_mass_a = 1e-8
    inv_mass_b = 1.0
    
    # Contact data
    contact_point = np.array([0., 0.5, 0.])
    normal = np.array([0., -1., 0.])  # From ball to ground
    
    # Positions
    pos_a = np.array([0., 0., 0.])
    pos_b = np.array([0., 1., 0.])
    
    # r vectors
    r_a = contact_point - pos_a
    r_b = contact_point - pos_b
    
    print(f"Contact point: {contact_point}")
    print(f"r_a (ground): {r_a}")
    print(f"r_b (ball): {r_b}")
    
    # Inertias
    inv_I_a = np.eye(3) * 1e-8
    inv_I_b = np.eye(3) * 10.0  # 1/0.1
    
    # Cross products
    r_cross_n_a = np.cross(r_a, normal)
    r_cross_n_b = np.cross(r_b, normal)
    
    print(f"\nr × n:")
    print(f"  r_a × n: {r_cross_n_a}")
    print(f"  r_b × n: {r_cross_n_b}")
    
    # Angular deltas
    ang_delta_a = inv_I_a @ r_cross_n_a
    ang_delta_b = inv_I_b @ r_cross_n_b
    
    print(f"\nI⁻¹(r × n):")
    print(f"  Ground: {ang_delta_a}")
    print(f"  Ball: {ang_delta_b}")
    
    # Angular terms in denominator
    angular_term_a = np.dot(normal, np.cross(ang_delta_a, r_a))
    angular_term_b = np.dot(normal, np.cross(ang_delta_b, r_b))
    
    print(f"\nn·((I⁻¹(r × n)) × r):")
    print(f"  Ground: {angular_term_a}")
    print(f"  Ball: {angular_term_b}")
    
    # Total denominator
    linear_denom = inv_mass_a + inv_mass_b
    total_denom = linear_denom + angular_term_a + angular_term_b
    
    print(f"\nDenominator calculation:")
    print(f"  Linear part: {inv_mass_a} + {inv_mass_b} = {linear_denom}")
    print(f"  Angular part: {angular_term_a} + {angular_term_b} = {angular_term_a + angular_term_b}")
    print(f"  Total: {total_denom}")
    print(f"\nRatio of denominators: {total_denom / linear_denom:.3f}")

if __name__ == "__main__":
    test_angular()