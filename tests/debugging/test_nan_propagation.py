#!/usr/bin/env python3
"""Test NaN propagation in tensor operations."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tinygrad import Tensor
import numpy as np

def test_where_with_nan():
    """Test how Tensor.where handles NaN values."""
    print("Testing Tensor.where with NaN values...")
    
    # Create test data
    cond = Tensor([True, False, True])
    x = Tensor([1.0, 2.0, float('nan')])
    y = Tensor([10.0, 20.0, 30.0])
    
    # Test where
    result = x.where(cond, y)
    print(f"Condition: {cond.numpy()}")
    print(f"x (if true): {x.numpy()}")
    print(f"y (if false): {y.numpy()}")
    print(f"Result: {result.numpy()}")
    
    # Test with all NaN
    x_nan = Tensor([float('nan'), float('nan'), float('nan')])
    result_nan = x_nan.where(cond, y)
    print(f"\nWith all NaN in x: {result_nan.numpy()}")
    
    # Test arithmetic with NaN
    print("\nTesting arithmetic with NaN:")
    a = Tensor([1.0, 2.0, float('nan')])
    b = Tensor([10.0, 20.0, 30.0])
    
    print(f"a + b = {(a + b).numpy()}")
    print(f"a * 0 = {(a * 0).numpy()}")
    print(f"a * 1 = {(a * 1).numpy()}")

def test_integration_issue():
    """Test the specific integration issue."""
    print("\n\nTesting integration-like operation:")
    
    # Simulate the integration update
    pos = Tensor([[0.0, 1.0, 0.0], [float('nan'), float('nan'), float('nan')]])
    vel = Tensor([[0.0, 1.0, 0.0], [10.0, 20.0, 30.0]])
    is_dynamic = Tensor([[True], [False]])
    dt = 0.016
    
    print(f"pos: {pos.numpy()}")
    print(f"vel: {vel.numpy()}")
    print(f"is_dynamic: {is_dynamic.numpy()}")
    
    # Update position like in integration
    new_pos = pos.where(is_dynamic, pos + vel * dt)
    print(f"new_pos: {new_pos.numpy()}")
    
    # What if we flip the condition?
    new_pos2 = (pos + vel * dt).where(is_dynamic, pos)
    print(f"new_pos2 (flipped): {new_pos2.numpy()}")

def test_detach_assignment():
    """Test detach and assignment pattern."""
    print("\n\nTesting detach and assignment:")
    
    # Create initial tensor
    bodies = Tensor([[1.0, 2.0, 3.0, 4.0],
                     [5.0, 6.0, 7.0, 8.0]])
    
    print(f"Original: {bodies.numpy()}")
    
    # Detach and modify
    new_bodies = bodies.detach()
    print(f"After detach: {new_bodies.numpy()}")
    
    # Assign slice
    new_bodies[:, 1:3] = Tensor([[10.0, 11.0], [12.0, 13.0]])
    print(f"After assignment: {new_bodies.numpy()}")
    
    # What happens with NaN?
    new_bodies[:, 0:1] = Tensor([[float('nan')], [float('nan')]])
    print(f"After NaN assignment: {new_bodies.numpy()}")

if __name__ == "__main__":
    test_where_with_nan()
    test_integration_issue()
    test_detach_assignment()