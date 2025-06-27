#!/usr/bin/env python3
"""Test solver behavior with empty contacts."""

import sys
import os
# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

import numpy as np
from physics.types import BodySchema
from physics.solver import resolve_collisions
from tinygrad import Tensor, dtypes

def test_empty_contacts():
    """Test what happens when resolve_collisions is called with no contacts."""
    print("Testing resolve_collisions with empty contacts...")
    
    # Create simple bodies
    bodies = np.zeros((2, BodySchema.NUM_PROPERTIES), dtype=np.float32)
    bodies[0, BodySchema.QUAT_W] = 1.0
    bodies[1, BodySchema.QUAT_W] = 1.0
    bodies[1, BodySchema.INV_MASS] = 1.0
    
    bodies_tensor = Tensor(bodies)
    
    # Create empty contact data (what narrowphase returns when no contacts)
    pair_indices = Tensor.zeros((0, 2), dtype=dtypes.int32)  # Empty pairs
    contact_normals = Tensor.zeros((0, 3))
    contact_depths = Tensor.zeros((0,))
    contact_points = Tensor.zeros((0, 3))
    contact_mask = Tensor.zeros((0,), dtype=dtypes.bool)
    
    print(f"Input shapes:")
    print(f"  pair_indices: {pair_indices.shape}")
    print(f"  contact_normals: {contact_normals.shape}")
    print(f"  contact_depths: {contact_depths.shape}")
    print(f"  contact_points: {contact_points.shape}")
    print(f"  contact_mask: {contact_mask.shape}")
    
    try:
        # Call resolve_collisions
        result = resolve_collisions(
            bodies_tensor, pair_indices, contact_normals, contact_depths,
            contact_points, contact_mask, restitution=0.1
        )
        
        print("\nResult:")
        print(f"  Shape: {result.shape}")
        print(f"  Body 1 pos: {result[1, BodySchema.POS_X:BodySchema.POS_Z+1].numpy()}")
        
        # Check for NaN
        if np.any(np.isnan(result.numpy())):
            print("\n!!! NaN detected in result !!!")
            # Find which properties have NaN
            for i in range(result.shape[0]):
                for j in range(result.shape[1]):
                    val = result[i, j].numpy()
                    if np.isnan(val):
                        print(f"  NaN at body {i}, property {j}")
                        
    except Exception as e:
        print(f"\nException: {e}")
        import traceback
        traceback.print_exc()

def test_gather_with_empty():
    """Test tensor gather with empty indices."""
    print("\n\nTesting gather with empty indices...")
    
    bodies = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    indices = Tensor.zeros((0,))  # Empty indices
    
    print(f"Bodies shape: {bodies.shape}")
    print(f"Indices shape: {indices.shape}")
    
    try:
        # Try to expand empty indices
        indices_expanded = indices.unsqueeze(1).expand(-1, bodies.shape[1])
        print(f"Expanded indices shape: {indices_expanded.shape}")
        
        # Try gather
        result = bodies.gather(0, indices_expanded)
        print(f"Gather result shape: {result.shape}")
        print(f"Gather result: {result.numpy()}")
    except Exception as e:
        print(f"Exception during gather: {e}")

if __name__ == "__main__":
    test_empty_contacts()
    test_gather_with_empty()