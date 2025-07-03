"""Tests for XPBD broadphase collision detection."""
import pytest
import numpy as np
from tinygrad import Tensor
from physics.xpbd.broadphase import uniform_spatial_hash


def test_uniform_spatial_hash_placeholder():
    """Test that uniform_spatial_hash function exists and can be called."""
    # Create dummy SoA tensors
    x = Tensor.zeros(5, 3)  # 5 bodies, 3D positions
    shape_type = Tensor.zeros(5)  # 5 shape types
    shape_params = Tensor.zeros(5, 3)  # 5 bodies, 3 shape parameters
    
    result = uniform_spatial_hash(x, shape_type, shape_params)
    # Should return empty candidate pairs tensor
    assert result.shape == (0, 2)


def test_uniform_spatial_hash_signature():
    """Test that function has correct signature."""
    import inspect
    sig = inspect.signature(uniform_spatial_hash)
    
    # Should have three parameters: x, shape_type, shape_params
    assert len(sig.parameters) == 3
    assert 'x' in sig.parameters
    assert 'shape_type' in sig.parameters
    assert 'shape_params' in sig.parameters