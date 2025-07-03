"""Tests for XPBD broadphase collision detection."""
import pytest
import numpy as np
from tinygrad import Tensor
from physics.xpbd.broadphase import uniform_spatial_hash


def test_uniform_spatial_hash_placeholder():
    """Test that uniform_spatial_hash function exists and can be called."""
    # Create dummy bodies tensor
    bodies = Tensor.zeros(5, 27)  # 5 bodies, 27 properties each
    
    try:
        result = uniform_spatial_hash(bodies)
        # Should return None from pass statement
        assert result is None
    except Exception as e:
        # Expected - function is not implemented yet
        assert "TODO" in str(e) or "NotImplementedError" in str(e)


def test_uniform_spatial_hash_signature():
    """Test that function has correct signature."""
    import inspect
    sig = inspect.signature(uniform_spatial_hash)
    
    # Should have one parameter: bodies
    assert len(sig.parameters) == 1
    assert 'bodies' in sig.parameters