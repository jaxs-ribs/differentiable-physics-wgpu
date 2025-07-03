"""Tests for XPBD narrowphase collision detection."""
import pytest
import numpy as np
from tinygrad import Tensor
from physics.xpbd.narrowphase import generate_contacts


def test_generate_contacts_placeholder():
    """Test that generate_contacts function exists and can be called."""
    # Create dummy SoA inputs
    x = Tensor.zeros(5, 3)  # 5 bodies, 3D positions
    # Create identity quaternions properly
    q = Tensor([[1.0, 0.0, 0.0, 0.0]] * 5)  # 5 identity quaternions
    candidate_pairs = Tensor.full((10, 2), -1)  # 10 invalid pairs
    shape_type = Tensor.zeros(5)  # 5 shape types
    shape_params = Tensor.zeros(5, 3)  # 5 bodies, 3 shape parameters
    
    result = generate_contacts(x, q, candidate_pairs, shape_type, shape_params)
    # Should return dictionary with empty arrays
    assert isinstance(result, dict)
    assert 'ids_a' in result
    assert 'ids_b' in result
    assert 'normal' in result
    assert 'p' in result
    assert 'compliance' in result


def test_generate_contacts_signature():
    """Test that function has correct signature."""
    import inspect
    sig = inspect.signature(generate_contacts)
    
    # Should have five parameters
    assert len(sig.parameters) == 5
    assert 'x' in sig.parameters
    assert 'q' in sig.parameters
    assert 'candidate_pairs' in sig.parameters
    assert 'shape_type' in sig.parameters
    assert 'shape_params' in sig.parameters