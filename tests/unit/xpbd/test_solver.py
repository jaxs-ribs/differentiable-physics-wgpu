"""Tests for XPBD constraint solver."""
import pytest
import numpy as np
from tinygrad import Tensor
from physics.xpbd.solver import solve_constraints


def test_solve_constraints_placeholder():
    """Test that solve_constraints function exists and can be called."""
    # Create dummy SoA inputs
    x = Tensor.zeros(5, 3)  # 5 bodies, 3D positions
    q = Tensor.zeros(5, 4)  # 5 bodies, quaternions
    contacts = Tensor.zeros(10, 10)  # 10 contacts with contact data
    inv_mass = Tensor.zeros(5)  # 5 inverse masses
    inv_inertia = Tensor.zeros(5, 3, 3)  # 5 inverse inertia tensors
    
    x_result, q_result = solve_constraints(x, q, contacts, inv_mass, inv_inertia, iterations=8)
    # Should return same positions and orientations (placeholder)
    assert x_result.shape == x.shape
    assert q_result.shape == q.shape


def test_solve_constraints_signature():
    """Test that function has correct signature."""
    import inspect
    sig = inspect.signature(solve_constraints)
    
    # Should have six parameters
    assert len(sig.parameters) == 6
    assert 'x' in sig.parameters
    assert 'q' in sig.parameters
    assert 'contacts' in sig.parameters
    assert 'inv_mass' in sig.parameters
    assert 'inv_inertia' in sig.parameters
    assert 'iterations' in sig.parameters
    
    # iterations should have default value
    assert sig.parameters['iterations'].default == 8


def test_solve_constraints_default_iterations():
    """Test that default iterations parameter works."""
    x = Tensor.zeros(3, 3)
    q = Tensor.zeros(3, 4)
    contacts = Tensor.zeros(5, 10)
    inv_mass = Tensor.zeros(3)
    inv_inertia = Tensor.zeros(3, 3, 3)
    
    # Should work with default iterations
    x_result, q_result = solve_constraints(x, q, contacts, inv_mass, inv_inertia)
    assert x_result.shape == x.shape
    assert q_result.shape == q.shape