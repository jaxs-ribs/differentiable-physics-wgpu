"""Tests for XPBD integration (forward prediction)."""
import pytest
import numpy as np
from tinygrad import Tensor
from physics.xpbd.integration import integrate


def test_integrate_placeholder():
    """Test that integrate function exists and can be called."""
    # Create dummy SoA inputs
    x = Tensor.zeros(5, 3)  # 5 bodies, 3D positions
    q = Tensor.zeros(5, 4)  # 5 bodies, quaternions
    v = Tensor.zeros(5, 3)  # 5 bodies, velocities
    omega = Tensor.zeros(5, 3)  # 5 bodies, angular velocities
    inv_mass = Tensor.zeros(5)  # 5 inverse masses
    dt = 0.016
    gravity = Tensor([0, -9.81, 0])
    
    x_pred, q_pred = integrate(x, q, v, omega, inv_mass, dt, gravity)
    # Should return same positions and orientations (placeholder)
    assert x_pred.shape == x.shape
    assert q_pred.shape == q.shape


def test_integrate_signature():
    """Test that function has correct signature."""
    import inspect
    sig = inspect.signature(integrate)
    
    # Should have seven parameters
    assert len(sig.parameters) == 7
    assert 'x' in sig.parameters
    assert 'q' in sig.parameters
    assert 'v' in sig.parameters
    assert 'omega' in sig.parameters
    assert 'inv_mass' in sig.parameters
    assert 'dt' in sig.parameters
    assert 'gravity' in sig.parameters