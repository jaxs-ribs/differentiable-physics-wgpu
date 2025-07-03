"""Tests for XPBD integration (forward prediction)."""
import pytest
import numpy as np
from tinygrad import Tensor
from physics.xpbd.integration import integrate, predict_state


def test_integrate_placeholder():
    """Test that integrate function exists and can be called."""
    # Create dummy SoA inputs
    x = Tensor.zeros(5, 3)  # 5 bodies, 3D positions
    q = Tensor.ones(5, 4)  # 5 bodies, quaternions (identity)
    q = q / (q * q).sum(axis=-1, keepdim=True).sqrt()  # Normalize
    v = Tensor.zeros(5, 3)  # 5 bodies, velocities
    omega = Tensor.zeros(5, 3)  # 5 bodies, angular velocities
    inv_mass = Tensor.zeros(5)  # 5 inverse masses
    dt = 0.016
    gravity = Tensor([0, -9.81, 0])
    
    x_pred, q_pred = integrate(x, q, v, omega, inv_mass, dt, gravity)
    # Should return valid tensors
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


def test_predict_state_exists():
    """Test that predict_state function exists and can be called."""
    # Create dummy SoA inputs
    x = Tensor.zeros(3, 3)  # 3 bodies, 3D positions
    q = Tensor([[1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0]])  # Identity quaternions
    v = Tensor.zeros(3, 3)  # 3 bodies, velocities
    omega = Tensor.zeros(3, 3)  # 3 bodies, angular velocities
    inv_mass = Tensor.ones(3)  # 3 inverse masses
    inv_inertia = Tensor.eye(3).unsqueeze(0).expand(3, -1, -1)
    dt = 0.016
    gravity = Tensor([0, -9.81, 0])
    
    x_pred, q_pred, v_new, omega_new = predict_state(x, q, v, omega, inv_mass, inv_inertia, gravity, dt)
    
    # Should return valid tensors
    assert x_pred.shape == x.shape
    assert q_pred.shape == q.shape
    assert v_new.shape == v.shape
    assert omega_new.shape == omega.shape


def test_predict_state_signature():
    """Test that predict_state has correct signature."""
    import inspect
    sig = inspect.signature(predict_state)
    
    # Should have eight parameters
    assert len(sig.parameters) == 8
    assert 'x' in sig.parameters
    assert 'q' in sig.parameters
    assert 'v' in sig.parameters
    assert 'omega' in sig.parameters
    assert 'inv_mass' in sig.parameters
    assert 'inv_inertia' in sig.parameters
    assert 'gravity' in sig.parameters
    assert 'dt' in sig.parameters