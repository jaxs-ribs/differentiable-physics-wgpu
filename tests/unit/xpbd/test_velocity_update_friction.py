"""Tests for XPBD velocity update with friction."""
import pytest
import numpy as np
from tinygrad import Tensor, dtypes
from physics.xpbd.velocity_update import apply_friction


def test_apply_friction_no_friction():
    """Test that zero friction coefficient has no effect."""
    # Two bodies, one contact
    v = Tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    omega = Tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    
    # Contact data
    contacts = {
        'ids_a': Tensor([0], dtype=dtypes.int32),
        'ids_b': Tensor([1], dtype=dtypes.int32),
        'normal': Tensor([[1.0, 0.0, 0.0]]),
        'friction': Tensor([0.0])  # No friction
    }
    
    lambda_acc = Tensor([1.0])  # Some accumulated impulse
    
    v_new, omega_new = apply_friction(v, omega, contacts, lambda_acc)
    
    # Velocities should be unchanged
    assert np.allclose(v_new.numpy(), v.numpy())
    assert np.allclose(omega_new.numpy(), omega.numpy())


def test_apply_friction_static_case():
    """Test static friction preventing relative motion."""
    # Two bodies in contact, moving together
    v = Tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    omega = Tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    
    contacts = {
        'ids_a': Tensor([0], dtype=dtypes.int32),
        'ids_b': Tensor([1], dtype=dtypes.int32),
        'normal': Tensor([[0.0, 1.0, 0.0]]),  # Vertical contact
        'friction': Tensor([0.5])  # High friction
    }
    
    lambda_acc = Tensor([10.0])  # High normal force
    
    v_new, omega_new = apply_friction(v, omega, contacts, lambda_acc)
    
    # Relative tangential velocity is zero, so friction should do nothing
    assert np.allclose(v_new.numpy(), v.numpy())


def test_apply_friction_dynamic_case():
    """Test dynamic friction opposing relative motion."""
    # Body A sliding on Body B (which is stationary)
    v = Tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    omega = Tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    
    contacts = {
        'ids_a': Tensor([0], dtype=dtypes.int32),
        'ids_b': Tensor([1], dtype=dtypes.int32),
        'normal': Tensor([[0.0, 1.0, 0.0]]),  # Vertical contact
        'friction': Tensor([0.2])  # Dynamic friction
    }
    
    lambda_acc = Tensor([5.0])  # Normal force
    
    v_new, omega_new = apply_friction(v, omega, contacts, lambda_acc)
    
    # Friction should oppose the tangential velocity of body A
    # Tangential velocity is (1, 0, 0)
    # Friction force should be in (-1, 0, 0) direction
    
    # Body A's velocity should decrease in X
    assert v_new.numpy()[0, 0] < v.numpy()[0, 0]
    
    # Body B's velocity should increase in X (reaction force)
    assert v_new.numpy()[1, 0] > v.numpy()[1, 0]


def test_friction_magnitude_limit():
    """Test that friction force is limited by μ * N."""
    # Sliding with very high velocity
    v = Tensor([[100.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    omega = Tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    
    contacts = {
        'ids_a': Tensor([0], dtype=dtypes.int32),
        'ids_b': Tensor([1], dtype=dtypes.int32),
        'normal': Tensor([[0.0, 1.0, 0.0]]),
        'friction': Tensor([0.1])
    }
    
    lambda_acc = Tensor([2.0])  # Small normal force
    
    v_new, omega_new = apply_friction(v, omega, contacts, lambda_acc)
    
    # Change in velocity (impulse) should be limited
    delta_v_a = v_new.numpy()[0] - v.numpy()[0]
    
    # Friction impulse magnitude should be <= μ * normal_impulse
    friction_impulse_mag = np.linalg.norm(delta_v_a)
    max_friction_impulse = 0.1 * 2.0  # μ * λ
    
    # Allow for some tolerance due to implementation details
    assert friction_impulse_mag <= max_friction_impulse + 1e-5


def test_multiple_contacts_friction():
    """Test friction with multiple contacts."""
    # Body 0 sliding on two other bodies (1 and 2)
    v = Tensor([
        [1.0, 0.0, 0.0],  # Sliding body
        [0.0, 0.0, 0.0],  # Static body 1
        [0.0, 0.0, 0.0]   # Static body 2
    ])
    omega = Tensor([[0.0, 0.0, 0.0]] * 3)
    
    contacts = {
        'ids_a': Tensor([0, 0], dtype=dtypes.int32),
        'ids_b': Tensor([1, 2], dtype=dtypes.int32),
        'normal': Tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
        'friction': Tensor([0.2, 0.3])  # Different friction coeffs
    }
    
    lambda_acc = Tensor([5.0, 5.0])  # Equal normal forces
    
    v_new, omega_new = apply_friction(v, omega, contacts, lambda_acc)
    
    # Body 0 should slow down due to friction from both contacts
    assert v_new.numpy()[0, 0] < v.numpy()[0, 0]
    
    # Bodies 1 and 2 should gain some velocity
    assert v_new.numpy()[1, 0] > 0
    assert v_new.numpy()[2, 0] > 0


def test_friction_with_rotation():
    """Test friction involving rotational motion."""
    # Spinning sphere on a static plane
    v = Tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    omega = Tensor([
        [0.0, 0.0, 10.0],  # Sphere spinning around Z
        [0.0, 0.0, 0.0]    # Plane is static
    ])
    
    contacts = {
        'ids_a': Tensor([0], dtype=dtypes.int32),
        'ids_b': Tensor([1], dtype=dtypes.int32),
        'normal': Tensor([[0.0, 1.0, 0.0]]),
        'friction': Tensor([0.5])
    }
    
    lambda_acc = Tensor([10.0])
    
    v_new, omega_new = apply_friction(v, omega, contacts, lambda_acc)
    
    # Friction should oppose the spin
    # Sphere's angular velocity around Z should decrease
    assert abs(omega_new.numpy()[0, 2]) < abs(omega.numpy()[0, 2])
    
    # Friction should induce linear motion (reaction)
    # This part is complex and depends on contact point, etc.
    # Just check that some linear velocity is induced
    assert not np.allclose(v_new.numpy()[0], 0.0)


def test_invalid_contact_handling_friction():
    """Test that invalid contacts are ignored."""
    v = Tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    omega = Tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    
    contacts = {
        'ids_a': Tensor([-1], dtype=dtypes.int32),  # Invalid contact
        'ids_b': Tensor([-1], dtype=dtypes.int32),
        'normal': Tensor([[0.0, 1.0, 0.0]]),
        'friction': Tensor([0.5])
    }
    
    lambda_acc = Tensor([10.0])
    
    v_new, omega_new = apply_friction(v, omega, contacts, lambda_acc)
    
    # Velocities should be unchanged
    assert np.allclose(v_new.numpy(), v.numpy())
    assert np.allclose(omega_new.numpy(), omega.numpy())
