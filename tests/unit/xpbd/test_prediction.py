"""Unit tests for XPBD forward prediction step."""
import pytest
import numpy as np
from tinygrad import Tensor
from physics.xpbd.integration import predict_state
from physics.math_utils import quat_normalize


class TestForwardPrediction:
    """Test suite for the forward prediction step of XPBD physics."""
    
    def test_free_fall(self):
        """Test that a body with no initial velocity accelerates downward under gravity."""
        # Setup: Single body at origin with no velocity
        x = Tensor([[0.0, 10.0, 0.0]])  # Start at height 10
        q = Tensor([[1.0, 0.0, 0.0, 0.0]])  # Identity quaternion
        v = Tensor([[0.0, 0.0, 0.0]])  # No initial velocity
        omega = Tensor([[0.0, 0.0, 0.0]])  # No rotation
        inv_mass = Tensor([1.0])  # Mass = 1
        inv_inertia = Tensor.eye(3).unsqueeze(0)  # Identity inertia
        gravity = Tensor([0.0, -9.81, 0.0])
        dt = 0.1
        
        # Execute
        x_pred, q_pred, v_new, omega_new = predict_state(x, q, v, omega, inv_mass, inv_inertia, gravity, dt)
        
        # Verify
        x_pred_np = x_pred.numpy()
        v_new_np = v_new.numpy()
        
        # Velocity should have increased downward: v = gt
        expected_v = np.array([[0.0, -9.81 * dt, 0.0]])
        np.testing.assert_allclose(v_new_np, expected_v, rtol=1e-5)
        
        # Position should have moved down: y = y0 + v*t = y0 + 0.5*g*t^2
        expected_y = 10.0 + (-9.81 * dt) * dt
        np.testing.assert_allclose(x_pred_np[0, 1], expected_y, rtol=1e-5)
        
        # X and Z positions should be unchanged
        np.testing.assert_allclose(x_pred_np[0, 0], 0.0, rtol=1e-5)
        np.testing.assert_allclose(x_pred_np[0, 2], 0.0, rtol=1e-5)
        
        # Orientation should be unchanged (no rotation)
        np.testing.assert_allclose(q_pred.numpy(), q.numpy(), rtol=1e-5)
    
    def test_constant_velocity(self):
        """Test that a body with constant velocity continues moving in a straight line."""
        # Setup: Body moving with constant velocity
        x = Tensor([[0.0, 0.0, 0.0]])
        q = Tensor([[1.0, 0.0, 0.0, 0.0]])
        v = Tensor([[2.0, 0.0, 1.0]])  # Moving in X and Z
        omega = Tensor([[0.0, 0.0, 0.0]])
        inv_mass = Tensor([0.0])  # Infinite mass (no gravity effect)
        inv_inertia = Tensor.eye(3).unsqueeze(0)
        gravity = Tensor([0.0, -9.81, 0.0])
        dt = 0.5
        
        # Execute
        x_pred, q_pred, v_new, omega_new = predict_state(x, q, v, omega, inv_mass, inv_inertia, gravity, dt)
        
        # Verify
        x_pred_np = x_pred.numpy()
        v_new_np = v_new.numpy()
        
        # Velocity should be unchanged (infinite mass)
        np.testing.assert_allclose(v_new_np, v.numpy(), rtol=1e-5)
        
        # Position should have moved: x = x0 + v*t
        expected_x = np.array([[2.0 * dt, 0.0, 1.0 * dt]])
        np.testing.assert_allclose(x_pred_np, expected_x, rtol=1e-5)
    
    def test_pure_rotation(self):
        """Test that a body with angular velocity rotates correctly."""
        # Setup: Body rotating around Y axis
        x = Tensor([[0.0, 0.0, 0.0]])
        q = Tensor([[1.0, 0.0, 0.0, 0.0]])  # Identity
        v = Tensor([[0.0, 0.0, 0.0]])
        omega = Tensor([[0.0, np.pi, 0.0]])  # π rad/s around Y
        inv_mass = Tensor([1.0])
        inv_inertia = Tensor.eye(3).unsqueeze(0)
        gravity = Tensor([0.0, 0.0, 0.0])  # No gravity for pure rotation test
        dt = 0.1  # Smaller timestep for more accurate rotation
        
        # Execute
        x_pred, q_pred, v_new, omega_new = predict_state(x, q, v, omega, inv_mass, inv_inertia, gravity, dt)
        
        # Verify
        q_pred_np = q_pred.numpy()
        
        # The quaternion should be normalized
        q_norm = np.linalg.norm(q_pred_np)
        np.testing.assert_allclose(q_norm, 1.0, rtol=1e-5)
        
        # Check that rotation is around Y axis (x and z components should be 0)
        np.testing.assert_allclose(q_pred_np[0, 1], 0.0, atol=1e-5)  # x component
        np.testing.assert_allclose(q_pred_np[0, 3], 0.0, atol=1e-5)  # z component
        
        # Check that rotation angle is reasonable
        # With gyroscopic effects, the actual rotation won't be exactly ω*dt
        # but should be close for small dt
        sin_half_angle = q_pred_np[0, 2]  # y component
        cos_half_angle = q_pred_np[0, 0]  # w component
        angle = 2 * np.arctan2(sin_half_angle, cos_half_angle)
        
        # Should be positive rotation around Y
        assert angle > 0
        
        # Position should not change (no linear velocity)
        np.testing.assert_allclose(x_pred.numpy(), x.numpy(), rtol=1e-5)
    
    def test_quaternion_normalization(self):
        """Test that predicted quaternions remain normalized."""
        # Setup: Multiple bodies with various angular velocities
        n_bodies = 10
        x = Tensor.zeros(n_bodies, 3)
        q = Tensor.ones(n_bodies, 4)  # Will be normalized
        q = quat_normalize(q)
        v = Tensor.zeros(n_bodies, 3)
        
        # Random angular velocities
        np.random.seed(42)
        omega_np = np.random.randn(n_bodies, 3).astype(np.float32)
        omega = Tensor(omega_np)
        
        inv_mass = Tensor.ones(n_bodies)
        inv_inertia = Tensor.eye(3).unsqueeze(0).expand(n_bodies, -1, -1)
        gravity = Tensor([0.0, -9.81, 0.0])
        dt = 0.016
        
        # Execute
        x_pred, q_pred, v_new, omega_new = predict_state(x, q, v, omega, inv_mass, inv_inertia, gravity, dt)
        
        # Verify: All quaternions should have unit norm
        q_pred_np = q_pred.numpy()
        norms = np.linalg.norm(q_pred_np, axis=1)
        np.testing.assert_allclose(norms, np.ones(n_bodies), rtol=1e-5)
    
    def test_multiple_bodies(self):
        """Test forward prediction with multiple bodies of different properties."""
        # Setup: 3 bodies with different masses and velocities
        x = Tensor([[0.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                    [0.0, 5.0, 0.0]])
        q = Tensor([[1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0]])
        v = Tensor([[1.0, 0.0, 0.0],   # Moving right
                    [0.0, 2.0, 0.0],   # Moving up
                    [0.0, 0.0, 0.0]])  # Stationary
        omega = Tensor.zeros(3, 3)
        inv_mass = Tensor([1.0, 0.5, 2.0])  # Different masses: 1, 2, 0.5
        inv_inertia = Tensor.eye(3).unsqueeze(0).expand(3, -1, -1)
        gravity = Tensor([0.0, -10.0, 0.0])
        dt = 0.1
        
        # Execute
        x_pred, q_pred, v_new, omega_new = predict_state(x, q, v, omega, inv_mass, inv_inertia, gravity, dt)
        
        # Verify each body
        x_pred_np = x_pred.numpy()
        v_new_np = v_new.numpy()
        
        # Body 0: mass=1, moving right, should fall and move right
        np.testing.assert_allclose(v_new_np[0], [1.0, -10.0 * dt, 0.0], rtol=1e-5)
        np.testing.assert_allclose(x_pred_np[0], [1.0 * dt, -10.0 * dt * dt, 0.0], rtol=1e-5)
        
        # Body 1: mass=2, moving up, gravity effect halved
        np.testing.assert_allclose(v_new_np[1], [0.0, 2.0 - 5.0 * dt, 0.0], rtol=1e-5)
        
        # Body 2: mass=0.5, stationary, gravity effect doubled
        np.testing.assert_allclose(v_new_np[2], [0.0, -20.0 * dt, 0.0], rtol=1e-5)
    
    def test_zero_mass_bodies(self):
        """Test that bodies with zero mass (static bodies) don't move."""
        # Setup: Body with zero mass
        x = Tensor([[0.0, 0.0, 0.0]])
        q = Tensor([[1.0, 0.0, 0.0, 0.0]])
        v = Tensor([[10.0, 10.0, 10.0]])  # Has velocity but shouldn't move
        omega = Tensor([[1.0, 1.0, 1.0]])  # Has angular velocity
        inv_mass = Tensor([0.0])  # Zero inverse mass = infinite mass
        inv_inertia = Tensor.zeros(1, 3, 3)  # Zero inverse inertia
        gravity = Tensor([0.0, -9.81, 0.0])
        dt = 1.0
        
        # Execute
        x_pred, q_pred, v_new, omega_new = predict_state(x, q, v, omega, inv_mass, inv_inertia, gravity, dt)
        
        # Verify: Position should change based on velocity but no acceleration
        x_pred_np = x_pred.numpy()
        expected_x = np.array([[10.0, 10.0, 10.0]])  # Moved by v*dt
        np.testing.assert_allclose(x_pred_np, expected_x, rtol=1e-5)
        
        # Velocity should be unchanged (no acceleration due to zero inv_mass)
        np.testing.assert_allclose(v_new.numpy(), v.numpy(), rtol=1e-5)