#!/usr/bin/env python3
"""Comprehensive tests for impulse calculation and resolution."""

import numpy as np
import pytest
from tinygrad import Tensor
from physics.engine import TensorPhysicsEngine
from physics.types import create_body_array, ShapeType


class TestImpulseResolution:
    """Test suite for impulse-based collision resolution."""
    
    def test_basic_impulse_calculation(self):
        """Test basic impulse magnitude for simple collision."""
        bodies = []
        
        # Static ground
        bodies.append(create_body_array(
            position=np.array([0., -1., 0.], dtype=np.float32),
            velocity=np.zeros(3, dtype=np.float32),
            orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
            angular_vel=np.zeros(3, dtype=np.float32),
            mass=1e8,
            inertia=np.eye(3, dtype=np.float32) * 1e8,
            shape_type=ShapeType.BOX,
            shape_params=np.array([10., 0.5, 10.], dtype=np.float32)
        ))
        
        # Falling sphere
        bodies.append(create_body_array(
            position=np.array([0., 1., 0.], dtype=np.float32),
            velocity=np.array([0., -5., 0.], dtype=np.float32),
            orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
            angular_vel=np.zeros(3, dtype=np.float32),
            mass=1.0,
            inertia=np.eye(3, dtype=np.float32) * 0.1,
            shape_type=ShapeType.SPHERE,
            shape_params=np.array([0.5, 0., 0.], dtype=np.float32)
        ))
        
        engine = TensorPhysicsEngine(bodies, dt=0.01, restitution=0.0)
        
        # Run until collision and check velocity change
        initial_momentum = -5.0  # mass * velocity
        for _ in range(100):
            bodies_tensor = engine.step()
            vel_y = bodies_tensor[1, 4].numpy()
            
            # After collision with restitution=0, velocity should be ~0
            if bodies_tensor[1, 1].numpy() <= 0.0:  # Near ground
                assert abs(vel_y) < 0.5, f"Velocity should be near zero, got {vel_y}"
                break
    
    def test_impulse_direction(self):
        """Test that impulses are applied in correct directions."""
        bodies = []
        
        # Two spheres colliding head-on
        bodies.append(create_body_array(
            position=np.array([-2., 0., 0.], dtype=np.float32),
            velocity=np.array([3., 0., 0.], dtype=np.float32),
            orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
            angular_vel=np.zeros(3, dtype=np.float32),
            mass=1.0,
            inertia=np.eye(3, dtype=np.float32) * 0.1,
            shape_type=ShapeType.SPHERE,
            shape_params=np.array([0.5, 0., 0.], dtype=np.float32)
        ))
        
        bodies.append(create_body_array(
            position=np.array([2., 0., 0.], dtype=np.float32),
            velocity=np.array([-3., 0., 0.], dtype=np.float32),
            orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
            angular_vel=np.zeros(3, dtype=np.float32),
            mass=1.0,
            inertia=np.eye(3, dtype=np.float32) * 0.1,
            shape_type=ShapeType.SPHERE,
            shape_params=np.array([0.5, 0., 0.], dtype=np.float32)
        ))
        
        engine = TensorPhysicsEngine(bodies, dt=0.01, restitution=0.5)
        
        # Run until after collision
        for _ in range(100):
            bodies_tensor = engine.step()
            dist = abs(bodies_tensor[0, 0].numpy() - bodies_tensor[1, 0].numpy())
            
            if dist < 2.0:  # Collision occurred
                # Run a few more steps
                for _ in range(10):
                    bodies_tensor = engine.step()
                
                # Velocities should have reversed
                vel1_x = bodies_tensor[0, 3].numpy()
                vel2_x = bodies_tensor[1, 3].numpy()
                
                assert vel1_x < 0, "First sphere should move left after collision"
                assert vel2_x > 0, "Second sphere should move right after collision"
                break
    
    def test_conservation_of_momentum(self):
        """Test momentum conservation in collisions."""
        bodies = []
        
        # Moving sphere
        m1 = 2.0
        v1 = 4.0
        bodies.append(create_body_array(
            position=np.array([-2., 0., 0.], dtype=np.float32),
            velocity=np.array([v1, 0., 0.], dtype=np.float32),
            orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
            angular_vel=np.zeros(3, dtype=np.float32),
            mass=m1,
            inertia=np.eye(3, dtype=np.float32) * 0.1,
            shape_type=ShapeType.SPHERE,
            shape_params=np.array([0.5, 0., 0.], dtype=np.float32)
        ))
        
        # Stationary sphere
        m2 = 1.0
        v2 = 0.0
        bodies.append(create_body_array(
            position=np.array([0., 0., 0.], dtype=np.float32),
            velocity=np.array([v2, 0., 0.], dtype=np.float32),
            orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
            angular_vel=np.zeros(3, dtype=np.float32),
            mass=m2,
            inertia=np.eye(3, dtype=np.float32) * 0.1,
            shape_type=ShapeType.SPHERE,
            shape_params=np.array([0.5, 0., 0.], dtype=np.float32)
        ))
        
        initial_momentum = m1 * v1 + m2 * v2
        
        engine = TensorPhysicsEngine(bodies, dt=0.01, restitution=0.8)
        
        # Run simulation
        for i in range(200):
            bodies_tensor = engine.step()
            
            # Calculate total momentum
            p1 = bodies_tensor[0, 3].numpy() * m1
            p2 = bodies_tensor[1, 3].numpy() * m2
            total_momentum = p1 + p2
            
            # Momentum should be conserved (within numerical tolerance)
            assert abs(total_momentum - initial_momentum) < 0.1, \
                f"Momentum not conserved: {total_momentum} vs {initial_momentum}"
    
    def test_angular_impulse(self):
        """Test off-center collisions generate angular velocity."""
        bodies = []
        
        # Static ground
        bodies.append(create_body_array(
            position=np.array([0., -2., 0.], dtype=np.float32),
            velocity=np.zeros(3, dtype=np.float32),
            orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
            angular_vel=np.zeros(3, dtype=np.float32),
            mass=1e8,
            inertia=np.eye(3, dtype=np.float32) * 1e8,
            shape_type=ShapeType.BOX,
            shape_params=np.array([10., 0.5, 10.], dtype=np.float32)
        ))
        
        # Box falling with horizontal velocity (will hit corner)
        bodies.append(create_body_array(
            position=np.array([0., 1., 0.], dtype=np.float32),
            velocity=np.array([2., -3., 0.], dtype=np.float32),
            orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
            angular_vel=np.zeros(3, dtype=np.float32),
            mass=1.0,
            inertia=np.eye(3, dtype=np.float32) * 0.1,
            shape_type=ShapeType.BOX,
            shape_params=np.array([0.5, 0.5, 0.5], dtype=np.float32)
        ))
        
        engine = TensorPhysicsEngine(bodies, dt=0.01)
        
        # Run until after collision
        for _ in range(100):
            bodies_tensor = engine.step()
            if bodies_tensor[1, 1].numpy() < -0.5:
                break
        
        # Box should have some angular velocity after off-center collision
        angular_vel = bodies_tensor[1, 9:12].numpy()
        angular_speed = np.linalg.norm(angular_vel)
        
        assert angular_speed > 0.1, \
            f"Off-center collision should generate angular velocity, got {angular_speed}"
    
    def test_coefficient_of_restitution(self):
        """Test different restitution values produce correct relative velocities."""
        for e in [0.0, 0.5, 1.0]:
            bodies = []
            
            # Two equal mass spheres
            bodies.append(create_body_array(
                position=np.array([-1., 0., 0.], dtype=np.float32),
                velocity=np.array([2., 0., 0.], dtype=np.float32),
                orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
                angular_vel=np.zeros(3, dtype=np.float32),
                mass=1.0,
                inertia=np.eye(3, dtype=np.float32) * 0.1,
                shape_type=ShapeType.SPHERE,
                shape_params=np.array([0.5, 0., 0.], dtype=np.float32)
            ))
            
            bodies.append(create_body_array(
                position=np.array([1., 0., 0.], dtype=np.float32),
                velocity=np.array([-2., 0., 0.], dtype=np.float32),
                orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
                angular_vel=np.zeros(3, dtype=np.float32),
                mass=1.0,
                inertia=np.eye(3, dtype=np.float32) * 0.1,
                shape_type=ShapeType.SPHERE,
                shape_params=np.array([0.5, 0., 0.], dtype=np.float32)
            ))
            
            initial_relative_vel = 4.0  # 2 - (-2)
            
            engine = TensorPhysicsEngine(bodies, dt=0.005, restitution=e)
            
            # Run until after collision
            for _ in range(200):
                bodies_tensor = engine.step()
                dist = abs(bodies_tensor[0, 0].numpy() - bodies_tensor[1, 0].numpy())
                
                if dist > 2.0:  # Separated after collision
                    v1 = bodies_tensor[0, 3].numpy()
                    v2 = bodies_tensor[1, 3].numpy()
                    final_relative_vel = abs(v1 - v2)
                    
                    expected = e * initial_relative_vel
                    assert abs(final_relative_vel - expected) < 0.5, \
                        f"e={e}: relative velocity {final_relative_vel} != {expected}"
                    break


if __name__ == "__main__":
    test = TestImpulseResolution()
    test.test_basic_impulse_calculation()
    test.test_impulse_direction()
    test.test_conservation_of_momentum()
    test.test_angular_impulse()
    test.test_coefficient_of_restitution()
    print("All impulse tests passed!")