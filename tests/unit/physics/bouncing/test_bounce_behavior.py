#!/usr/bin/env python3
"""Comprehensive tests for bounce behavior and restitution."""

import numpy as np
import pytest
from tinygrad import Tensor
from physics.engine import TensorPhysicsEngine
from physics.types import create_body_array, ShapeType


class TestBounceBehavior:
    """Test suite for bounce physics and restitution."""
    
    def test_basic_bounce(self):
        """Test basic ball bouncing off ground."""
        bodies = []
        
        # Ground box
        bodies.append(create_body_array(
            position=np.array([0., -2., 0.], dtype=np.float32),
            velocity=np.zeros(3, dtype=np.float32),
            orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
            angular_vel=np.zeros(3, dtype=np.float32),
            mass=1e8,  # Static
            inertia=np.eye(3, dtype=np.float32) * 1e8,
            shape_type=ShapeType.BOX,
            shape_params=np.array([10., 0.5, 10.], dtype=np.float32)
        ))
        
        # Falling ball
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
        
        engine = TensorPhysicsEngine(bodies, dt=0.01, restitution=0.5)
        
        # Run until collision
        for _ in range(50):
            bodies_tensor = engine.step()
            vel_y = bodies_tensor[1, 4].numpy()
            if vel_y > 0:  # Ball bounced
                assert vel_y < 5.0, "Ball should lose energy on bounce"
                assert vel_y > 0.1, "Ball should have some upward velocity"
                break
    
    def test_restitution_values(self):
        """Test different restitution coefficients."""
        for restitution in [0.0, 0.5, 0.9]:
            bodies = []
            
            # Ground
            bodies.append(create_body_array(
                position=np.array([0., 0., 0.], dtype=np.float32),
                velocity=np.zeros(3, dtype=np.float32),
                orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
                angular_vel=np.zeros(3, dtype=np.float32),
                mass=1e8,
                inertia=np.eye(3, dtype=np.float32) * 1e8,
                shape_type=ShapeType.BOX,
                shape_params=np.array([10., 0.5, 10.], dtype=np.float32)
            ))
            
            # Ball
            initial_vel = -5.0
            bodies.append(create_body_array(
                position=np.array([0., 1., 0.], dtype=np.float32),
                velocity=np.array([0., initial_vel, 0.], dtype=np.float32),
                orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
                angular_vel=np.zeros(3, dtype=np.float32),
                mass=1.0,
                inertia=np.eye(3, dtype=np.float32) * 0.1,
                shape_type=ShapeType.SPHERE,
                shape_params=np.array([0.5, 0., 0.], dtype=np.float32)
            ))
            
            engine = TensorPhysicsEngine(bodies, dt=0.01, restitution=restitution)
            
            # Run until bounce
            max_vel_after_bounce = 0.0
            for _ in range(100):
                bodies_tensor = engine.step()
                vel_y = bodies_tensor[1, 4].numpy()
                if vel_y > 0:
                    max_vel_after_bounce = max(max_vel_after_bounce, vel_y)
                    
            # Check restitution behavior
            expected_vel = abs(initial_vel) * restitution
            assert abs(max_vel_after_bounce - expected_vel) < 0.5, \
                f"Restitution {restitution}: expected ~{expected_vel}, got {max_vel_after_bounce}"
    
    def test_multiple_bounces(self):
        """Test energy dissipation over multiple bounces."""
        bodies = []
        
        # Ground
        bodies.append(create_body_array(
            position=np.array([0., 0., 0.], dtype=np.float32),
            velocity=np.zeros(3, dtype=np.float32),
            orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
            angular_vel=np.zeros(3, dtype=np.float32),
            mass=1e8,
            inertia=np.eye(3, dtype=np.float32) * 1e8,
            shape_type=ShapeType.BOX,
            shape_params=np.array([10., 0.5, 10.], dtype=np.float32)
        ))
        
        # Ball
        bodies.append(create_body_array(
            position=np.array([0., 5., 0.], dtype=np.float32),
            velocity=np.zeros(3, dtype=np.float32),
            orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
            angular_vel=np.zeros(3, dtype=np.float32),
            mass=1.0,
            inertia=np.eye(3, dtype=np.float32) * 0.1,
            shape_type=ShapeType.SPHERE,
            shape_params=np.array([0.5, 0., 0.], dtype=np.float32)
        ))
        
        engine = TensorPhysicsEngine(bodies, dt=0.01, restitution=0.7)
        
        # Track peak heights
        peak_heights = []
        last_vel = 0.0
        current_peak = 5.0
        
        for _ in range(1000):
            bodies_tensor = engine.step()
            pos_y = bodies_tensor[1, 1].numpy()
            vel_y = bodies_tensor[1, 4].numpy()
            
            # Detect peak (velocity changes from positive to negative)
            if last_vel > 0 and vel_y <= 0:
                peak_heights.append(pos_y)
                current_peak = pos_y
                
            last_vel = vel_y
            
            if len(peak_heights) >= 5:
                break
        
        # Each bounce should be lower than the last
        for i in range(1, len(peak_heights)):
            assert peak_heights[i] < peak_heights[i-1], \
                f"Peak {i} ({peak_heights[i]}) should be lower than peak {i-1} ({peak_heights[i-1]})"
    
    def test_angled_bounce(self):
        """Test ball bouncing at an angle."""
        bodies = []
        
        # Tilted ground
        bodies.append(create_body_array(
            position=np.array([0., -2., 0.], dtype=np.float32),
            velocity=np.zeros(3, dtype=np.float32),
            orientation=np.array([0.9659, 0.2588, 0., 0.], dtype=np.float32),  # 30 degree tilt
            angular_vel=np.zeros(3, dtype=np.float32),
            mass=1e8,
            inertia=np.eye(3, dtype=np.float32) * 1e8,
            shape_type=ShapeType.BOX,
            shape_params=np.array([10., 0.5, 10.], dtype=np.float32)
        ))
        
        # Ball falling straight down
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
        
        engine = TensorPhysicsEngine(bodies, dt=0.01, restitution=0.8)
        
        # Run until bounce
        for _ in range(100):
            bodies_tensor = engine.step()
            vel = bodies_tensor[1, 3:6].numpy()
            
            # After bounce, should have horizontal component
            if vel[1] > 0:  # Bounced
                assert abs(vel[0]) > 0.1, "Ball should have horizontal velocity after angled bounce"
                break


if __name__ == "__main__":
    test = TestBounceBehavior()
    test.test_basic_bounce()
    test.test_restitution_values()
    test.test_multiple_bounces()
    test.test_angled_bounce()
    print("All bounce tests passed!")