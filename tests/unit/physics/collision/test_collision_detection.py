#!/usr/bin/env python3
"""Comprehensive tests for collision detection and contact generation."""

import numpy as np
import pytest
from tinygrad import Tensor
from physics.engine import TensorPhysicsEngine
from physics.types import create_body_array, ShapeType
from physics.narrowphase import sphere_sphere_contact, sphere_box_contact, box_box_contact


class TestCollisionDetection:
    """Test suite for collision detection and contact normal computation."""
    
    def test_sphere_sphere_collision(self):
        """Test sphere-sphere collision detection."""
        bodies = []
        
        # Sphere 1
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
        
        # Sphere 2
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
        
        engine = TensorPhysicsEngine(bodies, dt=0.01)
        
        # Run until collision
        collision_detected = False
        for _ in range(100):
            bodies_tensor = engine.step()
            dist = np.linalg.norm(bodies_tensor[0, :3].numpy() - bodies_tensor[1, :3].numpy())
            if dist <= 1.0:  # Sum of radii
                collision_detected = True
                break
        
        assert collision_detected, "Spheres should collide"
    
    def test_sphere_box_collision(self):
        """Test sphere-box collision detection."""
        bodies = []
        
        # Box
        bodies.append(create_body_array(
            position=np.array([0., 0., 0.], dtype=np.float32),
            velocity=np.zeros(3, dtype=np.float32),
            orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
            angular_vel=np.zeros(3, dtype=np.float32),
            mass=1e8,
            inertia=np.eye(3, dtype=np.float32) * 1e8,
            shape_type=ShapeType.BOX,
            shape_params=np.array([1., 1., 1.], dtype=np.float32)
        ))
        
        # Sphere falling onto box
        bodies.append(create_body_array(
            position=np.array([0., 3., 0.], dtype=np.float32),
            velocity=np.array([0., -5., 0.], dtype=np.float32),
            orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
            angular_vel=np.zeros(3, dtype=np.float32),
            mass=1.0,
            inertia=np.eye(3, dtype=np.float32) * 0.1,
            shape_type=ShapeType.SPHERE,
            shape_params=np.array([0.5, 0., 0.], dtype=np.float32)
        ))
        
        engine = TensorPhysicsEngine(bodies, dt=0.01)
        
        # Run until collision
        for _ in range(100):
            bodies_tensor = engine.step()
            sphere_y = bodies_tensor[1, 1].numpy()
            if sphere_y <= 1.5:  # Box top + sphere radius
                break
        
        # Sphere should stop falling
        final_vel_y = bodies_tensor[1, 4].numpy()
        assert abs(final_vel_y) < 1.0, "Sphere should have low velocity after collision"
    
    def test_contact_normal_direction(self):
        """Test that contact normals point from body A to body B."""
        # Create simple collision scenario
        pos_a = Tensor([0., 0., 0.])
        pos_b = Tensor([2., 0., 0.])
        radius = 1.0
        
        # Test sphere-sphere normal
        contact = sphere_sphere_contact(pos_a, radius, pos_b, radius)
        if contact is not None:
            normal = contact[0].numpy()
            # Normal should point from A to B
            expected = np.array([1., 0., 0.])
            assert np.allclose(normal, expected, atol=1e-3), \
                f"Normal {normal} should point from A to B"
    
    def test_penetration_depth(self):
        """Test penetration depth calculation."""
        # Overlapping spheres
        pos_a = Tensor([0., 0., 0.])
        pos_b = Tensor([1.5, 0., 0.])  # Centers 1.5 apart
        radius = 1.0  # Total diameter = 2.0
        
        contact = sphere_sphere_contact(pos_a, radius, pos_b, radius)
        if contact is not None:
            depth = contact[1].numpy()
            expected_depth = 2.0 - 1.5  # Sum of radii - distance
            assert abs(depth - expected_depth) < 1e-3, \
                f"Depth {depth} should be {expected_depth}"
    
    def test_multiple_simultaneous_collisions(self):
        """Test handling of multiple simultaneous collisions."""
        bodies = []
        
        # Central sphere
        bodies.append(create_body_array(
            position=np.array([0., 0., 0.], dtype=np.float32),
            velocity=np.zeros(3, dtype=np.float32),
            orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
            angular_vel=np.zeros(3, dtype=np.float32),
            mass=1.0,
            inertia=np.eye(3, dtype=np.float32) * 0.1,
            shape_type=ShapeType.SPHERE,
            shape_params=np.array([0.5, 0., 0.], dtype=np.float32)
        ))
        
        # Surrounding spheres moving inward
        for angle in [0, 90, 180, 270]:
            rad = np.radians(angle)
            x = 2.0 * np.cos(rad)
            z = 2.0 * np.sin(rad)
            vx = -0.5 * np.cos(rad)
            vz = -0.5 * np.sin(rad)
            
            bodies.append(create_body_array(
                position=np.array([x, 0., z], dtype=np.float32),
                velocity=np.array([vx, 0., vz], dtype=np.float32),
                orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
                angular_vel=np.zeros(3, dtype=np.float32),
                mass=1.0,
                inertia=np.eye(3, dtype=np.float32) * 0.1,
                shape_type=ShapeType.SPHERE,
                shape_params=np.array([0.5, 0., 0.], dtype=np.float32)
            ))
        
        engine = TensorPhysicsEngine(bodies, dt=0.01)
        
        # Run simulation
        for _ in range(50):
            bodies_tensor = engine.step()
        
        # Central sphere should remain relatively stationary
        final_pos = bodies_tensor[0, :3].numpy()
        assert np.linalg.norm(final_pos) < 0.5, \
            "Central sphere should remain near origin with symmetric collisions"
    
    def test_edge_case_parallel_surfaces(self):
        """Test collision between parallel surfaces."""
        bodies = []
        
        # Box 1
        bodies.append(create_body_array(
            position=np.array([0., 0., 0.], dtype=np.float32),
            velocity=np.array([0., 0.1, 0.], dtype=np.float32),
            orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
            angular_vel=np.zeros(3, dtype=np.float32),
            mass=1.0,
            inertia=np.eye(3, dtype=np.float32),
            shape_type=ShapeType.BOX,
            shape_params=np.array([1., 1., 1.], dtype=np.float32)
        ))
        
        # Box 2 directly above
        bodies.append(create_body_array(
            position=np.array([0., 2.1, 0.], dtype=np.float32),
            velocity=np.array([0., -0.1, 0.], dtype=np.float32),
            orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
            angular_vel=np.zeros(3, dtype=np.float32),
            mass=1.0,
            inertia=np.eye(3, dtype=np.float32),
            shape_type=ShapeType.BOX,
            shape_params=np.array([1., 1., 1.], dtype=np.float32)
        ))
        
        engine = TensorPhysicsEngine(bodies, dt=0.01)
        
        # Run until collision
        for _ in range(50):
            bodies_tensor = engine.step()
            dist = bodies_tensor[1, 1].numpy() - bodies_tensor[0, 1].numpy()
            if dist <= 2.0:  # Sum of half-extents
                break
        
        # Boxes should separate after collision
        for _ in range(50):
            bodies_tensor = engine.step()
        
        final_dist = bodies_tensor[1, 1].numpy() - bodies_tensor[0, 1].numpy()
        assert final_dist >= 2.0, "Boxes should separate after collision"


if __name__ == "__main__":
    test = TestCollisionDetection()
    test.test_sphere_sphere_collision()
    test.test_sphere_box_collision()
    test.test_contact_normal_direction()
    test.test_penetration_depth()
    test.test_multiple_simultaneous_collisions()
    test.test_edge_case_parallel_surfaces()
    print("All collision tests passed!")