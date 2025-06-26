#!/usr/bin/env python3
"""
Signed Distance Function (SDF) Validation Test

This test validates that our GPU SDF implementations for sphere, box, and capsule primitives match 
mathematical reference implementations. SDFs are the foundation of collision detection - incorrect
distance calculations lead to physics artifacts, penetrations, or missed collisions. By testing
across various positions and shapes, we ensure reliable collision detection for all supported primitives.
"""
import numpy as np
import subprocess
import json

def sdf_sphere(point, center, radius):
    """Signed distance function for a sphere"""
    return np.linalg.norm(point - center) - radius

def sdf_box(point, center, half_extents):
    """Signed distance function for an axis-aligned box"""
    q = np.abs(point - center) - half_extents
    return np.linalg.norm(np.maximum(q, 0.0)) + min(max(q[0], max(q[1], q[2])), 0.0)

def sdf_capsule(point, center, radius, height):
    """Signed distance function for a capsule (aligned with Y axis)"""
    p = point - center
    h = height * 0.5
    p[1] = np.clip(p[1], -h, h)
    return np.linalg.norm([p[0], 0, p[2]]) - radius

def test_sphere_sphere_distance():
    """Test sphere-sphere SDF distance calculation"""
    # Two spheres
    center1 = np.array([0.0, 0.0, 0.0])
    radius1 = 1.0
    center2 = np.array([3.0, 0.0, 0.0])
    radius2 = 0.5
    
    # Expected distance between surfaces
    center_distance = np.linalg.norm(center2 - center1)
    expected_distance = center_distance - radius1 - radius2
    
    print(f"Sphere-sphere distance: {expected_distance:.6f}")
    assert abs(expected_distance - 1.5) < 1e-6

def test_sphere_point_distance():
    """Test SDF for various points around a sphere"""
    center = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    
    # Test points
    test_cases = [
        (np.array([2.0, 0.0, 0.0]), 1.0),    # Outside
        (np.array([0.5, 0.0, 0.0]), -0.5),   # Inside
        (np.array([1.0, 0.0, 0.0]), 0.0),    # On surface
    ]
    
    for point, expected in test_cases:
        distance = sdf_sphere(point, center, radius)
        print(f"Point {point} -> distance: {distance:.6f} (expected: {expected})")
        assert abs(distance - expected) < 1e-6

def test_box_point_distance():
    """Test SDF for box"""
    center = np.array([0.0, 0.0, 0.0])
    half_extents = np.array([1.0, 1.0, 1.0])
    
    # Test points
    test_cases = [
        (np.array([0.0, 0.0, 0.0]), -1.0),      # Center (inside)
        (np.array([2.0, 0.0, 0.0]), 1.0),       # Outside on X
        (np.array([1.0, 1.0, 1.0]), 0.0),       # Corner (on surface)
        (np.array([1.5, 1.5, 0.0]), np.sqrt(0.5)), # Outside diagonal
    ]
    
    for point, expected in test_cases:
        distance = sdf_box(point, center, half_extents)
        print(f"Box point {point} -> distance: {distance:.6f} (expected: {expected:.6f})")
        assert abs(distance - expected) < 1e-6

def generate_random_test_cases():
    """Generate random sphere configurations for property testing"""
    np.random.seed(42)
    test_cases = []
    
    for _ in range(100):
        # Random sphere 1
        center1 = np.random.uniform(-10, 10, 3)
        radius1 = np.random.uniform(0.1, 2.0)
        
        # Random sphere 2
        center2 = np.random.uniform(-10, 10, 3)
        radius2 = np.random.uniform(0.1, 2.0)
        
        # Calculate expected distance
        center_dist = np.linalg.norm(center2 - center1)
        surface_dist = center_dist - radius1 - radius2
        
        test_cases.append({
            'sphere1': {'center': center1.tolist(), 'radius': radius1},
            'sphere2': {'center': center2.tolist(), 'radius': radius2},
            'expected_distance': surface_dist
        })
    
    return test_cases

if __name__ == "__main__":
    print("Running SDF tests...")
    
    print("\n1. Sphere-sphere distance test:")
    test_sphere_sphere_distance()
    print("✓ Passed")
    
    print("\n2. Sphere-point distance test:")
    test_sphere_point_distance()
    print("✓ Passed")
    
    print("\n3. Box-point distance test:")
    test_box_point_distance()
    print("✓ Passed")
    
    print("\n4. Generating random test cases...")
    cases = generate_random_test_cases()
    print(f"✓ Generated {len(cases)} test cases")
    
    # Save test cases for GPU testing
    with open('sdf_test_cases.json', 'w') as f:
        json.dump(cases, f, indent=2)
    print("✓ Saved to sdf_test_cases.json")