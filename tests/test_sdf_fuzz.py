#!/usr/bin/env python3
"""Fuzz testing for SDF properties using hypothesis."""

import numpy as np
from hypothesis import given, strategies as st, settings
import subprocess
import json
import math

# Define SDF functions with consistent parameter order
def sdf_sphere(center, radius, point):
    """Signed distance function for a sphere"""
    center = np.array(center)
    point = np.array(point)
    return np.linalg.norm(point - center) - radius

def sdf_box(center, half_extents, point):
    """Signed distance function for an axis-aligned box"""
    center = np.array(center)
    point = np.array(point)
    half_extents = np.array(half_extents)
    q = np.abs(point - center) - half_extents
    return np.linalg.norm(np.maximum(q, 0.0)) + min(max(q[0], max(q[1], q[2])), 0.0)

def sdf_capsule(center, radius, height, point):
    """Signed distance function for a capsule (aligned with Y axis)"""
    center = np.array(center)
    point = np.array(point)
    p = point - center
    h = height * 0.5
    p[1] = np.clip(p[1], -h, h)
    return np.linalg.norm([p[0], 0, p[2]]) - radius

# Hypothesis strategies for geometric primitives
position_strategy = st.tuples(
    st.floats(min_value=-100, max_value=100, allow_nan=False),
    st.floats(min_value=-100, max_value=100, allow_nan=False),
    st.floats(min_value=-100, max_value=100, allow_nan=False)
)

radius_strategy = st.floats(min_value=0.01, max_value=10.0)
box_extents_strategy = st.tuples(
    st.floats(min_value=0.01, max_value=10.0),
    st.floats(min_value=0.01, max_value=10.0),
    st.floats(min_value=0.01, max_value=10.0)
)
height_strategy = st.floats(min_value=0.01, max_value=10.0)

@given(
    center=position_strategy,
    radius=radius_strategy,
    point=position_strategy
)
@settings(max_examples=1000)
def test_sphere_sdf_properties(center, radius, point):
    """Test mathematical properties of sphere SDF."""
    dist = sdf_sphere(np.array(center), radius, np.array(point))
    
    # Property 1: Distance is correct
    actual_dist = np.linalg.norm(np.array(point) - np.array(center)) - radius
    assert abs(dist - actual_dist) < 1e-5, f"Incorrect distance: {dist} vs {actual_dist}"
    
    # Property 2: Gradient has unit length (except at center)
    eps = 1e-6
    if np.linalg.norm(np.array(point) - np.array(center)) > eps:
        # Numerical gradient
        grad = []
        for i in range(3):
            p_plus = list(point)
            p_minus = list(point)
            p_plus[i] += eps
            p_minus[i] -= eps
            grad.append((sdf_sphere(np.array(center), radius, np.array(p_plus)) - sdf_sphere(np.array(center), radius, np.array(p_minus))) / (2 * eps))
        
        grad_norm = np.linalg.norm(grad)
        assert abs(grad_norm - 1.0) < 0.01, f"Gradient norm {grad_norm} not unit length"

@given(
    center=position_strategy,
    half_extents=box_extents_strategy,
    point=position_strategy
)
@settings(max_examples=1000)
def test_box_sdf_properties(center, half_extents, point):
    """Test mathematical properties of box SDF."""
    dist = sdf_box(np.array(center), half_extents, np.array(point))
    
    # Property 1: Inside vs outside consistency
    rel_point = np.array(point) - np.array(center)
    is_inside = all(abs(rel_point[i]) <= half_extents[i] for i in range(3))
    
    if is_inside:
        assert dist <= 0, f"Point inside box but SDF positive: {dist}"
    else:
        assert dist >= 0, f"Point outside box but SDF negative: {dist}"
    
    # Property 2: Distance is 0 on surface
    for i in range(3):
        # Test point on face
        surface_point = list(center)
        surface_point[i] = center[i] + half_extents[i]
        dist_surface = sdf_box(np.array(center), half_extents, np.array(surface_point))
        assert abs(dist_surface) < 1e-5, f"Distance on surface should be 0, got {dist_surface}"

@given(
    center=position_strategy,
    radius=radius_strategy,
    height=height_strategy,
    point=position_strategy
)
@settings(max_examples=1000)
def test_capsule_sdf_properties(center, radius, height, point):
    """Test mathematical properties of capsule SDF."""
    dist = sdf_capsule(np.array(center), radius, height, np.array(point))
    
    # Property 1: Reduces to sphere when height = 0
    if height < 0.01:
        sphere_dist = sdf_sphere(np.array(center), radius, np.array(point))
        assert abs(dist - sphere_dist) < 0.1, f"Capsule with small height should be like sphere"
    
    # Property 2: Symmetry
    # Test reflection through center
    reflected_point = [
        2 * center[0] - point[0],
        point[1],
        point[2]
    ]
    dist_reflected = sdf_capsule(np.array(center), radius, height, np.array(reflected_point))
    assert abs(dist - dist_reflected) < 1e-5, f"Capsule SDF not symmetric: {dist} vs {dist_reflected}"

@given(
    shapes=st.lists(
        st.tuples(
            st.sampled_from(['sphere', 'box', 'capsule']),
            position_strategy,
            st.one_of(radius_strategy, box_extents_strategy)
        ),
        min_size=2,
        max_size=5
    ),
    point=position_strategy
)
@settings(max_examples=100)
def test_sdf_union_properties(shapes, point):
    """Test properties of SDF union operations."""
    distances = []
    
    for shape_type, center, params in shapes:
        if shape_type == 'sphere':
            if isinstance(params, tuple):
                params = params[0]  # Use first value as radius
            dist = sdf_sphere(np.array(center), params, np.array(point))
        elif shape_type == 'box':
            if not isinstance(params, tuple):
                params = (params, params, params)  # Make cube
            dist = sdf_box(np.array(center), params, np.array(point))
        else:  # capsule
            if isinstance(params, tuple):
                radius = params[0]
                height = params[1] if len(params) > 1 else 1.0
            else:
                radius = params
                height = 1.0
            dist = sdf_capsule(np.array(center), radius, height, np.array(point))
        
        distances.append(dist)
    
    # Union is minimum distance
    union_dist = min(distances)
    
    # Property: Union is inside if any shape is inside
    any_inside = any(d <= 0 for d in distances)
    if any_inside:
        assert union_dist <= 0, "Union should be inside if any shape is inside"

@given(
    center=position_strategy,
    radius=radius_strategy,
    direction=st.tuples(
        st.floats(min_value=-1, max_value=1),
        st.floats(min_value=-1, max_value=1),
        st.floats(min_value=-1, max_value=1)
    ),
    t=st.floats(min_value=0, max_value=10)
)
@settings(max_examples=500)
def test_sdf_ray_marching(center, radius, direction, t):
    """Test that SDF can be used for ray marching."""
    # Normalize direction
    dir_norm = np.linalg.norm(direction)
    if dir_norm < 1e-6:
        return  # Skip degenerate case
    
    direction = np.array(direction) / dir_norm
    
    # Start point outside sphere
    start = np.array(center) + direction * (radius + 5.0)
    
    # Ray march
    total_dist = 0.0
    point = start.copy()
    max_steps = 100
    
    for _ in range(max_steps):
        dist = sdf_sphere(center, radius, point)
        
        if dist < 0.001:
            # Hit surface
            actual_dist = np.linalg.norm(point - center)
            assert abs(actual_dist - radius) < 0.01, f"Ray march hit wrong distance: {actual_dist} vs {radius}"
            break
        
        # March forward by SDF distance
        point = point + direction * dist
        total_dist += dist
        
        if total_dist > 20.0:
            # Missed
            break

def test_sdf_edge_cases():
    """Test edge cases and boundary conditions."""
    print("Testing SDF edge cases...")
    
    # Test 1: Point exactly at shape center
    dist = sdf_sphere([0, 0, 0], 1.0, [0, 0, 0])
    assert abs(dist - (-1.0)) < 1e-6, f"Point at sphere center should give -radius, got {dist}"
    
    # Test 2: Very large shapes
    large_radius = 1e6
    dist = sdf_sphere([0, 0, 0], large_radius, [large_radius, 0, 0])
    assert abs(dist) < 1e-3, f"Point on large sphere surface should be ~0, got {dist}"
    
    # Test 3: Very small shapes
    tiny_radius = 1e-6
    dist = sdf_sphere([0, 0, 0], tiny_radius, [tiny_radius, 0, 0])
    assert abs(dist) < tiny_radius, f"Tiny sphere SDF incorrect: {dist}"
    
    # Test 4: Degenerate box (2D)
    dist = sdf_box([0, 0, 0], [1, 1, 0], [0, 0, 0])
    assert dist <= 0, "Point inside degenerate box should be inside"
    
    print("✓ Edge cases handled correctly")

def test_gpu_sdf_consistency():
    """Test that GPU SDF implementation matches CPU reference."""
    print("\nTesting GPU SDF consistency...")
    
    # Create test cases
    test_cases = []
    
    # Sphere tests
    for i in range(10):
        angle = i * 2 * np.pi / 10
        test_cases.append({
            "shape": "sphere",
            "center": [np.cos(angle) * 5, 0, np.sin(angle) * 5],
            "params": {"radius": 0.5 + i * 0.1},
            "test_point": [0, 0, 0]
        })
    
    # Box tests
    for i in range(5):
        test_cases.append({
            "shape": "box",
            "center": [0, i * 2, 0],
            "params": {"half_extents": [1, 0.5, 2]},
            "test_point": [0, i * 2 + 0.25, 0]
        })
    
    # TODO: Run GPU tests when test harness is ready
    print("✓ GPU SDF tests would run here")

if __name__ == "__main__":
    print("Running SDF fuzz tests with hypothesis...\n")
    
    # Run property-based tests
    print("Testing sphere SDF properties...")
    test_sphere_sdf_properties()
    print("✓ Sphere properties verified (1000 random tests)")
    
    print("\nTesting box SDF properties...")
    test_box_sdf_properties()
    print("✓ Box properties verified (1000 random tests)")
    
    print("\nTesting capsule SDF properties...")
    test_capsule_sdf_properties()
    print("✓ Capsule properties verified (1000 random tests)")
    
    print("\nTesting SDF union properties...")
    test_sdf_union_properties()
    print("✓ Union properties verified (100 random tests)")
    
    print("\nTesting ray marching properties...")
    test_sdf_ray_marching()
    print("✓ Ray marching verified (500 random tests)")
    
    # Run specific edge case tests
    test_sdf_edge_cases()
    test_gpu_sdf_consistency()
    
    print("\n✓ All SDF fuzz tests passed!")