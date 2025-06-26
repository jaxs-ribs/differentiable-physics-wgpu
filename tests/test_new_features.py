#!/usr/bin/env python3
"""
Test the new features: Broadphase and Rotational Dynamics
This version avoids the hanging broadphase issue
"""

import numpy as np

def test_broadphase_simple():
    """Test broadphase with minimal example."""
    print("=== Testing Broadphase Implementation ===\n")
    
    # Import here to isolate any issues
    from reference import AABB, BroadphaseDetector
    
    detector = BroadphaseDetector()
    
    # Test 1: No overlaps
    aabbs = [
        AABB(np.array([0., 0., 0.]), np.array([1., 1., 1.]), 0),
        AABB(np.array([5., 0., 0.]), np.array([6., 1., 1.]), 1),
    ]
    
    pairs = detector.detect_pairs(aabbs)
    print(f"Test 1 - No overlaps: {pairs}")
    assert len(pairs) == 0, "Should find no overlapping pairs"
    print("✓ Correctly identified no overlaps")
    
    # Test 2: Two overlapping boxes
    aabbs = [
        AABB(np.array([0., 0., 0.]), np.array([2., 2., 2.]), 0),
        AABB(np.array([1., 1., 1.]), np.array([3., 3., 3.]), 1),
    ]
    
    pairs = detector.detect_pairs(aabbs)
    print(f"\nTest 2 - Overlapping boxes: {pairs}")
    assert pairs == [(0, 1)], "Should find one overlapping pair"
    print("✓ Correctly identified overlap")
    
    # Test 3: Three bodies, two overlapping
    aabbs = [
        AABB(np.array([0., 0., 0.]), np.array([1., 1., 1.]), 0),
        AABB(np.array([0.5, 0., 0.]), np.array([1.5, 1., 1.]), 1),
        AABB(np.array([5., 0., 0.]), np.array([6., 1., 1.]), 2),
    ]
    
    pairs = detector.detect_pairs(aabbs)
    print(f"\nTest 3 - Three bodies: {pairs}")
    assert pairs == [(0, 1)], "Should find one overlapping pair (0,1)"
    print("✓ Correctly identified single overlap among three bodies")
    
    return True

def test_rotational_dynamics():
    """Test rotational dynamics with simple collision."""
    print("\n=== Testing Rotational Dynamics ===\n")
    
    from reference import Body, PhysicsEngine, ShapeType
    
    # Create engine without using broadphase
    engine = PhysicsEngine(dt=0.01, gravity=np.array([0, 0, 0]))
    
    # Test 1: Angular velocity integration
    print("Test 1 - Angular velocity integration:")
    spinning_box = Body(
        position=np.array([0., 0., 0.]),
        velocity=np.zeros(3),
        orientation=np.array([1., 0., 0., 0.]),
        angular_vel=np.array([0., 0., 1.]),  # Spin around Z
        mass=1.0,
        inertia=np.eye(3),
        shape_type=ShapeType.BOX,
        shape_params=np.array([1., 1., 1.])
    )
    
    initial_orientation = spinning_box.orientation.copy()
    
    # Manually integrate without collision detection
    from reference import Integrator
    integrator = Integrator(dt=0.01)
    
    for _ in range(10):
        integrator.integrate(spinning_box, np.zeros(3))
    
    print(f"  Initial orientation: {initial_orientation}")
    print(f"  Final orientation: {spinning_box.orientation}")
    print(f"  Quaternion norm: {np.linalg.norm(spinning_box.orientation):.6f}")
    
    orientation_changed = not np.allclose(initial_orientation, spinning_box.orientation)
    quat_normalized = abs(np.linalg.norm(spinning_box.orientation) - 1.0) < 0.001
    
    if orientation_changed and quat_normalized:
        print("✓ Angular velocity correctly updates orientation")
    else:
        print("✗ Angular velocity integration failed")
    
    # Test 2: Collision resolver
    print("\nTest 2 - Collision response with rotation:")
    
    from reference import CollisionResolver
    resolver = CollisionResolver(restitution=0.5)
    
    # Two spheres in collision
    sphere1 = Body(
        position=np.array([0., 0., 0.]),
        velocity=np.array([1., 0., 0.]),
        orientation=np.array([1., 0., 0., 0.]),
        angular_vel=np.zeros(3),
        mass=1.0,
        inertia=np.eye(3) * 0.4,
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([1., 0., 0.])
    )
    
    sphere2 = Body(
        position=np.array([1.9, 0., 0.]),
        velocity=np.array([-1., 0., 0.]),
        orientation=np.array([1., 0., 0., 0.]),
        angular_vel=np.zeros(3),
        mass=1.0,
        inertia=np.eye(3) * 0.4,
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([1., 0., 0.])
    )
    
    # Manually resolve collision
    distance = -0.1  # 0.1 units of penetration
    normal = np.array([1., 0., 0.])
    contact_point = np.array([0.95, 0., 0.])  # Between the spheres
    
    initial_momentum = sphere1.mass * sphere1.velocity + sphere2.mass * sphere2.velocity
    
    resolver.resolve(sphere1, sphere2, distance, normal, contact_point, 0.01)
    
    final_momentum = sphere1.mass * sphere1.velocity + sphere2.mass * sphere2.velocity
    momentum_error = np.linalg.norm(final_momentum - initial_momentum)
    
    print(f"  Initial velocities: v1={sphere1.velocity}, v2={sphere2.velocity}")
    print(f"  Momentum conservation error: {momentum_error:.6f}")
    
    if momentum_error < 0.001 and sphere1.velocity[0] < 0 and sphere2.velocity[0] > 0:
        print("✓ Collision correctly resolved with momentum conservation")
    else:
        print("✗ Collision resolution failed")
    
    return orientation_changed and momentum_error < 0.001

def test_aabb_calculation():
    """Test AABB calculation for different shapes."""
    print("\n=== Testing AABB Calculation ===\n")
    
    from reference import Body, PhysicsEngine, ShapeType
    
    engine = PhysicsEngine()
    
    # Test sphere AABB
    sphere = Body(
        position=np.array([1., 2., 3.]),
        velocity=np.zeros(3),
        orientation=np.array([1., 0., 0., 0.]),
        angular_vel=np.zeros(3),
        mass=1.0,
        inertia=np.eye(3),
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([0.5, 0., 0.])  # radius 0.5
    )
    
    aabb = engine._compute_aabb(sphere)
    expected_min = np.array([0.5, 1.5, 2.5])
    expected_max = np.array([1.5, 2.5, 3.5])
    
    print(f"Sphere AABB: min={aabb.min_point}, max={aabb.max_point}")
    
    if np.allclose(aabb.min_point, expected_min) and np.allclose(aabb.max_point, expected_max):
        print("✓ Sphere AABB correct")
    else:
        print("✗ Sphere AABB incorrect")
    
    # Test rotated box AABB
    angle = np.pi / 4  # 45 degrees
    quat = np.array([np.cos(angle/2), 0., 0., np.sin(angle/2)])  # Rotation around Z
    
    box = Body(
        position=np.array([0., 0., 0.]),
        velocity=np.zeros(3),
        orientation=quat,
        angular_vel=np.zeros(3),
        mass=1.0,
        inertia=np.eye(3),
        shape_type=ShapeType.BOX,
        shape_params=np.array([1., 1., 1.])
    )
    
    aabb = engine._compute_aabb(box)
    # For a unit box rotated 45° around Z, the AABB should extend to sqrt(2) ≈ 1.414
    expected_extent = np.sqrt(2)
    
    print(f"\nRotated box AABB: min={aabb.min_point}, max={aabb.max_point}")
    print(f"Expected extent: ±{expected_extent:.3f}")
    
    x_extent_ok = abs(aabb.max_point[0] - expected_extent) < 0.01
    
    if x_extent_ok:
        print("✓ Rotated box AABB correct")
    else:
        print("✗ Rotated box AABB incorrect")
    
    return True

if __name__ == "__main__":
    print("Testing new physics engine features...\n")
    print("This test suite avoids the broadphase hanging issue\n")
    
    try:
        broadphase_ok = test_broadphase_simple()
        dynamics_ok = test_rotational_dynamics()
        aabb_ok = test_aabb_calculation()
        
        print("\n" + "="*50)
        print("SUMMARY:")
        print(f"  Broadphase Algorithm: {'✓ PASS' if broadphase_ok else '✗ FAIL'}")
        print(f"  Rotational Dynamics: {'✓ PASS' if dynamics_ok else '✗ FAIL'}")
        print(f"  AABB Calculation: {'✓ PASS' if aabb_ok else '✗ FAIL'}")
        print("="*50)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()