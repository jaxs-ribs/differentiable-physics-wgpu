#!/usr/bin/env python3
"""
Test script for the new implementations: Broadphase and Rotational Dynamics
"""

import numpy as np
from reference import Body, PhysicsEngine, ShapeType

def test_broadphase():
    """Test Sweep and Prune broadphase implementation."""
    print("=== Testing Broadphase (Sweep and Prune) ===\n")
    
    engine = PhysicsEngine()
    
    # Create a scene with some overlapping and non-overlapping bodies
    bodies = [
        # Group 1: Three overlapping spheres
        Body(position=np.array([0.0, 0.0, 0.0]), velocity=np.zeros(3),
             orientation=np.array([1.0, 0.0, 0.0, 0.0]), angular_vel=np.zeros(3),
             mass=1.0, inertia=np.eye(3), shape_type=ShapeType.SPHERE,
             shape_params=np.array([1.0, 0.0, 0.0])),
        
        Body(position=np.array([1.5, 0.0, 0.0]), velocity=np.zeros(3),
             orientation=np.array([1.0, 0.0, 0.0, 0.0]), angular_vel=np.zeros(3),
             mass=1.0, inertia=np.eye(3), shape_type=ShapeType.SPHERE,
             shape_params=np.array([1.0, 0.0, 0.0])),
        
        Body(position=np.array([0.0, 1.5, 0.0]), velocity=np.zeros(3),
             orientation=np.array([1.0, 0.0, 0.0, 0.0]), angular_vel=np.zeros(3),
             mass=1.0, inertia=np.eye(3), shape_type=ShapeType.SPHERE,
             shape_params=np.array([1.0, 0.0, 0.0])),
        
        # Group 2: Isolated sphere far away
        Body(position=np.array([10.0, 0.0, 0.0]), velocity=np.zeros(3),
             orientation=np.array([1.0, 0.0, 0.0, 0.0]), angular_vel=np.zeros(3),
             mass=1.0, inertia=np.eye(3), shape_type=ShapeType.SPHERE,
             shape_params=np.array([1.0, 0.0, 0.0]))
    ]
    
    for body in bodies:
        engine.add_body(body)
    
    print(f"Created {len(bodies)} bodies")
    
    # Test broadphase
    print("\nComparing broadphase methods:")
    sap_pairs = engine.get_broadphase_pairs()
    brute_pairs = engine.get_all_pairs_bruteforce()
    
    print(f"  Sweep and Prune found: {sorted(sap_pairs)}")
    print(f"  Brute force found: {sorted(brute_pairs)}")
    print(f"  Results match: {sorted(sap_pairs) == sorted(brute_pairs)}")
    
    # Expected: (0,1), (0,2), (1,2) - three overlapping spheres
    expected = [(0, 1), (0, 2), (1, 2)]
    if sorted(sap_pairs) == expected:
        print("✓ Broadphase correctly identified overlapping pairs")
    else:
        print("✗ Broadphase results incorrect")
    
    return sorted(sap_pairs) == sorted(brute_pairs)

def test_rotational_dynamics():
    """Test rotational dynamics implementation."""
    print("\n=== Testing Rotational Dynamics ===\n")
    
    engine = PhysicsEngine(dt=0.01, gravity=np.array([0, 0, 0]))  # No gravity
    
    # Create two spheres for head-on collision
    sphere1 = Body(
        position=np.array([-2.0, 0.0, 0.0]),
        velocity=np.array([2.0, 0.0, 0.0]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_vel=np.zeros(3),
        mass=1.0,
        inertia=np.eye(3) * 0.4,  # Solid sphere inertia
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([1.0, 0.0, 0.0])
    )
    
    sphere2 = Body(
        position=np.array([2.0, 0.0, 0.0]),
        velocity=np.array([-2.0, 0.0, 0.0]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_vel=np.zeros(3),
        mass=1.0,
        inertia=np.eye(3) * 0.4,
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([1.0, 0.0, 0.0])
    )
    
    engine.add_body(sphere1)
    engine.add_body(sphere2)
    
    print("Initial state:")
    print(f"  Sphere 1: pos={sphere1.position}, vel={sphere1.velocity}")
    print(f"  Sphere 2: pos={sphere2.position}, vel={sphere2.velocity}")
    
    # Calculate initial momentum
    initial_momentum = sphere1.mass * sphere1.velocity + sphere2.mass * sphere2.velocity
    initial_energy = 0.5 * sphere1.mass * np.dot(sphere1.velocity, sphere1.velocity) + \
                    0.5 * sphere2.mass * np.dot(sphere2.velocity, sphere2.velocity)
    
    # Run simulation until collision
    for i in range(50):
        engine.step()
        # Check if collision occurred (velocities reversed)
        if sphere1.velocity[0] < 0 and sphere2.velocity[0] > 0:
            break
    
    print(f"\nAfter collision (step {i+1}):")
    print(f"  Sphere 1: pos={sphere1.position}, vel={sphere1.velocity}")
    print(f"  Sphere 2: pos={sphere2.position}, vel={sphere2.velocity}")
    
    # Check momentum conservation
    final_momentum = sphere1.mass * sphere1.velocity + sphere2.mass * sphere2.velocity
    momentum_error = np.linalg.norm(final_momentum - initial_momentum)
    
    print(f"\nMomentum conservation:")
    print(f"  Initial: {initial_momentum}")
    print(f"  Final: {final_momentum}")
    print(f"  Error: {momentum_error:.6f}")
    
    if momentum_error < 0.01:
        print("✓ Momentum conserved")
    else:
        print("✗ Momentum not conserved")
    
    # Check that collision occurred
    if sphere1.velocity[0] < 0 and sphere2.velocity[0] > 0:
        print("✓ Collision detected and resolved")
    else:
        print("✗ Collision not properly resolved")
    
    return momentum_error < 0.01

def test_angular_velocity_integration():
    """Test that angular velocity updates orientation correctly."""
    print("\n=== Testing Angular Velocity Integration ===\n")
    
    engine = PhysicsEngine(dt=0.01, gravity=np.array([0, 0, 0]))
    
    # Create a spinning box
    box = Body(
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.zeros(3),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),  # Identity quaternion
        angular_vel=np.array([0.0, 0.0, 1.0]),  # Spinning around Z axis
        mass=1.0,
        inertia=np.eye(3),
        shape_type=ShapeType.BOX,
        shape_params=np.array([1.0, 1.0, 1.0])
    )
    
    engine.add_body(box)
    
    initial_quat = box.orientation.copy()
    print(f"Initial orientation: {initial_quat}")
    
    # Run for 100 steps
    for _ in range(100):
        engine.step()
    
    final_quat = box.orientation
    print(f"Final orientation: {final_quat}")
    
    # Check that quaternion is still normalized
    quat_norm = np.linalg.norm(final_quat)
    print(f"Quaternion norm: {quat_norm:.6f}")
    
    # Check that orientation changed
    orientation_changed = not np.allclose(initial_quat, final_quat)
    
    if abs(quat_norm - 1.0) < 0.01:
        print("✓ Quaternion remains normalized")
    else:
        print("✗ Quaternion normalization error")
    
    if orientation_changed:
        print("✓ Angular velocity updates orientation")
    else:
        print("✗ Orientation not updated")
    
    return abs(quat_norm - 1.0) < 0.01 and orientation_changed

if __name__ == "__main__":
    print("Testing new physics engine implementations...\n")
    
    # Run tests
    broadphase_ok = test_broadphase()
    dynamics_ok = test_rotational_dynamics()
    angular_ok = test_angular_velocity_integration()
    
    print("\n" + "="*50)
    print("SUMMARY:")
    print(f"  Broadphase (SAP): {'✓ PASS' if broadphase_ok else '✗ FAIL'}")
    print(f"  Rotational Dynamics: {'✓ PASS' if dynamics_ok else '✗ FAIL'}")
    print(f"  Angular Integration: {'✓ PASS' if angular_ok else '✗ FAIL'}")
    print("="*50)