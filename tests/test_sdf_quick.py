#!/usr/bin/env python3
"""Quick SDF validation test that runs faster than the full fuzz test"""

import numpy as np
from reference import Body, PhysicsEngine

def create_body(shape_type, position, orientation, shape_params):
    """Create a Body object for testing."""
    return Body(
        position=np.array(position),
        velocity=np.zeros(3),
        orientation=orientation,
        angular_vel=np.zeros(3),
        mass=1.0,
        inertia=np.eye(3),
        shape_type=shape_type,
        shape_params=np.array(shape_params)
    )

print("Running quick SDF validation tests...\n")

engine = PhysicsEngine()
test_count = 0
passed_count = 0

# Test all shape combinations
test_cases = [
    ("Sphere-Sphere", 0, 0, [1.0, 0, 0], [1.0, 0, 0]),
    ("Sphere-Capsule", 0, 1, [1.0, 0, 0], [0.5, 2.0, 0]),
    ("Sphere-Box", 0, 2, [1.0, 0, 0], [1.0, 1.0, 1.0]),
    ("Capsule-Capsule", 1, 1, [0.5, 2.0, 0], [0.5, 2.0, 0]),
    ("Capsule-Box", 1, 2, [0.5, 2.0, 0], [1.0, 1.0, 1.0]),
    ("Box-Box", 2, 2, [1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
]

print("1. Testing basic shape combinations:")
for name, type_a, type_b, params_a, params_b in test_cases:
    test_count += 1
    body_a = create_body(type_a, [0, 0, 0], np.array([1, 0, 0, 0]), params_a)
    body_b = create_body(type_b, [5, 0, 0], np.array([1, 0, 0, 0]), params_b)
    
    dist, normal, contact_point = engine._compute_sdf_distance(body_a, body_b)
    normal_mag = np.linalg.norm(normal)
    
    if abs(normal_mag - 1.0) < 1e-5 and dist > 0:
        print(f"  ✓ {name}: distance={dist:.3f}, normal OK")
        passed_count += 1
    else:
        print(f"  ✗ {name}: distance={dist:.3f}, normal_mag={normal_mag:.3f}")

# Test overlapping objects
print("\n2. Testing overlapping objects:")
for name, type_a, type_b, params_a, params_b in test_cases[:3]:  # Just first 3
    test_count += 1
    body_a = create_body(type_a, [0, 0, 0], np.array([1, 0, 0, 0]), params_a)
    body_b = create_body(type_b, [0.5, 0, 0], np.array([1, 0, 0, 0]), params_b)
    
    dist, normal, contact_point = engine._compute_sdf_distance(body_a, body_b)
    normal_mag = np.linalg.norm(normal)
    
    if abs(normal_mag - 1.0) < 1e-5 and dist < 0:
        print(f"  ✓ {name}: distance={dist:.3f} (negative), normal OK")
        passed_count += 1
    else:
        print(f"  ✗ {name}: distance={dist:.3f}, normal_mag={normal_mag:.3f}")

# Test rotated objects
print("\n3. Testing rotated objects:")
angle = np.pi / 4
quat = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])

for name, type_a, type_b, params_a, params_b in test_cases[2:4]:  # Box tests
    test_count += 1
    body_a = create_body(type_a, [0, 0, 0], quat, params_a)
    body_b = create_body(type_b, [3, 0, 0], quat, params_b)
    
    dist, normal, contact_point = engine._compute_sdf_distance(body_a, body_b)
    normal_mag = np.linalg.norm(normal)
    
    if abs(normal_mag - 1.0) < 1e-5:
        print(f"  ✓ {name} (rotated): distance={dist:.3f}, normal OK")
        passed_count += 1
    else:
        print(f"  ✗ {name} (rotated): distance={dist:.3f}, normal_mag={normal_mag:.3f}")

# Test edge case: coincident objects
print("\n4. Testing edge case - coincident capsule and box:")
test_count += 1
capsule = create_body(1, [0, 0, 0], np.array([1, 0, 0, 0]), [1.0, 1.0, 0])
box = create_body(2, [0, 0, 0], np.array([1, 0, 0, 0]), [2.0, 1.0, 1.0])

dist, normal, contact_point = engine._compute_sdf_distance(capsule, box)
normal_mag = np.linalg.norm(normal)

if abs(normal_mag - 1.0) < 1e-5:
    print(f"  ✓ Coincident capsule-box: distance={dist:.3f}, normal OK")
    passed_count += 1
else:
    print(f"  ✗ Coincident capsule-box: distance={dist:.3f}, normal_mag={normal_mag:.3f}")

print(f"\n{'='*50}")
print(f"Results: {passed_count}/{test_count} tests passed")
if passed_count == test_count:
    print("✅ All SDF tests passed!")
else:
    print("❌ Some tests failed!")
    exit(1)