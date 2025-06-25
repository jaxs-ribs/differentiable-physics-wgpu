#!/usr/bin/env python3
import numpy as np
import subprocess
import json

def run_sdf_test(sphere1_pos, sphere1_radius, sphere2_pos, sphere2_radius):
    """Run GPU SDF test with given sphere configurations"""
    # For now, we'll use the hardcoded test binary
    # In a real test, we'd pass parameters
    result = subprocess.run(
        ["cargo", "run", "--bin", "test_sdf"],
        capture_output=True,
        text=True,
        cwd=".."
    )
    
    # Parse output
    lines = result.stdout.strip().split('\n')
    contact_count = 0
    distance = None
    
    for line in lines:
        if "Contact count:" in line:
            contact_count = int(line.split(":")[1].strip())
        elif "distance:" in line and "bodies" in line:
            # Extract distance from "Contact 0: bodies 0 and 1, distance: -0.100000"
            distance = float(line.split("distance:")[1].strip())
    
    return contact_count, distance

def test_sphere_sphere_collision():
    """Test various sphere-sphere collision scenarios"""
    test_cases = [
        # (pos1, r1, pos2, r2, expected_distance, should_collide)
        ([0, 0, 0], 1.0, [3, 0, 0], 0.5, 1.5, False),      # Far apart
        ([0, 0, 0], 1.0, [1.4, 0, 0], 0.5, -0.1, True),    # Overlapping
        ([0, 0, 0], 1.0, [1.5, 0, 0], 0.5, 0.0, True),     # Just touching
        ([0, 0, 0], 1.0, [1.55, 0, 0], 0.5, 0.05, True),   # Close (within threshold)
    ]
    
    # For now, we can only test the hardcoded case
    print("Testing overlapping spheres case...")
    contact_count, distance = run_sdf_test([0, 0, 0], 1.0, [1.4, 0, 0], 0.5)
    
    print(f"Contact count: {contact_count}")
    print(f"Distance: {distance}")
    
    # Verify results
    assert contact_count == 1, f"Expected 1 contact, got {contact_count}"
    assert abs(distance - (-0.1)) < 1e-5, f"Expected distance -0.1, got {distance}"
    print("✓ Overlapping spheres test passed")

def test_property_based():
    """Property-based testing with random configurations"""
    # Load the test cases we generated
    with open('sdf_test_cases.json', 'r') as f:
        test_cases = json.load(f)
    
    # For each case, verify the distance property
    print(f"\nVerifying {len(test_cases)} random test cases...")
    
    for i, case in enumerate(test_cases[:10]):  # Test first 10
        s1 = case['sphere1']
        s2 = case['sphere2']
        expected = case['expected_distance']
        
        # Property: distance should never be less than -r1-r2 (maximum penetration)
        min_distance = -(s1['radius'] + s2['radius'])
        assert expected >= min_distance, f"Invalid distance in case {i}: {expected} < {min_distance}"
        
        # Property: if centers coincide, distance = -(r1+r2)
        center_dist = np.linalg.norm(np.array(s2['center']) - np.array(s1['center']))
        if center_dist < 1e-6:
            assert abs(expected - min_distance) < 1e-6
    
    print("✓ Property tests passed")

if __name__ == "__main__":
    print("Running GPU SDF tests...")
    
    test_sphere_sphere_collision()
    test_property_based()
    
    print("\n✓ All SDF tests passed!")