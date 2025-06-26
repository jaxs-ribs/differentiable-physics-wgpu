#!/usr/bin/env python3
"""Minimal test to isolate issues"""

import numpy as np

print("Testing endpoint processing...")

# Simple endpoint list
endpoints = [(0.0, 0, True), (1.0, 0, False), (0.5, 1, True), (1.5, 1, False)]

i = 0
iterations = 0
max_iterations = 10

while i < len(endpoints) and iterations < max_iterations:
    iterations += 1
    print(f"Iteration {iterations}, i={i}")
    
    current_pos = endpoints[i][0]
    starts = []
    ends = []
    
    start_i = i
    while i < len(endpoints) and endpoints[i][0] == current_pos:
        _, idx, is_min = endpoints[i]
        if is_min:
            starts.append(idx)
        else:
            ends.append(idx)
        i += 1
    
    print(f"  Processed indices {start_i} to {i-1}")
    print(f"  Position {current_pos}: starts={starts}, ends={ends}")

print(f"Finished. Processed all {i} endpoints in {iterations} iterations")

# Now test the actual reference import
print("\nTesting reference import...")
try:
    from reference import PhysicsEngine, Body, ShapeType
    print("✓ Import successful")
    
    # Try creating engine
    engine = PhysicsEngine()
    print("✓ Engine created")
    
    # Try minimal broadphase test
    from reference import AABB, BroadphaseDetector
    detector = BroadphaseDetector()
    aabbs = [
        AABB(np.array([0., 0., 0.]), np.array([1., 1., 1.]), 0)
    ]
    print("Testing detect_pairs with single AABB...")
    pairs = detector.detect_pairs(aabbs)
    print(f"✓ Pairs detected: {pairs}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()