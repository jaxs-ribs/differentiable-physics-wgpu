#!/usr/bin/env python3
"""Debug the broadphase hanging issue"""

import numpy as np
from reference import AABB, BroadphaseDetector

print("Testing broadphase with simple case...")

detector = BroadphaseDetector()

# Create two overlapping AABBs
aabbs = [
    AABB(min_point=np.array([-1., -1., -1.]), max_point=np.array([1., 1., 1.]), body_index=0),
    AABB(min_point=np.array([0., -1., -1.]), max_point=np.array([2., 1., 1.]), body_index=1)
]

print("AABBs created")
print(f"AABB 0: min={aabbs[0].min_point}, max={aabbs[0].max_point}")
print(f"AABB 1: min={aabbs[1].min_point}, max={aabbs[1].max_point}")

# Create endpoints manually
x_endpoints = []
for aabb in aabbs:
    idx = aabb.body_index
    x_endpoints.append((aabb.min_point[0], idx, True))
    x_endpoints.append((aabb.max_point[0], idx, False))

x_endpoints.sort(key=lambda x: x[0])
print(f"\nX endpoints: {x_endpoints}")

# Test the problematic function
print("\nTesting _find_axis_overlaps...")
overlaps = detector._find_axis_overlaps(x_endpoints)
print(f"Overlaps found: {overlaps}")

print("\nNow testing full detect_pairs...")
try:
    pairs = detector.detect_pairs(aabbs)
    print(f"Pairs found: {pairs}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()