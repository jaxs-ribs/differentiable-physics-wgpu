#!/usr/bin/env python3
"""Debug broadphase issue"""

from reference import Body, PhysicsEngine, ShapeType
import numpy as np

def create_sphere_body(position, radius=1.0, mass=1.0):
    """Create a sphere body at the given position."""
    return Body(
        position=np.array(position),
        velocity=np.zeros(3),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_vel=np.zeros(3),
        mass=mass,
        inertia=np.eye(3) * 0.4 * mass * radius * radius,
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([radius, 0.0, 0.0])
    )

print("Testing broadphase with 2 spheres...")
engine = PhysicsEngine()

# Group 1: Overlapping spheres at origin
engine.add_body(create_sphere_body([0, 0, 0], radius=1.0))
engine.add_body(create_sphere_body([1.5, 0, 0], radius=1.0))

print(f"Number of bodies: {len(engine.bodies)}")

# Get pairs from both methods
print("Getting brute force pairs...")
bruteforce_pairs = engine.get_all_pairs_bruteforce()
print(f"Brute force pairs: {bruteforce_pairs}")

print("Getting SAP pairs...")
# Add debug to broadphase
broadphase = engine.broadphase
aabbs = []
for i, body in enumerate(engine.bodies):
    aabb = engine._compute_aabb(body)
    aabb.body_index = i
    aabbs.append(aabb)
    print(f"AABB {i}: min={aabb.min_point}, max={aabb.max_point}")

print("Creating endpoints...")
x_endpoints = []
for aabb in aabbs:
    idx = aabb.body_index
    x_endpoints.append((aabb.min_point[0], idx, True))
    x_endpoints.append((aabb.max_point[0], idx, False))
x_endpoints.sort(key=lambda x: x[0])
print(f"X endpoints: {x_endpoints}")

print("Finding X overlaps...")
# Test just X axis
active = set()
overlaps = set()
i = 0
max_iterations = 100
while i < len(x_endpoints) and max_iterations > 0:
    max_iterations -= 1
    print(f"  i={i}, endpoint={x_endpoints[i]}")
    current_pos = x_endpoints[i][0]
    starts_at_pos = []
    ends_at_pos = []
    
    while i < len(x_endpoints) and x_endpoints[i][0] == current_pos:
        _, body_idx, is_min = x_endpoints[i]
        if is_min:
            starts_at_pos.append(body_idx)
        else:
            ends_at_pos.append(body_idx)
        i += 1
    
    print(f"    At pos {current_pos}: starts={starts_at_pos}, ends={ends_at_pos}, active={active}")
    
    # This is where it might hang
    for new_idx in starts_at_pos:
        for other_idx in active:
            pair = (min(new_idx, other_idx), max(new_idx, other_idx))
            overlaps.add(pair)
    
    # Update active set
    for idx in ends_at_pos:
        active.discard(idx)
    for idx in starts_at_pos:
        active.add(idx)
        
    print(f"    After update: active={active}, overlaps={overlaps}")

if max_iterations == 0:
    print("INFINITE LOOP DETECTED!")
    
print(f"X overlaps: {overlaps}")

print("Done!")