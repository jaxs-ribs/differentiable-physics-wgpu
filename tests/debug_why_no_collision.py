#!/usr/bin/env python3
"""Debug why collisions aren't happening"""
import numpy as np
from reference import PhysicsEngine, Body, ShapeType

# Hook into the entire collision system
class DebugEngine(PhysicsEngine):
    def _detect_and_resolve_collisions(self):
        print("\n=== Collision Detection Phase ===")
        # Compute AABBs
        aabbs = []
        for i, body in enumerate(self.bodies):
            aabb = self._compute_aabb(body)
            aabb.body_index = i
            aabbs.append(aabb)
            print(f"Body {i}: pos={body.position}, AABB min={aabb.min_point}, max={aabb.max_point}")
        
        # Broadphase
        potential_pairs = self.broadphase.detect_pairs(aabbs)
        print(f"Broadphase found {len(potential_pairs)} pairs: {potential_pairs}")
        
        # Manually check if AABBs overlap
        if len(aabbs) >= 2:
            aabb0, aabb1 = aabbs[0], aabbs[1]
            overlap_x = aabb0.max_point[0] >= aabb1.min_point[0] and aabb0.min_point[0] <= aabb1.max_point[0]
            overlap_y = aabb0.max_point[1] >= aabb1.min_point[1] and aabb0.min_point[1] <= aabb1.max_point[1]
            overlap_z = aabb0.max_point[2] >= aabb1.min_point[2] and aabb0.min_point[2] <= aabb1.max_point[2]
            print(f"Manual overlap check: X={overlap_x}, Y={overlap_y}, Z={overlap_z}")
        
        # Narrowphase
        for i, j in potential_pairs:
            print(f"\nProcessing pair ({i}, {j})...")
            body_a, body_b = self.bodies[i], self.bodies[j]
            distance, normal, contact = self._compute_sdf_distance(body_a, body_b)
            print(f"  Distance: {distance:.3f}")
            print(f"  Normal: {normal}")
            
            if distance < 0:
                print(f"  ✓ Collision detected! Resolving...")
                old_pos_a = body_a.position.copy()
                old_pos_b = body_b.position.copy()
                self.collision_resolver.resolve(body_a, body_b, distance, normal, contact, self.dt)
                print(f"  Position change A: {body_a.position - old_pos_a}")
                print(f"  Position change B: {body_b.position - old_pos_b}")
            else:
                print(f"  ✗ No collision (distance > 0)")

# Simple test
engine = DebugEngine(dt=0.01, gravity=np.array([0, -9.81, 0]))

ground = Body(
    position=np.array([0., 0., 0.]),
    velocity=np.array([0., 0., 0.]),
    orientation=np.array([1., 0., 0., 0.]),
    angular_vel=np.zeros(3),
    mass=1e8,
    inertia=np.eye(3) * 1e8,
    shape_type=ShapeType.SPHERE,
    shape_params=np.array([2., 0., 0.])
)

ball = Body(
    position=np.array([0., 3.5, 0.]),  # Should collide at y=3
    velocity=np.array([0., 0., 0.]),
    orientation=np.array([1., 0., 0., 0.]),
    angular_vel=np.zeros(3),
    mass=1.0,
    inertia=np.eye(3) * 0.4,
    shape_type=ShapeType.SPHERE,
    shape_params=np.array([1., 0., 0.])
)

engine.add_body(ground)
engine.add_body(ball)

print("Setup: Ground sphere at (0,0,0) radius 2, ball at (0,3.5,0) radius 1")
print("Ball should collide when bottom reaches y=2\n")

# Run a few steps
for i in range(50):
    print(f"\n--- Step {i} ---")
    print(f"Ball position: {ball.position}, velocity: {ball.velocity}")
    engine.step()
    
    # Check if ball went through
    if ball.position[1] < 2:
        print("\n❌ Ball went through the ground!")
        break