#!/usr/bin/env python3
"""Simple bounce test to verify physics is working correctly"""
import numpy as np
from reference import PhysicsEngine, Body, ShapeType

# Create engine
engine = PhysicsEngine(dt=0.01, gravity=np.array([0, -9.81, 0]))

# Ground (static)
ground = Body(
    position=np.array([0., 0., 0.]),
    velocity=np.array([0., 0., 0.]),
    orientation=np.array([1., 0., 0., 0.]),
    angular_vel=np.zeros(3),
    mass=1e8,  # Static
    inertia=np.eye(3) * 1e8,
    shape_type=ShapeType.SPHERE,
    shape_params=np.array([5., 0., 0.])
)

# Ball
ball = Body(
    position=np.array([0., 10., 0.]),
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

print("Simulating ball drop and bounce...")
print(f"Ball starts at y={ball.position[1]}, should bounce at y={ground.position[1] + ground.shape_params[0] + ball.shape_params[0]}")

# Track bounces
bounce_heights = []
last_velocity = 0
for i in range(1000):
    engine.step()
    
    # Detect bounce (velocity changes sign from negative to positive)
    if last_velocity < 0 and ball.velocity[1] > 0:
        bounce_heights.append(ball.position[1])
        print(f"Bounce {len(bounce_heights)} at height {ball.position[1]:.3f}")
    
    last_velocity = ball.velocity[1]
    
    if len(bounce_heights) >= 5:
        break

# Check restitution
if len(bounce_heights) >= 2:
    restitution = np.sqrt(bounce_heights[1] / bounce_heights[0])
    print(f"\nEffective restitution coefficient: {restitution:.3f}")
    print("(Should be ~0.8 based on CollisionResolver)")