#!/usr/bin/env python3
"""Create a test state dump for testing debug_viz"""
import numpy as np
from reference import PhysicsEngine, Body, ShapeType

# Create engine with 10 bodies
engine = PhysicsEngine(dt=0.01, gravity=np.array([0, -9.81, 0]))

# Add ground
ground = Body(
    position=np.array([0., 0., 0.]),
    velocity=np.array([0., 0., 0.]),
    orientation=np.array([1., 0., 0., 0.]),
    angular_vel=np.zeros(3),
    mass=1e8,  # Static
    inertia=np.eye(3) * 1e8,
    shape_type=ShapeType.SPHERE,
    shape_params=np.array([10., 0., 0.])  # Large ground
)
engine.add_body(ground)

# Add some spheres in a grid
for i in range(3):
    for j in range(3):
        body = Body(
            position=np.array([i*3-3, 5+j*2.5, 0.]),
            velocity=np.array([0., 0., 0.]),
            orientation=np.array([1., 0., 0., 0.]),
            angular_vel=np.zeros(3),
            mass=1.0,
            inertia=np.eye(3) * 0.4,
            shape_type=ShapeType.SPHERE,
            shape_params=np.array([1., 0., 0.])
        )
        engine.add_body(body)

# Get state and save
state = engine.get_state()
np.save('oracle_dump.npy', state)
print(f"Created oracle_dump.npy with {len(engine.bodies)} bodies")
print("State shape:", state.shape)