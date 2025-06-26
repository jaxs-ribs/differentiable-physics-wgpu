#!/usr/bin/env python3
"""Create a test state dump for testing debug_viz"""
import numpy as np
from reference import PhysicsEngine, Body, ShapeType

# --- Parameters ---
NUM_STEPS = 200
DT = 0.016

# --- Scene Creation ---
engine = PhysicsEngine(dt=DT, gravity=np.array([0, -9.81, 0]))

# Add ground
ground = Body(
    position=np.array([0., -5., 0.]),
    velocity=np.array([0., 0., 0.]),
    orientation=np.array([1., 0., 0., 0.]),
    angular_vel=np.zeros(3),
    mass=1e8,  # Static
    inertia=np.eye(3) * 1e8,
    shape_type=ShapeType.SPHERE,
    shape_params=np.array([5., 0., 0.])
)
engine.add_body(ground)

# Add some boxes to fall
for i in range(3):
    for j in range(3):
        body = Body(
            position=np.array([i*3.0 - 3.0, 5.0 + j*3.0, 0.]),
            velocity=np.array([0., 0., 0.]),
            orientation=np.array([1., 0., 0., 0.]),
            angular_vel=np.array([(np.random.rand() - 0.5) * 5.0, (np.random.rand() - 0.5) * 5.0, (np.random.rand() - 0.5) * 5.0]),
            mass=1.0,
            inertia=np.eye(3) * (2.0/5.0) * 1.0 * (1.0**2), # Solid sphere inertia
            shape_type=ShapeType.BOX,
            shape_params=np.array([1.0, 1.0, 1.0]) # Half-extents
        )
        engine.add_body(body)

# --- Simulation and Data Collection ---
all_states = []
for _ in range(NUM_STEPS):
    engine.step()
    all_states.append(engine.get_state())

# --- Save to File ---
state_array = np.array(all_states)
np.save('oracle_dump.npy', state_array)

print(f"Created oracle_dump.npy with {len(engine.bodies)} bodies over {NUM_STEPS} frames.")
print("State array shape:", state_array.shape)