import sys
sys.path.insert(0, ".")
sys.path.insert(0, "external/tinygrad")

import numpy as np
from physics.engine import PhysicsEngine
from physics.types import ShapeType
from scripts.scene_builder import SceneBuilder

# Very simple test - sphere on flat ground, no gravity
builder = SceneBuilder()

# Ground plane
builder.add_body(
    position=[0, 0, 0],
    shape_type=ShapeType.BOX,
    shape_params=[10, 0.05, 10],
    mass=float('inf'),
    friction=0.5
)

# Sphere resting on ground
builder.add_body(
    position=[0, 0.56, 0],  # Radius 0.5 + small overlap
    velocity=[2, 0, 0],  # Slow horizontal velocity
    shape_type=ShapeType.SPHERE,
    shape_params=[0.5, 0, 0],
    mass=1.0,
    friction=0.5
)

scene_data = builder.build()

engine = PhysicsEngine(
    x=scene_data['x'],
    q=scene_data['q'],
    v=scene_data['v'],
    omega=scene_data['omega'],
    inv_mass=scene_data['inv_mass'],
    inv_inertia=scene_data['inv_inertia'],
    shape_type=scene_data['shape_type'],
    shape_params=scene_data['shape_params'],
    friction=scene_data['friction'],
    gravity=np.array([0, 0, 0]),  # No gravity
    dt=0.016,
    restitution=0.0,
    solver_iterations=16,
    contact_compliance=0.0001
)

print("Initial state:")
print(f"Sphere position: {engine.x.numpy()[1]}")
print(f"Sphere velocity: {engine.v.numpy()[1]}")

# Run for several steps
velocities = []
for i in range(20):
    engine.step()
    state = engine.get_state()
    vel = state['v'][1]
    velocities.append(vel[0])  # X velocity
    
    if i % 5 == 4:
        print(f"\nStep {i+1}:")
        print(f"  Position: {state['x'][1]}")
        print(f"  Velocity: {vel}")

# Check if velocity is decreasing
print("\nX velocities over time:")
for i in range(0, len(velocities), 5):
    print(f"  Step {i}: {velocities[i]:.4f}")

# Calculate deceleration
if len(velocities) > 1:
    decel = (velocities[0] - velocities[-1]) / (len(velocities) * 0.016)
    print(f"\nMeasured deceleration: {decel:.3f} m/s²")
    print(f"Expected deceleration (μ*g=0.5*9.81): 4.905 m/s²")
    print(f"(Note: No gravity in this test, so friction should be minimal)")