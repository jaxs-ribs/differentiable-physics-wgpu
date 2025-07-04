import sys
sys.path.insert(0, ".")
sys.path.insert(0, "external/tinygrad")

import numpy as np
from physics.engine import PhysicsEngine
from physics.types import ShapeType
from scripts.scene_builder import SceneBuilder

# Test basic contact stability - sphere should rest on ground
builder = SceneBuilder()

# Ground plane
builder.add_body(
    position=[0, 0, 0],
    shape_type=ShapeType.BOX,
    shape_params=[10, 0.05, 10],
    mass=float('inf'),
    friction=0.5
)

# Sphere starting with slight penetration
builder.add_body(
    position=[0, 0.54, 0],  # 0.01 penetration
    velocity=[0, 0, 0],  # No initial velocity
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
    gravity=np.array([0, -9.81, 0]),
    dt=0.016,
    restitution=0.0,
    solver_iterations=16,
    contact_compliance=0.0001
)

print("Initial sphere position:", engine.x.numpy()[1])

# Run for 1 second
for i in range(62):
    engine.step()
    if i % 10 == 9:
        state = engine.get_state()
        print(f"Step {i+1}: position = {state['x'][1]}, velocity = {state['v'][1]}")

# Check final position - should be stable around y=0.55
final_pos = engine.get_state()['x'][1]
print(f"\nFinal position: {final_pos}")
print(f"Expected position (approx): [0, 0.55, 0]")
print(f"Deviation from expected: {abs(final_pos[1] - 0.55):.4f}")