import sys
sys.path.insert(0, ".")
sys.path.insert(0, "external/tinygrad")

import numpy as np
from physics.engine import PhysicsEngine
from physics.types import ShapeType
from scripts.scene_builder import SceneBuilder

# Test simple sphere on flat ground with horizontal velocity
builder = SceneBuilder()

# Ground plane
builder.add_body(
    position=[0, -1, 0],
    shape_type=ShapeType.BOX,
    shape_params=[10, 0.05, 10],
    mass=float('inf'),
    friction=0.5
)

# Sphere sliding on ground
builder.add_body(
    position=[0, -0.42, 0],  # More overlap to ensure contact
    velocity=[5, 0, 0],  # Moving horizontally
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

print("Initial state:")
print(f"Sphere position: {engine.x.numpy()[1]}")
print(f"Sphere velocity: {engine.v.numpy()[1]}")

# Run for a short time
for i in range(50):
    engine.step()
    if (i + 1) % 10 == 0:
        state = engine.get_state()
        vel = state['v'][1]
        pos = state['x'][1]
        speed = np.linalg.norm(vel[:2])  # Horizontal speed
        print(f"\nStep {i+1}:")
        print(f"  Position: {pos}")
        print(f"  Velocity: {vel}")
        print(f"  Horizontal speed: {speed:.3f}")

# Without friction, sphere should maintain speed
# With friction, it should slow down