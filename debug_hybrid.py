import sys
sys.path.insert(0, ".")
sys.path.insert(0, "external/tinygrad")

import numpy as np
from physics.engine import PhysicsEngine
from physics.types import ShapeType
from scripts.scene_builder import SceneBuilder

# Create simple test scene
builder = SceneBuilder()

# Ground plane
builder.add_body(
    position=[0, -1, 0],
    shape_type=ShapeType.BOX,
    shape_params=[10, 0.05, 10],
    mass=float('inf')
)

# Sphere close to ground
builder.add_body(
    position=[0, -0.4, 0],  # Close to ground
    velocity=[0, -1, 0],  # Moving down
    shape_type=ShapeType.SPHERE,
    shape_params=[0.5, 0, 0],
    mass=1.0
)

scene_data = builder.build()

# Create engine
engine = PhysicsEngine(
    x=scene_data['x'],
    q=scene_data['q'],
    v=scene_data['v'],
    omega=scene_data['omega'],
    inv_mass=scene_data['inv_mass'],
    inv_inertia=scene_data['inv_inertia'],
    shape_type=scene_data['shape_type'],
    shape_params=scene_data['shape_params'],
    gravity=np.array([0, -9.81, 0]),
    dt=0.016,
    restitution=0.5,
    solver_iterations=16,
    contact_compliance=0.001
)

print("Initial state:")
print(f"Sphere position: {engine.x.numpy()[1]}")
print(f"Sphere velocity: {engine.v.numpy()[1]}")

# Calculate expected collision geometry
plane_top = -0.95  # -1 + 0.05
sphere_radius = 0.5
print(f"\nPlane top: {plane_top}")
print(f"Sphere radius: {sphere_radius}")
print(f"Expected rest position: {plane_top + sphere_radius}")

# Run a few steps
for i in range(10):
    engine.step()
    pos = engine.x.numpy()[1]
    vel = engine.v.numpy()[1]
    sphere_bottom = pos[1] - sphere_radius
    penetration = max(0, plane_top - sphere_bottom)
    
    print(f"\nStep {i+1}:")
    print(f"  Position: {pos}")
    print(f"  Velocity: {vel}")
    print(f"  Sphere bottom: {sphere_bottom:.4f}")
    print(f"  Penetration: {penetration*1000:.2f} mm")
    
    if abs(vel[1]) < 0.01 and i > 5:
        print("  Sphere has settled")
        break