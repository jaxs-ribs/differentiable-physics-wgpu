import sys
sys.path.insert(0, ".")
sys.path.insert(0, "external/tinygrad")

import numpy as np
from physics.engine import PhysicsEngine
from physics.types import ShapeType
from scripts.scene_builder import SceneBuilder

print("=== Comprehensive Friction Test ===\n")

# Test 1: Horizontal sliding with friction
print("Test 1: Horizontal sliding")
for mu in [0.0, 0.5, 1.0]:
    builder = SceneBuilder()
    
    # Ground
    builder.add_body(
        position=[0, 0, 0],
        shape_type=ShapeType.BOX,
        shape_params=[10, 0.05, 10],
        mass=float('inf'),
        friction=mu
    )
    
    # Sphere
    builder.add_body(
        position=[0, 0.54, 0],  # Small penetration for contact
        velocity=[5, 0, 0],
        shape_type=ShapeType.SPHERE,
        shape_params=[0.5, 0, 0],
        mass=1.0,
        friction=mu
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
    
    initial_vel = 5.0
    
    # Run for 0.5 seconds
    for _ in range(31):
        engine.step()
    
    final_state = engine.get_state()
    final_vel = final_state['v'][1][0]  # X velocity
    
    # Expected: v = v0 - μ*g*t
    expected_vel = max(0, initial_vel - mu * 9.81 * 0.5)
    
    print(f"  μ={mu}: final velocity = {final_vel:.3f}, expected = {expected_vel:.3f}")

# Test 2: Simple slope test
print("\nTest 2: 30° slope sliding")
theta = np.radians(30)

for mu in [0.0, 0.3]:
    builder = SceneBuilder()
    
    # Tilted plane
    half_angle = theta / 2
    plane_orientation = [np.cos(half_angle), 0, 0, np.sin(half_angle)]
    
    builder.add_body(
        position=[0, 0, 0],
        shape_type=ShapeType.BOX,
        shape_params=[10, 0.05, 10],
        mass=float('inf'),
        orientation=plane_orientation,
        friction=mu
    )
    
    # Sphere on slope
    builder.add_body(
        position=[-1*np.sin(theta), 1*np.cos(theta) + 0.02, 0],  # On slope with small penetration
        shape_type=ShapeType.SPHERE,
        shape_params=[0.5, 0, 0],
        mass=1.0,
        velocity=[0, 0, 0],
        friction=mu
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
        dt=0.008,
        restitution=0.0,
        solver_iterations=16,
        contact_compliance=0.0001
    )
    
    # Let settle
    for _ in range(50):
        engine.step()
    
    initial_pos = engine.get_state()['x'][1].copy()
    
    # Measure motion
    for _ in range(125):  # 1 second
        engine.step()
    
    final_pos = engine.get_state()['x'][1]
    displacement = np.linalg.norm(final_pos - initial_pos)
    
    # Expected: s = 0.5 * a * t^2, where a = g*(sin(θ) - μ*cos(θ))
    expected_accel = 9.81 * (np.sin(theta) - mu * np.cos(theta))
    expected_disp = 0.5 * expected_accel * 1.0**2
    
    print(f"  μ={mu}: displacement = {displacement:.3f}m, expected = {expected_disp:.3f}m")
    print(f"         acceleration = {2*displacement:.3f} m/s², expected = {expected_accel:.3f} m/s²")