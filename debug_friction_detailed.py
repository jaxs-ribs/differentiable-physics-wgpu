import sys
sys.path.insert(0, ".")
sys.path.insert(0, "external/tinygrad")

import numpy as np
from physics.engine import PhysicsEngine
from physics.types import ShapeType
from scripts.scene_builder import SceneBuilder

# Test with different friction values
friction_values = [0.0, 0.3, 0.6, 0.9]

for mu in friction_values:
    print(f"\n=== Testing with friction coefficient μ = {mu} ===")
    
    builder = SceneBuilder()
    
    # Ground plane
    builder.add_body(
        position=[0, -1, 0],
        shape_type=ShapeType.BOX,
        shape_params=[10, 0.05, 10],
        mass=float('inf'),
        friction=mu
    )
    
    # Sphere sliding on ground
    builder.add_body(
        position=[0, -0.42, 0],  # More overlap to ensure contact
        velocity=[5, 0, 0],  # Moving horizontally
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
    
    initial_speed = 5.0
    
    # Run for 1 second (62.5 steps)
    for i in range(63):
        engine.step()
    
    final_state = engine.get_state()
    final_vel = final_state['v'][1]
    final_speed = np.linalg.norm(final_vel[:2])
    
    # Expected deceleration: a = μ * g
    expected_decel = mu * 9.81
    expected_final_speed = max(0, initial_speed - expected_decel * 1.0)
    
    print(f"  Final velocity: {final_vel}")
    print(f"  Final speed: {final_speed:.3f} m/s")
    print(f"  Expected speed: {expected_final_speed:.3f} m/s")
    print(f"  Speed reduction: {initial_speed - final_speed:.3f} m/s")