import sys
sys.path.insert(0, ".")
sys.path.insert(0, "external/tinygrad")

import numpy as np
from physics.engine import PhysicsEngine
from physics.types import ShapeType
from scripts.scene_builder import SceneBuilder

# Test different parameter combinations
test_params = [
    {"compliance": 0.0001, "iterations": 16, "dt": 0.016},
    {"compliance": 0.00001, "iterations": 16, "dt": 0.016},
    {"compliance": 0.0001, "iterations": 32, "dt": 0.016},
    {"compliance": 0.0001, "iterations": 16, "dt": 0.008},
    {"compliance": 0.00001, "iterations": 32, "dt": 0.008},
]

for i, params in enumerate(test_params):
    print(f"\nTest {i+1}: compliance={params['compliance']}, iterations={params['iterations']}, dt={params['dt']}")
    
    # Create scene
    builder = SceneBuilder()
    builder.add_body(
        position=[0, -1, 0],
        shape_type=ShapeType.BOX,
        shape_params=[10, 0.05, 10],
        mass=float('inf')
    )
    builder.add_body(
        position=[0, 1, 0],
        velocity=[0, 0, 0],
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
        dt=params['dt'],
        restitution=0.1,
        solver_iterations=params['iterations'],
        contact_compliance=params['compliance']
    )
    
    # Run for 1 second
    num_steps = int(1.0 / params['dt'])
    for _ in range(num_steps):
        engine.step()
    
    # Calculate penetration
    final_state = engine.get_state()
    sphere_final_y = final_state['x'][1, 1]
    sphere_bottom = sphere_final_y - 0.5
    plane_top = -0.95
    penetration = max(0, plane_top - sphere_bottom)
    
    print(f"  Final penetration: {penetration*1000:.2f} mm")
    print(f"  Final velocity: {final_state['v'][1, 1]:.4f} m/s")
    
    if penetration <= 0.0005:
        print("  ✓ PASSES 0.5mm requirement!")
    else:
        print(f"  ✗ Fails (need {(penetration/0.0005):.1f}x improvement)")