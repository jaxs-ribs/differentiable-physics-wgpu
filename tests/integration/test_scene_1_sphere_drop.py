import numpy as np
import pytest
from physics.engine import PhysicsEngine
from physics.types import ShapeType
from scripts.scene_builder import SceneBuilder


def test_sphere_drop_penetration():
    builder = SceneBuilder()
    
    builder.add_body(
        position=[0, 0, 0],
        shape_type=ShapeType.BOX,
        shape_params=[50, 0.05, 50],
        mass=float('inf'),
        friction=0.5
    )
    
    builder.add_body(
        position=[0, 1, 0],
        shape_type=ShapeType.SPHERE,
        shape_params=[0.5, 0, 0],
        mass=1.0,
        velocity=[0, 0, 0],
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
        dt=0.004,
        restitution=0.0,
        solver_iterations=32,
        contact_compliance=0.00001
    )
    
    num_steps = 250
    max_penetration = 0.0
    
    for _ in range(num_steps):
        engine.step()
        state = engine.get_state()
        
        sphere_pos = state['x'][1]
        sphere_bottom = sphere_pos[1] - 0.5
        plane_top = 0 + 0.05
        
        penetration = plane_top - sphere_bottom
        if penetration > 0:
            max_penetration = max(max_penetration, penetration)
    
    tolerance_mm = 0.5
    tolerance_m = tolerance_mm / 1000.0
    
    assert max_penetration < tolerance_m, \
        f"Maximum penetration {max_penetration*1000:.2f}mm exceeds tolerance {tolerance_mm}mm"
    
    final_state = engine.get_state()
    final_sphere_pos = final_state['x'][1]
    expected_rest_height = 0 + 0.05 + 0.5
    
    assert abs(final_sphere_pos[1] - expected_rest_height) < 0.02, \
        f"Sphere not at rest: position={final_sphere_pos[1]:.4f}, expected={expected_rest_height:.4f}"