import numpy as np
import pytest
from physics.engine import PhysicsEngine
from physics.types import ShapeType
from scripts.scene_builder import SceneBuilder


@pytest.mark.skip(reason="Restitution not yet implemented in velocity solver")
def test_elastic_bounce_restitution():
    builder = SceneBuilder()
    
    builder.add_body(
        position=[0, 0, 0],
        shape_type=ShapeType.BOX,
        shape_params=[50, 0.05, 50],
        mass=float('inf'),
        friction=0.0
    )
    
    initial_height = 2.0
    builder.add_body(
        position=[0, initial_height, 0],
        shape_type=ShapeType.SPHERE,
        shape_params=[0.5, 0, 0],
        mass=1.0,
        velocity=[0, 0, 0],
        friction=0.0
    )
    
    scene_data = builder.build()
    
    restitution = 0.8
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
        dt=0.002,
        restitution=restitution,
        solver_iterations=32,
        contact_compliance=0.00001
    )
    
    max_heights = []
    direction = "falling"
    local_max = initial_height
    ground_level = 0.55
    
    for step in range(2000):
        engine.step()
        state = engine.get_state()
        
        sphere_pos = state['x'][1]
        sphere_vel = state['v'][1]
        current_y = sphere_pos[1]
        current_vy = sphere_vel[1]
        
        if direction == "falling" and current_y < ground_level + 0.1 and current_vy > 0.5:
            direction = "rising"
        elif direction == "rising":
            if current_y > local_max:
                local_max = current_y
            elif current_y < local_max - 0.02 and current_vy < 0:
                max_heights.append(local_max)
                direction = "falling"
                local_max = 0
                
        if len(max_heights) >= 2:
            break
    
    assert len(max_heights) >= 2, f"Did not detect enough bounces, only found {len(max_heights)}"
    
    drop_height = initial_height - (0 + 0.05 + 0.5)
    expected_second_bounce_height = drop_height * (restitution ** 2) + (0 + 0.05 + 0.5)
    
    second_bounce_height = max_heights[1]
    second_bounce_drop_height = second_bounce_height - (0 + 0.05 + 0.5)
    
    relative_error = abs(second_bounce_drop_height - drop_height * (restitution ** 2)) / (drop_height * (restitution ** 2))
    
    assert relative_error < 0.02, \
        f"Second bounce height error {relative_error*100:.1f}% exceeds 2% tolerance. " \
        f"Expected drop height: {drop_height * (restitution ** 2):.3f}m, " \
        f"Measured: {second_bounce_drop_height:.3f}m"