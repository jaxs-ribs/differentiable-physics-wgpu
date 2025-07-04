#!/usr/bin/env python3
"""Test script demonstrating advanced collision detection capabilities."""

import numpy as np
from physics.types import ShapeType
from physics.engine import TensorPhysicsEngine
from scripts.scene_builder import SceneBuilder

def create_test_scene():
    """Create a scene with various collision pairs."""
    builder = SceneBuilder()
    
    # Ground plane (thin box)
    builder.add_body(
        position=[0.0, -1.0, 0.0],
        mass=1e10,  # Static
        shape_type=ShapeType.BOX,
        shape_params=[10.0, 0.1, 10.0],
        friction=0.8
    )
    
    # Box on ground
    builder.add_body(
        position=[-2.0, 1.0, 0.0],
        mass=1.0,
        shape_type=ShapeType.BOX,
        shape_params=[0.5, 0.5, 0.5],
        friction=0.5
    )
    
    # Sphere falling onto box
    builder.add_body(
        position=[-2.0, 3.0, 0.0],
        mass=0.5,
        shape_type=ShapeType.SPHERE,
        shape_params=[0.3, 0.0, 0.0],
        friction=0.3
    )
    
    # Capsule standing upright
    builder.add_body(
        position=[0.0, 2.0, 0.0],
        mass=0.8,
        shape_type=ShapeType.CAPSULE,
        shape_params=[0.3, 1.0, 0.0],  # radius=0.3, half_height=1.0
        friction=0.4
    )
    
    # Another capsule, tilted
    angle = np.pi / 6  # 30 degrees
    builder.add_body(
        position=[2.0, 2.0, 0.0],
        mass=0.8,
        shape_type=ShapeType.CAPSULE,
        shape_params=[0.3, 1.0, 0.0],
        orientation=[np.cos(angle/2), 0.0, 0.0, np.sin(angle/2)],  # Rotated around Z
        friction=0.4
    )
    
    # Box for capsule to collide with
    builder.add_body(
        position=[2.0, 0.5, 0.0],
        mass=2.0,
        shape_type=ShapeType.BOX,
        shape_params=[0.8, 0.4, 0.8],
        friction=0.6
    )
    
    return builder.build()

def main():
    # Create scene
    body_data = create_test_scene()
    
    # Initialize engine
    engine = TensorPhysicsEngine(
        x=body_data['x'],
        q=body_data['q'],
        v=body_data['v'],
        omega=body_data['omega'],
        inv_mass=body_data['inv_mass'],
        inv_inertia=body_data['inv_inertia'],
        shape_type=body_data['shape_type'],
        shape_params=body_data['shape_params'],
        friction=body_data['friction'],
        gravity=np.array([0.0, -9.81, 0.0])
    )
    
    # Simulate
    dt = 0.016  # 60 Hz
    steps = 120  # 2 seconds
    
    print("Simulating advanced collision scene...")
    print(f"Bodies: {len(body_data['x'])}")
    print("Shapes:")
    for i, shape_type in enumerate(body_data['shape_type']):
        shape_name = ["Sphere", "?", "Box", "Capsule"][shape_type]
        print(f"  Body {i}: {shape_name} at position {body_data['x'][i]}")
    print()
    
    for step in range(steps):
        engine.step()
        
        if step % 30 == 0:  # Print every 0.5 seconds
            x_np = engine.x.numpy()
            v_np = engine.v.numpy()
            print(f"t={step*dt:.2f}s:")
            for i in range(len(x_np)):
                pos = x_np[i]
                vel = np.linalg.norm(v_np[i])
                print(f"  Body {i}: pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}], |v|={vel:.2f}")
    
    print("\nSimulation complete!")
    
    # Check final positions
    print("\nFinal positions:")
    x_final = engine.x.numpy()
    for i in range(len(x_final)):
        pos = x_final[i]
        print(f"  Body {i}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

if __name__ == "__main__":
    main()