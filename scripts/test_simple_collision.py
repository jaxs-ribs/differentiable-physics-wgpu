#!/usr/bin/env python3
"""Simple test for collision detection."""

import numpy as np
from physics.types import ShapeType
from physics.engine import TensorPhysicsEngine
from scripts.scene_builder import SceneBuilder

def main():
    # Create simple scene with just two boxes
    builder = SceneBuilder()
    
    # Ground plane
    builder.add_body(
        position=[0.0, -1.0, 0.0],
        mass=1e10,
        shape_type=ShapeType.BOX,
        shape_params=[10.0, 0.1, 10.0],
        friction=0.5
    )
    
    # Falling box
    builder.add_body(
        position=[0.0, 2.0, 0.0],
        mass=1.0,
        shape_type=ShapeType.BOX,
        shape_params=[0.5, 0.5, 0.5],
        friction=0.5
    )
    
    body_data = builder.build()
    print("Created scene with", len(body_data['x']), "bodies")
    
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
    
    # Run one step
    print("Running physics step...")
    engine.step()
    print("Step complete!")
    
    # Check positions
    x_np = engine.x.numpy()
    print("Positions after step:")
    for i, pos in enumerate(x_np):
        print(f"  Body {i}: {pos}")

if __name__ == "__main__":
    main()