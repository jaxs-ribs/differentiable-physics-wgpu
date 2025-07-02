#!/usr/bin/env python3
"""Create default initial state file for the physics simulation."""
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from physics.types import create_body_array_defaults, ShapeType

def create_default_scene():
    """Create a default scene with ground and falling spheres."""
    bodies = []
    
    # Add a static ground box
    bodies.append(create_body_array_defaults(
        position=np.array([0, -5, 0], dtype=np.float32),
        mass=1e9,  # Very large mass makes it effectively static
        shape_type=ShapeType.BOX,
        shape_params=np.array([20, 1, 20], dtype=np.float32)
    ))
    
    # Add some dynamic spheres
    for i in range(5):
        bodies.append(create_body_array_defaults(
            position=np.array([-2.0 + i, 5.0, 0.0], dtype=np.float32),
            mass=1.0,
            shape_type=ShapeType.SPHERE,
            shape_params=np.array([0.5, 0, 0], dtype=np.float32)
        ))
    
    return np.stack(bodies)

def main():
    """Generate and save the default initial state."""
    # Create artifacts directory if it doesn't exist
    artifacts_dir = Path(__file__).parent.parent / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    
    # Generate the default scene
    initial_state = create_default_scene()
    
    # Save to file
    output_path = artifacts_dir / "initial_state.npy"
    np.save(output_path, initial_state)
    
    print(f"Created default initial state: {output_path}")
    print(f"  Shape: {initial_state.shape}")
    print(f"  Bodies: {initial_state.shape[0]} (1 ground box + 5 spheres)")

if __name__ == "__main__":
    main()