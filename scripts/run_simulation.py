#!/usr/bin/env python3
"""Simulation workhorse for the physics engine.

This script is responsible for:
- Defining the common interface for all physics engines.
- Implementing wrappers for each engine (Naive, C, WebGPU).
- Setting up a consistent initial scene for simulations.
- Running the simulation for a specified number of steps.
- Saving the results to the artifacts/ directory.
"""
import argparse
import numpy as np
import time
from pathlib import Path

# Add parent directory to path to import physics modules
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Add tinygrad to path
tinygrad_path = os.path.join(parent_dir, "external", "tinygrad")
if os.path.exists(tinygrad_path):
    sys.path.insert(0, tinygrad_path)

from physics.engine import TensorPhysicsEngine
from physics.engine_c import CEngine 

class PhysicsEngine:
    """Abstract base class for a physics engine wrapper."""
    def __init__(self, bodies: np.ndarray, dt: float = 0.01):
        raise NotImplementedError

    def step(self) -> None:
        raise NotImplementedError

    def get_state(self) -> np.ndarray:
        raise NotImplementedError

class NaiveEngine(PhysicsEngine):
    """Wrapper for the pure Python 'Oracle' engine."""
    def __init__(self, bodies: np.ndarray, dt: float = 0.01):
        self.engine = TensorPhysicsEngine(bodies, dt=dt)

    def step(self) -> None:
        self.engine.step()

    def get_state(self) -> np.ndarray:
        return self.engine.get_state()



class WebGPUEngine(PhysicsEngine):
    """Placeholder for the future WebGPU engine."""
    def __init__(self, bodies: np.ndarray, dt: float = 0.01):
        print("WebGPU Engine not implemented.")
        self.bodies = bodies

    def step(self) -> None:
        pass

    def get_state(self) -> np.ndarray:
        return self.bodies

ENGINE_MAP = {
    'naive': NaiveEngine,
    'c': CEngine,
    'webgpu': WebGPUEngine,
}

def create_default_scene() -> np.ndarray:
    """Creates a consistent set of initial bodies for simulations."""
    from physics.types import create_body_array, ShapeType
    bodies = []
    # Add a static ground box
    bodies.append(create_body_array(
        position=np.array([0, -5, 0]),
        mass=1e9,
        shape_type=ShapeType.BOX,
        shape_params=np.array([20, 1, 20])
    ))
    # Add some dynamic spheres
    for i in range(5):
        bodies.append(create_body_array(
            position=np.array([-2.0 + i, 5.0, 0.0]),
            mass=1.0,
            shape_type=ShapeType.SPHERE,
            shape_params=np.array([0.5, 0, 0])
        ))
    return np.stack(bodies)

def main():
    parser = argparse.ArgumentParser(description="Physics Simulation Workhorse")
    parser.add_argument("--modes", nargs='+', required=True, help="List of engine modes to run (e.g., naive c)")
    parser.add_argument("--steps", type=int, required=True, help="Number of simulation steps")
    parser.add_argument("--save-all-frames", action="store_true", help="Save all frames, not just the first and last")
    
    args = parser.parse_args()
    
    initial_bodies = create_default_scene()
    
    for mode in args.modes:
        if mode not in ENGINE_MAP:
            print(f"Unknown engine mode: {mode}")
            continue
            
        print(f"Running simulation for mode: {mode}")
        engine_class = ENGINE_MAP[mode]
        engine = engine_class(initial_bodies.copy())
        
        states = [engine.get_state()]
        
        start_time = time.time()
        for _ in range(args.steps):
            engine.step()
            if args.save_all_frames:
                states.append(engine.get_state())
        
        if not args.save_all_frames:
            states.append(engine.get_state())
            
        end_time = time.time()
        print(f"Simulation took {end_time - start_time:.2f} seconds.")
        
        # Save results
        output_dir = Path("artifacts")
        output_dir.mkdir(exist_ok=True)
        timestamp = int(time.time())
        filename = f"{mode}_sim_{args.steps}steps_{timestamp}.npy"
        if args.save_all_frames:
            filename = f"{mode}_sim_{args.steps}steps_{timestamp}_all.npy"
        
        output_path = output_dir / filename
        np.save(output_path, np.array(states))
        print(f"Saved simulation to: {output_path}")

if __name__ == "__main__":
    main()
