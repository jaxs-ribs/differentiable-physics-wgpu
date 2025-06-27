#!/usr/bin/env python3
"""Main entry point for physics simulation with numpy dump output."""
import argparse
import numpy as np
from .types import create_body_array, ShapeType
from .engine import PhysicsEngine

def create_test_scene() -> list[np.ndarray]:
  """Create a simple test scene with a ground box and falling sphere."""
  bodies = []
  
  # Ground box (static)
  bodies.append(create_body_array(
    position=np.array([0., -2., 0.], dtype=np.float32), 
    velocity=np.zeros(3, dtype=np.float32),
    orientation=np.array([1., 0., 0., 0.], dtype=np.float32),  # Identity quaternion
    angular_vel=np.zeros(3, dtype=np.float32),
    mass=1e8,  # Very large mass = effectively static
    inertia=np.eye(3, dtype=np.float32) * 1e8,
    shape_type=ShapeType.BOX,
    shape_params=np.array([10., 0.5, 10.], dtype=np.float32)  # 20x1x20 ground plane
  ))
  
  # Falling sphere (dynamic)
  bodies.append(create_body_array(
    position=np.array([0., 5., 0.], dtype=np.float32),
    velocity=np.zeros(3, dtype=np.float32),
    orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
    angular_vel=np.zeros(3, dtype=np.float32),
    mass=1.0,
    inertia=np.eye(3, dtype=np.float32) * (2.0/5.0) * 1.0 * (0.5**2),  # Solid sphere: I = 2/5 * m * r²
    shape_type=ShapeType.SPHERE,
    shape_params=np.array([0.5, 0., 0.], dtype=np.float32)  # radius = 0.5
  ))
  
  return bodies

def main():
  parser = argparse.ArgumentParser(description='Physics engine simulation with numpy dump output.')
  parser.add_argument('--steps', type=int, default=200, help='Number of simulation steps')
  parser.add_argument('--output', type=str, default='oracle_dump.npy', help='Output numpy file')
  parser.add_argument('--dt', type=float, default=0.016, help='Timestep in seconds')
  args = parser.parse_args()
  
  # Create physics engine and scene
  engine = PhysicsEngine(dt=args.dt)
  bodies = create_test_scene()
  engine.set_bodies(bodies)
  
  # Run simulation and collect states
  all_states = []
  print(f"Running simulation for {args.steps} steps (dt={args.dt}s)...")
  
  for i in range(args.steps):
    if i % 20 == 0: print(f"  Step {i}...")
    engine.step()
    all_states.append(engine.get_state())
  
  # Save to numpy file
  state_array = np.array(all_states)
  np.save(args.output, state_array)
  
  print(f"\nCreated {args.output} with {len(bodies)} bodies over {args.steps} frames.")
  print(f"State array shape: {state_array.shape}")
  print(f"Each frame contains {len(bodies)} bodies with {state_array.shape[2]} properties each.")

if __name__ == "__main__":
  main()