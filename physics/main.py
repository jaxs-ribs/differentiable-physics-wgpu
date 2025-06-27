#!/usr/bin/env python3
"""Main entry point for physics simulation with numpy dump output."""
import argparse
import numpy as np
from .types import create_body_array, ShapeType
from .engine import TensorPhysicsEngine

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
  parser.add_argument('--output', type=str, default='artifacts/oracle_dump.npy', help='Output numpy file')
  parser.add_argument('--dt', type=float, default=0.016, help='Timestep in seconds')
  parser.add_argument('--mode', type=str, default='nstep', choices=['single', 'nstep'], 
                      help='Simulation mode: single-step or n-step JIT')
  parser.add_argument('--save-intermediate', action='store_true', 
                      help='Save intermediate frames (only for single-step mode)')
  args = parser.parse_args()
  
  # Create physics engine and scene
  bodies = create_test_scene()
  bodies_array = np.stack(bodies)
  engine = TensorPhysicsEngine(bodies_array, dt=args.dt)
  
  # Run simulation based on mode
  if args.mode == 'nstep':
    print(f"Running N-step JIT simulation for {args.steps} steps (dt={args.dt}s)...")
    
    # Capture initial state
    initial_state = engine.get_state()
    
    # Run the entire simulation as a single JIT-compiled operation
    engine.run_simulation(args.steps)
    
    # Get final state
    final_state = engine.get_state()
    
    # For N-step mode, we only save initial and final states
    all_states = [initial_state, final_state]
  else:  # single-step mode
    print(f"Running single-step simulation for {args.steps} steps (dt={args.dt}s)...")
    all_states = []
    
    if args.save_intermediate:
      # Save all intermediate states
      for i in range(args.steps + 1):  # +1 to include initial state
        if i % 20 == 0: print(f"  Step {i}...")
        if i > 0:
          engine.step()
        all_states.append(engine.get_state())
    else:
      # Only save initial and final states
      initial_state = engine.get_state()
      for i in range(args.steps):
        if i % 20 == 0: print(f"  Step {i}...")
        engine.step()
      final_state = engine.get_state()
      all_states = [initial_state, final_state]
  
  state_array = np.array(all_states)
  # Ensure output directory exists
  import os
  os.makedirs(os.path.dirname(args.output), exist_ok=True)
  np.save(args.output, state_array)
  
  print(f"\nCreated {args.output} with {len(bodies)} bodies.")
  print(f"State array shape: {state_array.shape}")
  if args.mode == 'nstep':
    print(f"Simulation ran {args.steps} steps as a single JIT-compiled operation.")
    print("(Saved initial and final states only)")
  else:
    if args.save_intermediate:
      print(f"Simulation ran {args.steps} steps in single-step mode with all frames saved.")
    else:
      print(f"Simulation ran {args.steps} steps in single-step mode.")
      print("(Saved initial and final states only)")
  print(f"Each state contains {len(bodies)} bodies with {state_array.shape[2]} properties each.")

if __name__ == "__main__":
  main()