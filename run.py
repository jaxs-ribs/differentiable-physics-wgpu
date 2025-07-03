#!/usr/bin/env python3
"""XPBD physics simulation runner with video rendering.

Pipeline: create_scene → run_simulation → collect_trajectory → render_video
"""
import sys
import os
import time
import numpy as np
from pathlib import Path

# Add physics_core to path
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)

# Add tinygrad to path
tinygrad_path = os.path.join(parent_dir, "external", "tinygrad")
if os.path.exists(tinygrad_path):
    sys.path.insert(0, tinygrad_path)

from physics.engine import TensorPhysicsEngine
from physics.types import ShapeType, create_body_array_defaults
from scripts.cli_parser import create_argument_parser
from scripts.file_operations import (
    generate_timestamped_filename,
    save_numpy_array,
    ensure_directory_exists
)
from scripts.renderer import RendererInvoker


def create_default_scene():
    """Create a simple test scene with falling objects."""
    bodies_list = []
    
    # Ground plane (static)
    ground = create_body_array_defaults(
        position=np.array([0, -5, 0], dtype=np.float32),
        mass=1e8,  # Effectively static
        shape_type=ShapeType.BOX,
        shape_params=np.array([10, 0.5, 10], dtype=np.float32)
    )
    bodies_list.append(ground)
    
    # Falling sphere
    sphere = create_body_array_defaults(
        position=np.array([0, 5, 0], dtype=np.float32),
        velocity=np.array([1, 0, 0], dtype=np.float32),
        mass=1.0,
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([1, 0, 0], dtype=np.float32)
    )
    bodies_list.append(sphere)
    
    # Falling box
    box = create_body_array_defaults(
        position=np.array([3, 8, 0], dtype=np.float32),
        mass=2.0,
        shape_type=ShapeType.BOX,
        shape_params=np.array([1, 1, 1], dtype=np.float32)
    )
    bodies_list.append(box)
    
    return np.stack(bodies_list)


def run_simulation_with_trajectory(engine: TensorPhysicsEngine, steps: int, verbose: bool = False):
    trajectory = []
    
    # Collect initial state
    trajectory.append(engine.get_state().copy())
    
    print(f"Running simulation for {steps} steps...")
    start_time = time.time()
    
    try:
        for i in range(steps):
            engine.step()
            trajectory.append(engine.get_state().copy())
            
            if verbose and (i + 1) % 50 == 0:
                print(f"  Step {i + 1}/{steps}")
    
    except Exception as e:
        print(f"Simulation failed (expected with XPBD placeholders): {e}")
        return None, 0
    
    elapsed = time.time() - start_time
    
    # Convert to trajectory array (steps, bodies, properties)
    trajectory_array = np.stack(trajectory)
    
    print(f"Simulation completed in {elapsed:.3f}s")
    print(f"Average step time: {elapsed/steps*1000:.2f}ms")
    print(f"Trajectory shape: {trajectory_array.shape}")
    
    return trajectory_array, elapsed


def render_video(trajectory_path: Path, video_output_path: Path, 
                duration: float, fps: int, verbose: bool = False):
    try:
        renderer = RendererInvoker()
        
        print(f"Rendering video...")
        print(f"  Input: {trajectory_path}")
        print(f"  Output: {video_output_path}")
        print(f"  Duration: {duration:.2f}s @ {fps} fps")
        
        success = renderer.render_video(
            trajectory_path=trajectory_path,
            output_path=video_output_path,
            duration=duration,
            fps=fps,
            verbose=verbose
        )
        
        if success:
            print(f"✓ Video rendered successfully: {video_output_path}")
            return True
        else:
            print("✗ Video rendering failed")
            return False
            
    except Exception as e:
        print(f"✗ Rendering error: {e}")
        return False


def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    
    print("=" * 60)
    print("XPBD Physics Engine - Simulation & Rendering Pipeline")
    print("=" * 60)
    
    # Create scene
    print("Creating scene...")
    bodies = create_default_scene()
    gravity = np.array([0, args.gravity, 0], dtype=np.float32)
    
    # Create engine
    print("Initializing XPBD engine...")
    engine = TensorPhysicsEngine(
        bodies, 
        gravity=gravity, 
        dt=args.dt, 
        restitution=args.restitution
    )
    
    print(f"  Bodies: {bodies.shape[0]}")
    print(f"  dt: {args.dt:.4f}s")
    print(f"  gravity: {args.gravity:.2f} m/s²")
    print(f"  restitution: {args.restitution:.2f}")
    
    # Run simulation
    trajectory, sim_time = run_simulation_with_trajectory(
        engine, args.steps, args.verbose
    )
    
    if trajectory is None:
        print("✗ Simulation failed, no trajectory to render")
        return 1
    
    # Save trajectory
    artifacts_dir = Path("artifacts")
    ensure_directory_exists(artifacts_dir)
    
    trajectory_filename = generate_timestamped_filename("trajectory", "npy")
    trajectory_path = artifacts_dir / trajectory_filename
    
    print(f"Saving trajectory to {trajectory_path}")
    save_numpy_array(trajectory, trajectory_path)
    
    # Render video (unless disabled)
    if not args.no_render:
        video_filename = args.video_output or generate_timestamped_filename("simulation", "mp4")
        video_path = artifacts_dir / video_filename
        
        # Auto-calculate video duration from simulation
        duration = args.steps * args.dt
        
        success = render_video(
            trajectory_path=trajectory_path,
            video_output_path=video_path,
            duration=duration,
            fps=args.video_fps,
            verbose=args.verbose
        )
        
        if not success:
            print("Video rendering failed, but trajectory was saved")
            return 1
    else:
        print("Rendering disabled, trajectory saved only")
    
    print("\n✓ Pipeline completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())