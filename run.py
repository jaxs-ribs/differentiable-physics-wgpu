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
from physics.types import ShapeType
from scripts.cli_parser import create_argument_parser
from scripts.file_operations import (
    generate_timestamped_filename,
    save_numpy_array,
    ensure_directory_exists
)
from scripts.renderer import RendererInvoker
from scripts.scene_builder import SceneBuilder


def create_default_scene():
    """Create a simple test scene with falling objects."""
    builder = SceneBuilder()
    
    # Ground plane (static)
    builder.add_body(
        position=[0, -5, 0],
        mass=1e8,  # Effectively static
        shape_type=ShapeType.BOX,
        shape_params=[10, 0.5, 10]  # Large flat box
    )
    
    # Falling sphere
    builder.add_body(
        position=[0, 5, 0],
        velocity=[1, 0, 0],  # Moving right
        mass=1.0,
        shape_type=ShapeType.SPHERE,
        shape_params=[1, 0, 0]  # Radius = 1
    )
    
    # Falling box
    builder.add_body(
        position=[3, 8, 0],
        mass=2.0,
        shape_type=ShapeType.BOX,
        shape_params=[1, 1, 1]  # Unit cube
    )
    
    return builder.build()


def run_simulation_with_trajectory(engine: TensorPhysicsEngine, steps: int, verbose: bool = False):
    trajectory = []
    
    # Collect initial state
    trajectory.append(engine.get_state())
    
    print(f"Running simulation for {steps} steps...")
    start_time = time.time()
    
    try:
        for i in range(steps):
            engine.step()
            trajectory.append(engine.get_state())
            
            if verbose and (i + 1) % 50 == 0:
                print(f"  Step {i + 1}/{steps}")
    
    except Exception as e:
        print(f"Simulation failed (expected with XPBD placeholders): {e}")
        return None, 0
    
    elapsed = time.time() - start_time
    
    # Convert SoA trajectory to legacy format for renderer
    trajectory_array = convert_soa_to_legacy_trajectory(trajectory)
    
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
    soa_data = create_default_scene()
    gravity = np.array([0, args.gravity, 0], dtype=np.float32)
    
    # Create engine
    print("Initializing XPBD engine...")
    engine = TensorPhysicsEngine(
        x=soa_data['x'],
        q=soa_data['q'],
        v=soa_data['v'],
        omega=soa_data['omega'],
        inv_mass=soa_data['inv_mass'],
        inv_inertia=soa_data['inv_inertia'],
        shape_type=soa_data['shape_type'],
        shape_params=soa_data['shape_params'],
        gravity=gravity, 
        dt=args.dt, 
        restitution=args.restitution
    )
    
    print(f"  Bodies: {soa_data['x'].shape[0]}")
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


def convert_soa_to_legacy_trajectory(trajectory: list[dict[str, np.ndarray]]) -> np.ndarray:
    """Convert SoA trajectory format to legacy (N, bodies, 27) format for renderer."""
    n_steps = len(trajectory)
    n_bodies = trajectory[0]['x'].shape[0]
    
    # Create legacy trajectory array
    legacy_trajectory = np.zeros((n_steps, n_bodies, 27), dtype=np.float32)
    
    for step_idx, state in enumerate(trajectory):
        # Position (0:3)
        legacy_trajectory[step_idx, :, 0:3] = state['x']
        # Velocity (3:6)
        legacy_trajectory[step_idx, :, 3:6] = state['v']
        # Quaternion (6:10)
        legacy_trajectory[step_idx, :, 6:10] = state['q']
        # Angular velocity (10:13)
        legacy_trajectory[step_idx, :, 10:13] = state['omega']
        # Inverse mass (13)
        legacy_trajectory[step_idx, :, 13] = state['inv_mass']
        # Inverse inertia matrix (14:23) - flatten 3x3 to 9 elements
        legacy_trajectory[step_idx, :, 14:23] = state['inv_inertia'].reshape(n_bodies, 9)
        # Shape type (23)
        legacy_trajectory[step_idx, :, 23] = state['shape_type']
        # Shape parameters (24:27)
        legacy_trajectory[step_idx, :, 24:27] = state['shape_params']
    
    return legacy_trajectory


if __name__ == "__main__":
    sys.exit(main())