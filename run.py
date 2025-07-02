#!/usr/bin/env python3
"""Main entry point for running physics simulations with video rendering.

This script provides a complete pipeline for physics simulation and visualization:
simulate → collect trajectory → render video.

Usage:
    # Run with defaults (pure mode, 200 steps, render video)
    python3 run.py
    
    # Run C-accelerated simulation with custom video output
    python3 run.py --mode c --steps 500 --video-output my_simulation.mp4
    
    # Run without rendering (performance testing)
    python3 run.py --no-render
    
    # Save final state for checkpointing
    python3 run.py --final-state-output checkpoint.npy
"""
import argparse
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import tempfile
import os

# Add parent directory to path
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)

# Import the simulation runner and renderer
from scripts.run_simulation import SimulationRunner
from scripts.renderer import RendererInvoker

def get_timestamped_filename(prefix: str = "simulation", extension: str = "mp4") -> str:
    """Generate a timestamped filename."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{prefix}_{timestamp}.{extension}"

def main():
    """Main entry point for physics simulation and rendering."""
    # Define argument parser with defaults
    parser = argparse.ArgumentParser(
        description="Physics Engine - Simulation and rendering pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="pure",
        choices=["pure", "c", "wgpu"],
        help="Physics execution backend (default: pure)"
    )
    
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of simulation steps (default: 200)"
    )
    
    parser.add_argument(
        "--input-file",
        type=str,
        default="artifacts/initial_state.npy",
        help="Path to initial state .npy file (default: artifacts/initial_state.npy)"
    )
    
    parser.add_argument(
        "--final-state-output",
        type=str,
        help="Path to save final simulation state .npy file (optional)"
    )
    
    parser.add_argument(
        "--video-output",
        type=str,
        help=f"Path to save rendered video (default: artifacts/simulation_TIMESTAMP.mp4)"
    )
    
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable video rendering (for performance testing)"
    )
    
    parser.add_argument(
        "--video-duration",
        type=float,
        default=5.0,
        help="Video duration in seconds (default: 5.0)"
    )
    
    parser.add_argument(
        "--video-fps",
        type=int,
        default=60,
        help="Video frames per second (default: 60)"
    )
    
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable performance profiling"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate input file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' not found.")
        sys.exit(1)
    
    # Determine if we need to collect trajectory
    collect_trajectory = not args.no_render
    
    # Create simulation runner with resolved configuration
    runner = SimulationRunner(
        mode=args.mode,
        steps=args.steps,
        input_file_path=args.input_file,
        enable_profiling=args.profile,
        collect_trajectory=collect_trajectory
    )
    
    # Run the simulation
    print(f"Running {args.mode} mode simulation for {args.steps} steps...")
    if args.no_render:
        print("Rendering disabled - trajectory collection skipped")
    
    simulation_data, metrics = runner.run()
    
    # Print summary statistics
    print("\nSimulation Complete!")
    print(f"Total time: {metrics['total_time']:.3f} seconds")
    print(f"Steps per second: {metrics['steps_per_second']:.1f}")
    
    if args.profile and 'profile_data' in metrics:
        print("\nProfiling data:")
        for key, value in metrics['profile_data'].items():
            print(f"  {key}: {value}")
    
    # Handle rendering if enabled
    if not args.no_render:
        # Prepare video output path
        if args.video_output:
            video_path = Path(args.video_output)
        else:
            artifacts_dir = Path("artifacts")
            artifacts_dir.mkdir(exist_ok=True)
            video_path = artifacts_dir / get_timestamped_filename("simulation", "mp4")
        
        # Create parent directory if needed
        video_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save trajectory to temporary file
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp_file:
            tmp_trajectory_path = tmp_file.name
            
            # Format trajectory for renderer
            formatted_trajectory = RendererInvoker.format_trajectory_for_renderer(simulation_data)
            np.save(tmp_trajectory_path, formatted_trajectory)
            
            print(f"\nRendering video to: {video_path}")
            
            # Create renderer invoker
            renderer = RendererInvoker(
                trajectory_path=tmp_trajectory_path,
                video_output_path=str(video_path),
                duration=args.video_duration,
                fps=args.video_fps
            )
            
            # Render the video
            # Use --gpu flag for C mode, --oracle for others
            use_gpu_flag = (args.mode == "c")
            success = renderer.render(use_gpu_flag=use_gpu_flag)
            
            # Clean up temporary file
            try:
                os.unlink(tmp_trajectory_path)
            except:
                pass
            
            if not success:
                print("Warning: Video rendering failed", file=sys.stderr)
    
    # Save final state if requested
    if args.final_state_output:
        output_path = Path(args.final_state_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract final state from trajectory or use directly
        if collect_trajectory:
            final_state = simulation_data[-1]  # Last frame
        else:
            final_state = simulation_data
        
        np.save(output_path, final_state)
        print(f"\nFinal state saved to: {output_path}")

if __name__ == "__main__":
    main()