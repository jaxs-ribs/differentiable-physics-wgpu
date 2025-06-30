#!/usr/bin/env python3
"""Main entry point for running physics simulations and rendering.

This script provides a centralized command-line interface for controlling
the physics engine, including selecting different backends (naive, C, WebGPU),
configuring simulation parameters, and launching the renderer.

Usage:
    # Run naive simulation and render the result
    python3 run.py --naive

    # Run C-accelerated simulation with 500 steps, without rendering
    python3 run.py --c --steps 500 --no-render

    # Run naive and C simulations and render them together for comparison
    python3 run.py --naive --c

    # Render the most recent simulation file(s)
    python3 run.py
"""
import argparse
import subprocess
import sys
from pathlib import Path

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(text):
    """Print a section header."""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{text:^60}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

def run_simulation(args):
    """Run the physics simulation using specified modes."""
    print_header("Running Physics Simulation")
    
    modes = []
    if args.naive: modes.append('naive')
    if args.c: modes.append('c')
    if args.webgpu: modes.append('webgpu')
    
    cmd = [
        "python3", "scripts/run_simulation.py",
        "--modes", *modes,
        "--steps", str(args.steps),
    ]
    
    if args.save_all_frames:
        cmd.append("--save-all-frames")
        
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\n{RED}Simulation script failed!{RESET}")
        sys.exit(1)
        
    return modes

def run_renderer(modes):
    """Run the renderer to visualize simulation outputs."""
    print_header("Launching Renderer")
    
    renderer_path = "./renderer/target/release/renderer"
    if not Path(renderer_path).exists():
        print(f"{RED}Renderer executable not found at {renderer_path}{RESET}")
        print("Please build the renderer first: cd renderer && cargo build --release")
        sys.exit(1)

    # Find the latest .npy files corresponding to the executed modes
    artifact_dir = Path("artifacts")
    all_files = sorted(artifact_dir.glob("*.npy"), key=lambda p: p.stat().st_mtime, reverse=True)
    
    files_to_render = []
    if not modes: # Render-only mode
        if not all_files:
            print(f"{RED}No .npy files found in artifacts/ to render.{RESET}")
            sys.exit(1)
        files_to_render.append(all_files[0])
    else:
        # This is a simplification. A more robust solution would be to get the output paths
        # directly from the simulation script.
        files_to_render = all_files[:len(modes)]

    # Define colors for different modes
    colors = {
        "naive": ("0.2", "0.5", "1.0"),
        "c": ("1.0", "0.9", "0.2"),
        "webgpu": ("0.8", "0.3", "1.0"),
    }
    
    cmd = [renderer_path]
    for i, file_path in enumerate(files_to_render):
        mode = modes[i] if i < len(modes) else "oracle" # Default to oracle if mode not found
        color = colors.get(mode, ("1.0", "1.0", "1.0"))
        
        # This part of the command structure is based on the plan in task.md
        # It assumes the renderer will be updated to handle these arguments.
        cmd.extend([str(file_path), "--color", *color, "--alpha", "0.7"])

    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="Physics Engine Runner")
    parser.add_argument("--naive", action="store_true", help="Run the pure Python 'Oracle' engine")
    parser.add_argument("--c", action="store_true", help="Run the C-accelerated engine")
    parser.add_argument("--webgpu", action="store_true", help="Run the WebGPU engine (not implemented)")
    parser.add_argument("--steps", type=int, default=200, help="Number of simulation steps")
    parser.add_argument("--save-all-frames", action="store_true", help="Save every frame of the simulation")
    parser.add_argument("--no-render", action="store_true", help="Do not run the renderer after simulation")
    
    args = parser.parse_args()
    
    # Simulation mode if any engine is selected
    if args.naive or args.c or args.webgpu:
        executed_modes = run_simulation(args)
        if not args.no_render:
            run_renderer(executed_modes)
    # Render-only mode if no engine is selected
    else:
        run_renderer([])

if __name__ == "__main__":
    main()
