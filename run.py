#!/usr/bin/env python3
"""Main entry point for running physics simulations with video rendering.

This script provides a complete pipeline for physics simulation and visualization:
simulate → collect trajectory → render video.
"""
import sys
import os
from pathlib import Path
import numpy as np

# Add parent directory to path
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)

# Import modular components
from scripts.cli_parser import create_argument_parser
from scripts.config import SimulationConfig, RenderingConfig, OutputConfig
from scripts.file_operations import (
    ensure_initial_state_exists,
    generate_timestamped_filename,
    save_numpy_array,
    create_temporary_file,
    safe_delete_file,
    extract_final_state
)
from scripts.output_handler import SimulationOutputHandler
from scripts.run_simulation import SimulationRunner
from scripts.renderer import RendererInvoker
from scripts.error_handler import ErrorHandler, SimulationError


def run_simulation_pipeline(sim_config: SimulationConfig, 
                           render_config: RenderingConfig,
                           output_config: OutputConfig,
                           output_handler: SimulationOutputHandler) -> None:
    """Execute the complete simulation pipeline.
    
    Args:
        sim_config: Simulation configuration
        render_config: Rendering configuration
        output_config: Output configuration
        output_handler: Output handler for console messages
    """
    # Run simulation
    runner = SimulationRunner(
        mode=sim_config.mode.value,
        steps=sim_config.steps,
        input_file_path=str(sim_config.input_file),
        enable_profiling=sim_config.enable_profiling,
        collect_trajectory=sim_config.collect_trajectory
    )
    
    output_handler.print_simulation_start(
        sim_config.mode.value, 
        sim_config.steps,
        render_config.enabled
    )
    
    simulation_data, metrics = runner.run()
    
    output_handler.print_simulation_complete(metrics)
    output_handler.print_profiling_data(metrics.get('profile_data'))
    
    # Handle rendering if enabled
    if render_config.enabled:
        render_video(simulation_data, sim_config, render_config, output_handler)
    
    # Save final state if requested
    if output_config.final_state_output:
        save_final_state(simulation_data, output_config, sim_config, output_handler)


def render_video(simulation_data, sim_config, render_config, output_handler):
    """Render simulation trajectory to video.
    
    Args:
        simulation_data: Trajectory data from simulation
        sim_config: Simulation configuration
        render_config: Rendering configuration
        output_handler: Output handler for console messages
    """
    # Prepare video output path
    if render_config.video_output:
        video_path = render_config.video_output
    else:
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        video_path = artifacts_dir / generate_timestamped_filename("simulation", "mp4")
    
    video_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save trajectory to temporary file
    tmp_file = create_temporary_file('.npy')
    tmp_trajectory_path = tmp_file.name
    
    try:
        # Format and save trajectory
        formatted_trajectory = RendererInvoker.format_trajectory_for_renderer(simulation_data)
        save_numpy_array(formatted_trajectory, Path(tmp_trajectory_path))
        
        output_handler.print_video_rendering_start(video_path)
        
        # Calculate video duration
        physics_timestep = 0.016  # TODO: Get from physics constants
        if render_config.duration:
            video_duration = render_config.duration
        else:
            video_duration = sim_config.steps * physics_timestep
            output_handler.print_video_duration(
                video_duration, sim_config.steps, physics_timestep
            )
        
        # Create and run renderer
        renderer = RendererInvoker(
            trajectory_path=tmp_trajectory_path,
            video_output_path=str(video_path),
            duration=video_duration,
            fps=render_config.fps
        )
        
        use_gpu_flag = (sim_config.mode.value == "c")
        success = renderer.render(use_gpu_flag=use_gpu_flag)
        
        if success:
            output_handler.print_video_saved(video_path)
        else:
            output_handler.print_rendering_failed()
    
    finally:
        safe_delete_file(tmp_trajectory_path)


def save_final_state(simulation_data, output_config, sim_config, output_handler):
    """Save the final simulation state to file.
    
    Args:
        simulation_data: Simulation trajectory or final state
        output_config: Output configuration
        sim_config: Simulation configuration
        output_handler: Output handler for console messages
    """
    final_state = extract_final_state(
        simulation_data, 
        sim_config.collect_trajectory
    )
    
    save_numpy_array(final_state, output_config.final_state_output)
    output_handler.print_final_state_saved(output_config.final_state_output)


def main():
    """Main entry point for physics simulation and rendering."""
    # Initialize error handler
    error_handler = ErrorHandler(verbose=True)
    
    try:
        # Parse command-line arguments
        parser = create_argument_parser()
        args = parser.parse_args()
        
        # Create configurations from arguments
        sim_config = SimulationConfig.from_args(args)
        render_config = RenderingConfig.from_args(args)
        output_config = OutputConfig.from_args(args)
        
        # Create output handler
        output_handler = SimulationOutputHandler(verbose=True)
        
        # Ensure initial state exists
        with error_handler.error_context("ensure initial state exists"):
            ensure_initial_state_exists(sim_config.input_file)
        
        # Run the simulation pipeline
        with error_handler.error_context("run simulation pipeline"):
            run_simulation_pipeline(sim_config, render_config, output_config, output_handler)
            
    except SimulationError as e:
        error_handler.handle_error(e, critical=True)
    except Exception as e:
        error_handler.handle_error(e, critical=True)

if __name__ == "__main__":
    main()