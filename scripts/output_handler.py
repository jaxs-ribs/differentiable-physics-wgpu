"""Output handling module for simulation results.

Handles printing results and saving outputs following Clean Code principles.
"""
from typing import Dict, Any, Optional
from pathlib import Path
import sys


class SimulationOutputHandler:
    """Handles all output operations for simulation results."""
    
    def __init__(self, verbose: bool = True):
        """Initialize the output handler.
        
        Args:
            verbose: Whether to print detailed output
        """
        self.verbose = verbose
    
    def print_simulation_start(self, mode: str, steps: int, 
                               render_enabled: bool) -> None:
        """Print simulation start message.
        
        Args:
            mode: Execution mode
            steps: Number of steps
            render_enabled: Whether rendering is enabled
        """
        if not self.verbose:
            return
            
        print(f"Running {mode} mode simulation for {steps} steps...")
        if not render_enabled:
            print("Rendering disabled - trajectory collection skipped")
    
    def print_simulation_complete(self, metrics: Dict[str, Any]) -> None:
        """Print simulation completion summary.
        
        Args:
            metrics: Simulation metrics dictionary
        """
        if not self.verbose:
            return
            
        print("\nSimulation Complete!")
        print(f"Total time: {metrics['total_time']:.3f} seconds")
        print(f"Steps per second: {metrics['steps_per_second']:.1f}")
    
    def print_profiling_data(self, profile_data: Dict[str, Any]) -> None:
        """Print profiling data if available.
        
        Args:
            profile_data: Profiling metrics dictionary
        """
        if not self.verbose or not profile_data:
            return
            
        print("\nProfiling data:")
        for key, value in profile_data.items():
            print(f"  {key}: {value}")
    
    def print_video_rendering_start(self, video_path: Path) -> None:
        """Print video rendering start message.
        
        Args:
            video_path: Path where video will be saved
        """
        if not self.verbose:
            return
            
        print(f"\nRendering video to: {video_path}")
    
    def print_video_duration(self, duration: float, steps: int, 
                             timestep: float) -> None:
        """Print calculated video duration.
        
        Args:
            duration: Video duration in seconds
            steps: Number of simulation steps
            timestep: Physics timestep
        """
        if not self.verbose:
            return
            
        print(f"Video duration: {duration:.2f}s ({steps} steps × {timestep}s)")
    
    def print_video_saved(self, video_path: Path) -> None:
        """Print video saved message.
        
        Args:
            video_path: Path where video was saved
        """
        if not self.verbose:
            return
            
        print(f"Video saved to: {video_path}")
    
    def print_rendering_failed(self) -> None:
        """Print rendering failure warning."""
        print("Warning: Video rendering failed", file=sys.stderr)
    
    def print_final_state_saved(self, output_path: Path) -> None:
        """Print final state saved message.
        
        Args:
            output_path: Path where state was saved
        """
        if not self.verbose:
            return
            
        print(f"\nFinal state saved to: {output_path}")
    
    def print_error(self, message: str) -> None:
        """Print error message to stderr.
        
        Args:
            message: Error message
        """
        print(f"Error: {message}", file=sys.stderr)
    
    def print_initial_state_info(self, filepath: Path, shape: tuple) -> None:
        """Print initial state information.
        
        Args:
            filepath: Path to initial state file
            shape: Shape of the state array
        """
        if not self.verbose:
            return
            
        print(f"Loaded initial state from {filepath}")
        print(f"  Shape: {shape}")