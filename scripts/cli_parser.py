"""Command-line interface parser for the physics simulation.

Follows Single Responsibility Principle - only handles CLI parsing.
"""
import argparse
from typing import Optional
from .config import PhysicsConstants


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Physics Engine - Simulation and rendering pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_get_usage_examples()
    )
    
    _add_simulation_arguments(parser)
    _add_rendering_arguments(parser)
    _add_output_arguments(parser)
    _add_debugging_arguments(parser)
    
    return parser


def _get_usage_examples() -> str:
    """Get formatted usage examples for the help text."""
    return """
Usage Examples:
    # Run with defaults (pure mode, 200 steps, render video)
    python3 run.py
    
    # Run C-accelerated simulation with custom video output
    python3 run.py --mode c --steps 500 --video-output my_simulation.mp4
    
    # Run without rendering (performance testing)
    python3 run.py --no-render
    
    # Save final state for checkpointing
    python3 run.py --final-state-output checkpoint.npy
    
    # Custom video settings
    python3 run.py --video-duration 10.0 --video-fps 30
    
    # Run with profiling
    python3 run.py --profile
"""


def _add_simulation_arguments(parser: argparse.ArgumentParser) -> None:
    """Add simulation-related arguments to the parser."""
    simulation_group = parser.add_argument_group('Simulation Options')
    
    simulation_group.add_argument(
        "--mode",
        type=str,
        default="pure",
        choices=["pure", "c", "wgpu"],
        help="Physics execution backend (default: pure)"
    )
    
    simulation_group.add_argument(
        "--steps",
        type=int,
        default=PhysicsConstants.DEFAULT_STEPS,
        help=f"Number of simulation steps (default: {PhysicsConstants.DEFAULT_STEPS})"
    )
    
    simulation_group.add_argument(
        "--input-file",
        type=str,
        default=PhysicsConstants.DEFAULT_INPUT_FILE,
        help=f"Path to initial state .npy file (default: {PhysicsConstants.DEFAULT_INPUT_FILE})"
    )


def _add_rendering_arguments(parser: argparse.ArgumentParser) -> None:
    """Add rendering-related arguments to the parser."""
    rendering_group = parser.add_argument_group('Rendering Options')
    
    rendering_group.add_argument(
        "--no-render",
        action="store_true",
        help="Disable video rendering (for performance testing)"
    )
    
    rendering_group.add_argument(
        "--video-output",
        type=str,
        help="Path to save rendered video (default: artifacts/simulation_TIMESTAMP.mp4)"
    )
    
    rendering_group.add_argument(
        "--video-duration",
        type=float,
        help="Video duration in seconds (default: auto-calculated from simulation)"
    )
    
    rendering_group.add_argument(
        "--video-fps",
        type=int,
        default=PhysicsConstants.DEFAULT_FPS,
        help=f"Video frames per second (default: {PhysicsConstants.DEFAULT_FPS})"
    )


def _add_output_arguments(parser: argparse.ArgumentParser) -> None:
    """Add output-related arguments to the parser."""
    output_group = parser.add_argument_group('Output Options')
    
    output_group.add_argument(
        "--final-state-output",
        type=str,
        help="Path to save final simulation state .npy file (optional)"
    )


def _add_debugging_arguments(parser: argparse.ArgumentParser) -> None:
    """Add debugging-related arguments to the parser."""
    debug_group = parser.add_argument_group('Debugging Options')
    
    debug_group.add_argument(
        "--profile",
        action="store_true",
        help="Enable performance profiling"
    )