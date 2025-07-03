import argparse


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="XPBD Physics Engine - Simulation and rendering pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_get_usage_examples()
    )
    
    _add_simulation_arguments(parser)
    _add_rendering_arguments(parser)
    _add_output_arguments(parser)
    
    return parser


def _get_usage_examples() -> str:
    return """
Usage Examples:
    # Run with defaults (200 steps, render video)
    python3 run.py
    
    # Run custom simulation with video output
    python3 run.py --steps 500 --video-output my_simulation.mp4
    
    # Run without rendering (performance testing)
    python3 run.py --no-render
    
    # Custom video settings
    python3 run.py --video-fps 60
"""


def _add_simulation_arguments(parser: argparse.ArgumentParser) -> None:
    simulation_group = parser.add_argument_group('Simulation Options')
    
    simulation_group.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of simulation steps (default: 200)"
    )
    
    simulation_group.add_argument(
        "--dt",
        type=float,
        default=0.016,
        help="Timestep in seconds (default: 0.016)"
    )
    
    simulation_group.add_argument(
        "--gravity",
        type=float,
        default=-9.81,
        help="Gravity acceleration (default: -9.81)"
    )
    
    simulation_group.add_argument(
        "--restitution",
        type=float,
        default=0.1,
        help="Restitution coefficient (default: 0.1)"
    )


def _add_rendering_arguments(parser: argparse.ArgumentParser) -> None:
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
        "--video-fps",
        type=int,
        default=30,
        help="Video frames per second (default: 30)"
    )


def _add_output_arguments(parser: argparse.ArgumentParser) -> None:
    output_group = parser.add_argument_group('Output Options')
    
    output_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )