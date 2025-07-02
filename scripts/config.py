"""Configuration classes for the physics simulation pipeline.

Following Clean Code principles:
- Single Responsibility: Each class handles one aspect of configuration
- Open/Closed: Easy to extend with new config options
- Interface Segregation: Small, focused interfaces
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from enum import Enum


class ExecutionMode(Enum):
    """Supported physics execution backends."""
    PURE = "pure"
    C = "c"
    WGPU = "wgpu"


@dataclass
class SimulationConfig:
    """Configuration for physics simulation."""
    mode: ExecutionMode
    steps: int
    input_file: Path
    enable_profiling: bool = False
    collect_trajectory: bool = True
    
    @classmethod
    def from_args(cls, args) -> 'SimulationConfig':
        """Create config from command-line arguments."""
        return cls(
            mode=ExecutionMode(args.mode),
            steps=args.steps,
            input_file=Path(args.input_file),
            enable_profiling=args.profile,
            collect_trajectory=not args.no_render
        )


@dataclass
class RenderingConfig:
    """Configuration for video rendering."""
    enabled: bool
    video_output: Optional[Path]
    duration: Optional[float]
    fps: int
    
    @classmethod
    def from_args(cls, args) -> 'RenderingConfig':
        """Create config from command-line arguments."""
        return cls(
            enabled=not args.no_render,
            video_output=Path(args.video_output) if args.video_output else None,
            duration=args.video_duration,
            fps=args.video_fps
        )


@dataclass
class OutputConfig:
    """Configuration for output files."""
    final_state_output: Optional[Path]
    
    @classmethod
    def from_args(cls, args) -> 'OutputConfig':
        """Create config from command-line arguments."""
        return cls(
            final_state_output=Path(args.final_state_output) if args.final_state_output else None
        )


@dataclass
class PhysicsConstants:
    """Physics engine constants."""
    DEFAULT_TIMESTEP: float = 0.016  # 60 Hz
    DEFAULT_STEPS: int = 200
    DEFAULT_FPS: int = 60
    DEFAULT_INPUT_FILE: str = "artifacts/initial_state.npy"