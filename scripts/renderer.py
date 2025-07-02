#!/usr/bin/env python3
"""Renderer invocation module for the physics engine.

This module provides clean interfaces for video rendering from simulation trajectories.
Follows Single Responsibility Principle - handles only rendering concerns.
"""
import subprocess
import sys
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum


class RenderMode(Enum):
    """Rendering backend modes."""
    GPU = "gpu"
    ORACLE = "oracle"


@dataclass
class RenderCommand:
    """Encapsulates a render command configuration."""
    renderer_path: Path
    trajectory_path: Path
    video_output_path: Path
    mode: RenderMode
    duration: float
    fps: int
    
    def to_args(self) -> List[str]:
        """Convert configuration to command-line arguments."""
        args = [str(self.renderer_path)]
        
        # Add simulation file with appropriate flag
        mode_flag = f"--{self.mode.value}"
        args.extend([mode_flag, str(self.trajectory_path)])
        
        # Add video recording parameters
        args.extend([
            "--record", str(self.video_output_path),
            "--duration", str(self.duration),
            "--fps", str(self.fps)
        ])
        
        return args


class RendererLocator:
    """Handles locating the renderer executable."""
    
    @staticmethod
    def find_renderer() -> Path:
        """Locate the renderer executable.
        
        Returns:
            Path to the renderer executable
            
        Raises:
            RuntimeError: If renderer not found
        """
        physics_core_dir = Path(__file__).parent.parent
        renderer_path = physics_core_dir / "renderer" / "target" / "release" / "renderer"
        
        if not renderer_path.exists():
            raise RuntimeError(
                f"Renderer executable not found at {renderer_path}\n"
                "Please build the renderer first: cd renderer && cargo build --release"
            )
        
        return renderer_path


class TrajectoryFormatter:
    """Handles trajectory format conversion for the renderer."""
    
    # Renderer expects 18 properties per body
    RENDERER_PROPS_PER_BODY = 18
    
    # Property indices in physics engine format
    POSITION_INDICES = slice(0, 3)
    VELOCITY_INDICES = slice(3, 6)
    ORIENTATION_INDICES = slice(6, 10)
    ANGULAR_VELOCITY_INDICES = slice(10, 13)
    INV_MASS_INDEX = 13
    SHAPE_TYPE_INDEX = 23
    SHAPE_PARAMS_INDICES = slice(24, 27)
    
    @classmethod
    def format_for_renderer(cls, trajectory: np.ndarray) -> np.ndarray:
        """Convert physics engine trajectory format to renderer format.
        
        Args:
            trajectory: Physics engine trajectory of shape (frames, bodies, 27)
            
        Returns:
            Formatted trajectory of shape (frames, bodies * 18)
        """
        frames, num_bodies, _ = trajectory.shape
        
        # Create output array
        renderer_data = np.zeros(
            (frames, num_bodies, cls.RENDERER_PROPS_PER_BODY), 
            dtype=np.float32
        )
        
        # Map properties
        cls._map_kinematics(trajectory, renderer_data)
        cls._map_mass(trajectory, renderer_data)
        cls._map_shape(trajectory, renderer_data)
        
        # Reshape to expected format
        return renderer_data.reshape(frames, num_bodies * cls.RENDERER_PROPS_PER_BODY)
    
    @classmethod
    def _map_kinematics(cls, source: np.ndarray, target: np.ndarray) -> None:
        """Map kinematic properties (position, velocity, orientation, angular velocity)."""
        target[:, :, 0:3] = source[:, :, cls.POSITION_INDICES]
        target[:, :, 3:6] = source[:, :, cls.VELOCITY_INDICES]
        target[:, :, 6:10] = source[:, :, cls.ORIENTATION_INDICES]
        target[:, :, 10:13] = source[:, :, cls.ANGULAR_VELOCITY_INDICES]
    
    @classmethod
    def _map_mass(cls, source: np.ndarray, target: np.ndarray) -> None:
        """Convert inverse mass to mass, handling infinite mass bodies."""
        inv_mass = source[:, :, cls.INV_MASS_INDEX]
        with np.errstate(divide='ignore'):
            mass = np.where(inv_mass > 0, 1.0 / inv_mass, 1e9)
        target[:, :, 13] = mass
    
    @classmethod
    def _map_shape(cls, source: np.ndarray, target: np.ndarray) -> None:
        """Map shape type and parameters."""
        target[:, :, 14] = source[:, :, cls.SHAPE_TYPE_INDEX]
        target[:, :, 15:18] = source[:, :, cls.SHAPE_PARAMS_INDICES]


class RendererInvoker:
    """Manages the external renderer subprocess for video generation."""
    
    def __init__(self, trajectory_path: str, video_output_path: str,
                 duration: float = 5.0, fps: int = 60):
        """Initialize the renderer invoker.
        
        Args:
            trajectory_path: Path to the trajectory .npy file
            video_output_path: Path for the output video file
            duration: Video duration in seconds (default: 5.0)
            fps: Frames per second for the video (default: 60)
        """
        self.trajectory_path = Path(trajectory_path)
        self.video_output_path = Path(video_output_path)
        self.duration = duration
        self.fps = fps
        
        # Locate renderer executable
        self.renderer_path = RendererLocator.find_renderer()
        
    def render(self, use_gpu_flag: bool = False, verbose: bool = True) -> bool:
        """Execute the renderer to create a video.
        
        Args:
            use_gpu_flag: If True, use GPU backend; otherwise use Oracle backend
            verbose: Whether to print renderer output
            
        Returns:
            True if rendering succeeded, False otherwise
        """
        # Create render command
        command = self._create_render_command(use_gpu_flag)
        
        # Execute rendering
        return self._execute_render_command(command, verbose)
    
    def _create_render_command(self, use_gpu_flag: bool) -> RenderCommand:
        """Create render command configuration."""
        mode = RenderMode.GPU if use_gpu_flag else RenderMode.ORACLE
        
        return RenderCommand(
            renderer_path=self.renderer_path,
            trajectory_path=self.trajectory_path,
            video_output_path=self.video_output_path,
            mode=mode,
            duration=self.duration,
            fps=self.fps
        )
    
    def _execute_render_command(self, command: RenderCommand, verbose: bool) -> bool:
        """Execute the render command and handle results."""
        cmd_args = command.to_args()
        
        if verbose:
            self._print_launch_message(cmd_args)
        
        try:
            result = self._run_subprocess(cmd_args)
            self._handle_subprocess_output(result, verbose)
            
            if result.returncode != 0:
                self._print_error(f"Renderer failed with exit code {result.returncode}")
                return False
            
            if verbose:
                print(f"Video saved to: {self.video_output_path}")
            
            return True
            
        except subprocess.SubprocessError as e:
            self._print_error(f"Failed to run renderer: {e}")
            return False
    
    def _run_subprocess(self, cmd_args: List[str]) -> subprocess.CompletedProcess:
        """Run the renderer subprocess."""
        return subprocess.run(
            cmd_args,
            capture_output=True,
            text=True,
            check=False
        )
    
    def _handle_subprocess_output(self, result: subprocess.CompletedProcess, 
                                  verbose: bool) -> None:
        """Handle subprocess stdout and stderr."""
        if verbose and result.stdout:
            print("Renderer output:")
            print(result.stdout)
        
        if result.stderr:
            self._print_error("Renderer errors:")
            self._print_error(result.stderr)
    
    def _print_launch_message(self, cmd_args: List[str]) -> None:
        """Print the launch command."""
        print(f"Launching renderer: {' '.join(cmd_args)}")
    
    def _print_error(self, message: str) -> None:
        """Print error message to stderr."""
        print(message, file=sys.stderr)
    
    @staticmethod
    def format_trajectory_for_renderer(trajectory: np.ndarray) -> np.ndarray:
        """Convert physics engine trajectory format to renderer format.
        
        Args:
            trajectory: Physics engine trajectory of shape (frames, bodies, 27)
            
        Returns:
            Formatted trajectory of shape (frames, bodies * 18)
        """
        return TrajectoryFormatter.format_for_renderer(trajectory)