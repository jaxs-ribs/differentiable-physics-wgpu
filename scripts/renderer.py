#!/usr/bin/env python3
"""Renderer invocation module for the physics engine.

This module provides the RendererInvoker class that manages the external
Rust-based renderer subprocess to create videos from simulation trajectories.
"""
import subprocess
import sys
from pathlib import Path
from typing import Optional, List
import numpy as np


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
        self.renderer_path = self._find_renderer()
        
    def _find_renderer(self) -> Path:
        """Locate the renderer executable.
        
        Returns:
            Path to the renderer executable
            
        Raises:
            RuntimeError: If renderer not found
        """
        # Expected path relative to physics_core
        physics_core_dir = Path(__file__).parent.parent
        renderer_path = physics_core_dir / "renderer" / "target" / "release" / "renderer"
        
        if not renderer_path.exists():
            raise RuntimeError(
                f"Renderer executable not found at {renderer_path}\n"
                "Please build the renderer first: cd renderer && cargo build --release"
            )
        
        return renderer_path
    
    def render(self, use_gpu_flag: bool = False, verbose: bool = True) -> bool:
        """Execute the renderer to create a video.
        
        Args:
            use_gpu_flag: If True, use --gpu flag; otherwise use --oracle flag
            verbose: Whether to print renderer output
            
        Returns:
            True if rendering succeeded, False otherwise
        """
        # Build command
        cmd = [str(self.renderer_path)]
        
        # Add simulation file with appropriate flag
        if use_gpu_flag:
            cmd.extend(["--gpu", str(self.trajectory_path)])
        else:
            cmd.extend(["--oracle", str(self.trajectory_path)])
        
        # Add video recording parameters
        cmd.extend([
            "--record", str(self.video_output_path),
            "--duration", str(self.duration),
            "--fps", str(self.fps)
        ])
        
        if verbose:
            print(f"Launching renderer: {' '.join(cmd)}")
        
        try:
            # Run renderer subprocess
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            # Print output if verbose
            if verbose and result.stdout:
                print("Renderer output:")
                print(result.stdout)
            
            # Print errors if any
            if result.stderr:
                print("Renderer errors:", file=sys.stderr)
                print(result.stderr, file=sys.stderr)
            
            # Check return code
            if result.returncode != 0:
                print(f"Renderer failed with exit code {result.returncode}", file=sys.stderr)
                return False
            
            if verbose:
                print(f"Video saved to: {self.video_output_path}")
            
            return True
            
        except subprocess.SubprocessError as e:
            print(f"Failed to run renderer: {e}", file=sys.stderr)
            return False
    
    @staticmethod
    def format_trajectory_for_renderer(trajectory: np.ndarray) -> np.ndarray:
        """Convert physics engine trajectory format to renderer format.
        
        The renderer expects 18 properties per body:
        - Position (3), Velocity (3), Orientation (4), Angular velocity (3)
        - Mass (1), Shape type (1), Shape parameters (3)
        
        Args:
            trajectory: Physics engine trajectory of shape (frames, bodies, 27)
            
        Returns:
            Formatted trajectory of shape (frames, bodies * 18)
        """
        frames, num_bodies, props = trajectory.shape
        
        # Map from physics engine indices to renderer indices
        # Physics engine has 27 properties, renderer needs 18
        renderer_data = np.zeros((frames, num_bodies, 18), dtype=np.float32)
        
        # Copy relevant properties
        renderer_data[:, :, 0:3] = trajectory[:, :, 0:3]    # Position
        renderer_data[:, :, 3:6] = trajectory[:, :, 3:6]    # Velocity
        renderer_data[:, :, 6:10] = trajectory[:, :, 6:10]  # Orientation (quaternion)
        renderer_data[:, :, 10:13] = trajectory[:, :, 10:13] # Angular velocity
        
        # Mass = 1 / inv_mass (handle infinite mass)
        inv_mass = trajectory[:, :, 13]
        with np.errstate(divide='ignore'):
            mass = np.where(inv_mass > 0, 1.0 / inv_mass, 1e9)
        renderer_data[:, :, 13] = mass
        
        renderer_data[:, :, 14] = trajectory[:, :, 23]      # Shape type
        renderer_data[:, :, 15:18] = trajectory[:, :, 24:27] # Shape parameters
        
        # Reshape to (frames, bodies * 18) as expected by renderer
        return renderer_data.reshape(frames, num_bodies * 18)