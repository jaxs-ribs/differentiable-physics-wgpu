#!/usr/bin/env python3
"""Simulation runner for the physics engine.

This module provides the SimulationRunner class that handles:
- Loading initial state from file
- Translating execution mode strings to enums
- Initializing the physics engine
- Running the simulation loop
- Collecting performance metrics
"""
import time
import numpy as np
from pathlib import Path

# Add parent directory to path to import physics modules
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Add tinygrad to path
tinygrad_path = os.path.join(parent_dir, "external", "tinygrad")
if os.path.exists(tinygrad_path):
    sys.path.insert(0, tinygrad_path)

from physics.types import ExecutionMode
from physics.engine import TensorPhysicsEngine


class SimulationRunner:
    """Manages physics simulation execution with configurable backends."""
    
    def __init__(self, mode: str, steps: int, input_file_path: str, 
                 enable_profiling: bool = False, collect_trajectory: bool = True):
        """Initialize the simulation runner.
        
        Args:
            mode: Execution mode string ("pure", "c", or "wgpu")
            steps: Number of simulation steps to execute
            input_file_path: Path to .npy file containing initial state
            enable_profiling: Whether to collect detailed performance metrics
            collect_trajectory: Whether to collect full trajectory data for rendering
        """
        self.mode = mode
        self.steps = steps
        self.input_file_path = input_file_path
        self.enable_profiling = enable_profiling
        self.collect_trajectory = collect_trajectory
        
        # Load initial state
        self.initial_state = self._load_initial_state()
        
        # Translate mode string to enum
        self.execution_mode = self._get_execution_mode()
        
        # Initialize physics engine
        self.engine = TensorPhysicsEngine(
            bodies=self.initial_state,
            execution_mode=self.execution_mode
        )
    
    def _load_initial_state(self) -> np.ndarray:
        """Load initial bodies state from file.
        
        Returns:
            Numpy array of shape (num_bodies, num_properties)
        """
        try:
            state = np.load(self.input_file_path)
            print(f"Loaded initial state from {self.input_file_path}")
            print(f"  Shape: {state.shape}")
            return state
        except Exception as e:
            raise RuntimeError(f"Failed to load initial state from {self.input_file_path}: {e}")
    
    def _get_execution_mode(self) -> ExecutionMode:
        """Convert mode string to ExecutionMode enum.
        
        Returns:
            ExecutionMode enum value
        """
        mode_map = {
            "pure": ExecutionMode.PURE,
            "c": ExecutionMode.C,
            "wgpu": ExecutionMode.WGPU
        }
        
        if self.mode not in mode_map:
            raise ValueError(f"Unknown execution mode: {self.mode}")
        
        return mode_map[self.mode]
    
    def run(self) -> tuple[np.ndarray, dict]:
        """Execute the simulation.
        
        Returns:
            Tuple of (trajectory_or_final_state, metrics_dict)
            If collect_trajectory is True, returns full trajectory (steps+1, num_bodies, props)
            Otherwise returns only final state (num_bodies, props)
        """
        metrics = {}
        trajectory = [] if self.collect_trajectory else None
        
        # Start timing
        start_time = time.time()
        
        # Collect initial state if needed
        if self.collect_trajectory:
            trajectory.append(self.engine.get_state().copy())
        
        if self.enable_profiling:
            # Detailed profiling - time each step
            step_times = []
            for i in range(self.steps):
                step_start = time.time()
                self.engine.step()
                step_end = time.time()
                step_times.append(step_end - step_start)
                
                if self.collect_trajectory:
                    trajectory.append(self.engine.get_state().copy())
            
            metrics['profile_data'] = {
                'avg_step_time': np.mean(step_times),
                'min_step_time': np.min(step_times),
                'max_step_time': np.max(step_times),
                'std_step_time': np.std(step_times)
            }
        else:
            # Normal execution - just run all steps
            for _ in range(self.steps):
                self.engine.step()
                if self.collect_trajectory:
                    trajectory.append(self.engine.get_state().copy())
        
        # End timing
        end_time = time.time()
        total_time = end_time - start_time
        
        # Prepare return data
        if self.collect_trajectory:
            # Stack trajectory frames into single array
            result_data = np.stack(trajectory)
            print(f"Collected trajectory shape: {result_data.shape}")
        else:
            # Just return final state
            result_data = self.engine.get_state()
        
        # Calculate metrics
        metrics['total_time'] = total_time
        metrics['steps_per_second'] = self.steps / total_time
        metrics['execution_mode'] = self.mode
        metrics['num_steps'] = self.steps
        metrics['trajectory_collected'] = self.collect_trajectory
        
        return result_data, metrics