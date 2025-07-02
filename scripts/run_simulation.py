#!/usr/bin/env python3
"""Simulation runner for the physics engine.

Follows Single Responsibility Principle - handles only simulation execution.
Separates concerns: state loading, metrics collection, and trajectory management.
"""
import time
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass

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


@dataclass
class SimulationMetrics:
    """Encapsulates simulation performance metrics."""
    total_time: float
    steps_per_second: float
    execution_mode: str
    num_steps: int
    trajectory_collected: bool
    profile_data: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, any]:
        """Convert metrics to dictionary format."""
        result = {
            'total_time': self.total_time,
            'steps_per_second': self.steps_per_second,
            'execution_mode': self.execution_mode,
            'num_steps': self.num_steps,
            'trajectory_collected': self.trajectory_collected
        }
        if self.profile_data:
            result['profile_data'] = self.profile_data
        return result


class StateLoader:
    """Handles loading initial state from file."""
    
    @staticmethod
    def load_initial_state(file_path: str) -> np.ndarray:
        """Load initial bodies state from file.
        
        Args:
            file_path: Path to the .npy file
            
        Returns:
            Numpy array of shape (num_bodies, num_properties)
            
        Raises:
            RuntimeError: If loading fails
        """
        try:
            state = np.load(file_path)
            print(f"Loaded initial state from {file_path}")
            print(f"  Shape: {state.shape}")
            return state
        except Exception as e:
            raise RuntimeError(f"Failed to load initial state from {file_path}: {e}")


class TrajectoryCollector:
    """Manages trajectory collection during simulation."""
    
    def __init__(self, collect: bool = True):
        """Initialize the trajectory collector.
        
        Args:
            collect: Whether to collect trajectory data
        """
        self.collect = collect
        self.trajectory: List[np.ndarray] = [] if collect else None
    
    def add_frame(self, state: np.ndarray) -> None:
        """Add a frame to the trajectory.
        
        Args:
            state: Current simulation state
        """
        if self.collect:
            self.trajectory.append(state.copy())
    
    def get_result(self, final_state: np.ndarray) -> np.ndarray:
        """Get the collected trajectory or final state.
        
        Args:
            final_state: The final simulation state
            
        Returns:
            Either the full trajectory or just the final state
        """
        if self.collect:
            result = np.stack(self.trajectory)
            print(f"Collected trajectory shape: {result.shape}")
            return result
        else:
            return final_state


class SimulationProfiler:
    """Handles performance profiling of simulation steps."""
    
    def __init__(self, enabled: bool = False):
        """Initialize the profiler.
        
        Args:
            enabled: Whether profiling is enabled
        """
        self.enabled = enabled
        self.step_times: List[float] = [] if enabled else None
    
    def record_step_time(self, duration: float) -> None:
        """Record the duration of a simulation step.
        
        Args:
            duration: Time taken for the step in seconds
        """
        if self.enabled:
            self.step_times.append(duration)
    
    def get_profile_data(self) -> Optional[Dict[str, float]]:
        """Get profiling statistics.
        
        Returns:
            Dictionary with profiling metrics or None if disabled
        """
        if not self.enabled or not self.step_times:
            return None
        
        return {
            'avg_step_time': np.mean(self.step_times),
            'min_step_time': np.min(self.step_times),
            'max_step_time': np.max(self.step_times),
            'std_step_time': np.std(self.step_times)
        }


class ExecutionModeMapper:
    """Maps string mode names to ExecutionMode enums."""
    
    MODE_MAP = {
        "pure": ExecutionMode.PURE,
        "c": ExecutionMode.C,
        "wgpu": ExecutionMode.WGPU
    }
    
    @classmethod
    def get_execution_mode(cls, mode_string: str) -> ExecutionMode:
        """Convert mode string to ExecutionMode enum.
        
        Args:
            mode_string: Mode name as string
            
        Returns:
            ExecutionMode enum value
            
        Raises:
            ValueError: If mode is unknown
        """
        if mode_string not in cls.MODE_MAP:
            raise ValueError(f"Unknown execution mode: {mode_string}")
        
        return cls.MODE_MAP[mode_string]


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
        
        # Initialize components
        self.state_loader = StateLoader()
        self.trajectory_collector = TrajectoryCollector(collect_trajectory)
        self.profiler = SimulationProfiler(enable_profiling)
        
        # Load initial state
        self.initial_state = self.state_loader.load_initial_state(input_file_path)
        
        # Get execution mode
        self.execution_mode = ExecutionModeMapper.get_execution_mode(mode)
        
        # Initialize physics engine
        self.engine = TensorPhysicsEngine(
            bodies=self.initial_state,
            execution_mode=self.execution_mode
        )
    
    def run(self) -> Tuple[np.ndarray, Dict[str, any]]:
        """Execute the simulation.
        
        Returns:
            Tuple of (trajectory_or_final_state, metrics_dict)
        """
        # Record initial state
        self.trajectory_collector.add_frame(self.engine.get_state())
        
        # Run simulation
        total_time = self._run_simulation_loop()
        
        # Get results
        final_state = self.engine.get_state()
        result_data = self.trajectory_collector.get_result(final_state)
        
        # Build metrics
        metrics = self._build_metrics(total_time)
        
        return result_data, metrics.to_dict()
    
    def _run_simulation_loop(self) -> float:
        """Run the main simulation loop.
        
        Returns:
            Total execution time in seconds
        """
        start_time = time.time()
        
        if self.profiler.enabled:
            self._run_profiled_simulation()
        else:
            self._run_normal_simulation()
        
        end_time = time.time()
        return end_time - start_time
    
    def _run_profiled_simulation(self) -> None:
        """Run simulation with step-by-step profiling."""
        for _ in range(self.steps):
            step_start = time.time()
            self.engine.step()
            step_end = time.time()
            
            self.profiler.record_step_time(step_end - step_start)
            self.trajectory_collector.add_frame(self.engine.get_state())
    
    def _run_normal_simulation(self) -> None:
        """Run simulation without profiling."""
        for _ in range(self.steps):
            self.engine.step()
            self.trajectory_collector.add_frame(self.engine.get_state())
    
    def _build_metrics(self, total_time: float) -> SimulationMetrics:
        """Build simulation metrics.
        
        Args:
            total_time: Total simulation time in seconds
            
        Returns:
            SimulationMetrics object
        """
        return SimulationMetrics(
            total_time=total_time,
            steps_per_second=self.steps / total_time,
            execution_mode=self.mode,
            num_steps=self.steps,
            trajectory_collected=self.trajectory_collector.collect,
            profile_data=self.profiler.get_profile_data()
        )