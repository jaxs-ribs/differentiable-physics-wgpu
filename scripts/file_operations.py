"""File operations module for the physics simulation.

Follows Single Responsibility Principle - handles all file I/O operations.
"""
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional
import tempfile
import os


def ensure_directory_exists(directory: Path) -> None:
    """Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Path to the directory
    """
    directory.mkdir(parents=True, exist_ok=True)


def generate_timestamped_filename(prefix: str = "simulation", 
                                  extension: str = "mp4") -> str:
    """Generate a filename with current timestamp.
    
    Args:
        prefix: Filename prefix
        extension: File extension
        
    Returns:
        Timestamped filename string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{prefix}_{timestamp}.{extension}"


def save_numpy_array(array: np.ndarray, filepath: Path) -> None:
    """Save a numpy array to file.
    
    Args:
        array: Array to save
        filepath: Destination path
    """
    ensure_directory_exists(filepath.parent)
    np.save(filepath, array)


def load_numpy_array(filepath: Path) -> np.ndarray:
    """Load a numpy array from file.
    
    Args:
        filepath: Path to the .npy file
        
    Returns:
        Loaded numpy array
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is corrupted
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    try:
        return np.load(filepath)
    except Exception as e:
        raise ValueError(f"Failed to load array from {filepath}: {e}")


def create_temporary_file(suffix: str = '.npy') -> tempfile.NamedTemporaryFile:
    """Create a temporary file.
    
    Args:
        suffix: File suffix
        
    Returns:
        NamedTemporaryFile object
    """
    return tempfile.NamedTemporaryFile(suffix=suffix, delete=False)


def safe_delete_file(filepath: str) -> None:
    """Safely delete a file, ignoring errors.
    
    Args:
        filepath: Path to file to delete
    """
    try:
        os.unlink(filepath)
    except Exception:
        pass  # Ignore deletion errors


def ensure_initial_state_exists(filepath: Path) -> None:
    """Ensure the initial state file exists, creating default if needed.
    
    Args:
        filepath: Path to initial state file
    """
    if not filepath.exists() and str(filepath) == "artifacts/initial_state.npy":
        print("Default initial state not found. Creating it...")
        
        # Import here to avoid circular dependency
        from .create_default_scene import create_default_scene
        
        initial_state = create_default_scene()
        save_numpy_array(initial_state, filepath)
        
        print(f"Created default initial state: {filepath}")
        print(f"  Shape: {initial_state.shape}")


def extract_final_state(simulation_data: np.ndarray, 
                        is_trajectory: bool) -> np.ndarray:
    """Extract the final state from simulation data.
    
    Args:
        simulation_data: Either trajectory (frames, bodies, props) or final state
        is_trajectory: Whether data contains full trajectory
        
    Returns:
        Final state array (bodies, props)
    """
    if is_trajectory:
        return simulation_data[-1]  # Last frame
    else:
        return simulation_data