"""Error handling module for the physics simulation pipeline.

Provides centralized error handling and logging functionality.
Following Single Responsibility Principle - handles only error management.
"""
import sys
import logging
from typing import Optional, Type
from pathlib import Path
from contextlib import contextmanager


class SimulationError(Exception):
    """Base exception for simulation-related errors."""
    pass


class ConfigurationError(SimulationError):
    """Raised when configuration is invalid."""
    pass


class FileOperationError(SimulationError):
    """Raised when file operations fail."""
    pass


class RenderingError(SimulationError):
    """Raised when rendering fails."""
    pass


class PhysicsEngineError(SimulationError):
    """Raised when physics engine encounters an error."""
    pass


class ErrorHandler:
    """Centralized error handling for the simulation pipeline."""
    
    def __init__(self, log_file: Optional[Path] = None, verbose: bool = True):
        """Initialize the error handler.
        
        Args:
            log_file: Optional path to log file
            verbose: Whether to print errors to console
        """
        self.verbose = verbose
        self._setup_logging(log_file)
    
    def _setup_logging(self, log_file: Optional[Path]) -> None:
        """Setup logging configuration."""
        handlers = []
        
        if log_file:
            handlers.append(logging.FileHandler(log_file))
        
        if self.verbose:
            handlers.append(logging.StreamHandler(sys.stderr))
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=handlers
        )
        
        self.logger = logging.getLogger('physics_simulation')
    
    def handle_error(self, error: Exception, critical: bool = True) -> None:
        """Handle an error with appropriate logging.
        
        Args:
            error: The exception that occurred
            critical: Whether this error should terminate execution
        """
        error_type = type(error).__name__
        error_message = str(error)
        
        if isinstance(error, SimulationError):
            self.logger.error(f"{error_type}: {error_message}")
        else:
            self.logger.exception(f"Unexpected error: {error_message}")
        
        if critical:
            sys.exit(1)
    
    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        self.logger.warning(message)
        if self.verbose:
            print(f"Warning: {message}", file=sys.stderr)
    
    def log_info(self, message: str) -> None:
        """Log an info message."""
        self.logger.info(message)
    
    @contextmanager
    def error_context(self, operation: str, error_type: Type[SimulationError] = SimulationError):
        """Context manager for handling errors in a specific operation.
        
        Args:
            operation: Description of the operation being performed
            error_type: Type of error to raise if exception occurs
        """
        try:
            yield
        except Exception as e:
            self.logger.error(f"Error during {operation}: {e}")
            raise error_type(f"Failed to {operation}: {e}") from e


def validate_file_exists(file_path: Path, file_description: str) -> None:
    """Validate that a required file exists.
    
    Args:
        file_path: Path to the file
        file_description: Human-readable description of the file
        
    Raises:
        FileOperationError: If file doesn't exist
    """
    if not file_path.exists():
        raise FileOperationError(f"{file_description} not found: {file_path}")


def validate_positive_number(value: float, parameter_name: str) -> None:
    """Validate that a number is positive.
    
    Args:
        value: The value to validate
        parameter_name: Name of the parameter for error messages
        
    Raises:
        ConfigurationError: If value is not positive
    """
    if value <= 0:
        raise ConfigurationError(f"{parameter_name} must be positive, got {value}")


def validate_execution_mode(mode: str, valid_modes: list) -> None:
    """Validate execution mode is supported.
    
    Args:
        mode: The execution mode string
        valid_modes: List of valid mode strings
        
    Raises:
        ConfigurationError: If mode is not valid
    """
    if mode not in valid_modes:
        raise ConfigurationError(
            f"Invalid execution mode '{mode}'. Valid modes: {', '.join(valid_modes)}"
        )