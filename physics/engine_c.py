#!/usr/bin/env python3
"""C-accelerated physics engine wrapper.

This module provides the CEngine class, which conforms to the PhysicsEngine
interface and uses TinyGrad's custom C operations for its implementation.
"""
import numpy as np
from pathlib import Path
import sys

# Add path to import tinygrad and custom_ops
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "custom_ops"))

from tinygrad import Tensor
from custom_ops.python.extension import enable_physics_on_device, physics_enabled
from custom_ops.python.patterns import physics_step
from physics.engine import PhysicsEngine

class CEngine(PhysicsEngine):
    """Wrapper for the C-accelerated physics engine."""
    def __init__(self, bodies: np.ndarray, dt: float = 0.01):
        # Check if the C library is compiled
        lib_path = Path(__file__).parent.parent / "custom_ops" / "build" / ("libphysics.dylib" if sys.platform == "darwin" else "libphysics.so")
        if not lib_path.exists():
            raise RuntimeError("C physics library not found. Please compile it first by running: cd custom_ops/src && make")

        # Enable the custom physics operations on the CPU device
        enable_physics_on_device("CPU")
        
        self.bodies = Tensor(bodies.astype(np.float32))
        self.dt = dt

    def step(self) -> None:
        """Perform one physics step using the custom C operation."""
        # The pattern matcher in TinyGrad will replace this Python call
        # with a call to the C function `physics_step`.
        with physics_enabled("CPU"):
            self.bodies = physics_step(self.bodies, self.dt)

    def get_state(self) -> np.ndarray:
        """Get the current state of all bodies."""
        return self.bodies.numpy()
