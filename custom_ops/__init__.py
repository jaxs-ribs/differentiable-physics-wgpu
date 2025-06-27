"""
TinyGrad Custom Physics Operations

This module provides high-performance physics operations for TinyGrad
through custom C implementations and pattern matching integration.
"""

from .python.extension import enable_physics_on_device, disable_physics_on_device, physics_enabled
from .python.tensor_ops import PhysicsTensor, create_physics_world

__all__ = [
    'enable_physics_on_device',
    'disable_physics_on_device', 
    'physics_enabled',
    'PhysicsTensor',
    'create_physics_world'
]