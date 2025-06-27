"""
Physics Extension for TinyGrad Devices
Extends existing TinyGrad devices with physics operation support
"""
from pathlib import Path
import sys

# Add parent directory to path to import tinygrad
sys.path.append(str(Path(__file__).parent.parent))

from tinygrad.device import Device
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import PatternMatcher
from physics_patterns import create_physics_patterns, create_physics_renderer_extensions, get_physics_lib

class PhysicsEnabledRenderer(Renderer):
    """
    Wrapper renderer that adds physics patterns to an existing renderer
    """
    def __init__(self, base_renderer: Renderer):
        super().__init__()
        self.base_renderer = base_renderer
        
        # Copy attributes from base renderer
        self.device = base_renderer.device
        self.suffix = base_renderer.suffix
        self.supports_float4 = base_renderer.supports_float4
        self.has_local = base_renderer.has_local
        self.has_shared = base_renderer.has_shared
        self.global_max = base_renderer.global_max
        self.local_max = base_renderer.local_max
        self.shared_max = base_renderer.shared_max
        self.tensor_cores = base_renderer.tensor_cores
        self.code_for_op = base_renderer.code_for_op.copy()
        
        # Add physics patterns
        self.physics_patterns = create_physics_patterns()
        self.physics_renderer_patterns = create_physics_renderer_extensions()
        
        # Combine pattern matchers
        if base_renderer.pre_matcher is not None:
            # Combine with existing pre_matcher
            combined_patterns = list(base_renderer.pre_matcher.patterns)
            combined_patterns.extend(self.physics_patterns.patterns)
            self.pre_matcher = PatternMatcher(combined_patterns)
        else:
            self.pre_matcher = self.physics_patterns
            
        if base_renderer.extra_matcher is not None:
            # Combine with existing extra_matcher
            combined_patterns = list(base_renderer.extra_matcher.patterns)
            combined_patterns.extend(self.physics_renderer_patterns.patterns)
            self.extra_matcher = PatternMatcher(combined_patterns)
        else:
            self.extra_matcher = self.physics_renderer_patterns
    
    def render(self, uops):
        """Render UOps, delegating to base renderer"""
        # The pattern matching happens automatically through pre_matcher and extra_matcher
        return self.base_renderer.render(uops)

def enable_physics_on_device(device_name: str = "CPU"):
    """
    Enable physics operations on a specific TinyGrad device
    
    Args:
        device_name: Name of the device to enable physics on (default: "CPU")
    """
    # Get the device
    device = Device[device_name]
    
    # Wrap the renderer with physics support
    if not isinstance(device.renderer, PhysicsEnabledRenderer):
        device.renderer = PhysicsEnabledRenderer(device.renderer)
        
        # Load physics library to ensure it's available
        get_physics_lib()
        
        print(f"Physics operations enabled on device: {device_name}")
    else:
        print(f"Physics operations already enabled on device: {device_name}")

def disable_physics_on_device(device_name: str = "CPU"):
    """
    Disable physics operations on a specific TinyGrad device
    
    Args:
        device_name: Name of the device to disable physics on
    """
    device = Device[device_name]
    
    if isinstance(device.renderer, PhysicsEnabledRenderer):
        # Restore original renderer
        device.renderer = device.renderer.base_renderer
        print(f"Physics operations disabled on device: {device_name}")
    else:
        print(f"Physics operations not enabled on device: {device_name}")

# Context manager for temporary physics enablement
class physics_enabled:
    """Context manager to temporarily enable physics operations"""
    def __init__(self, device_name: str = "CPU"):
        self.device_name = device_name
        self.was_enabled = False
    
    def __enter__(self):
        device = Device[self.device_name]
        self.was_enabled = isinstance(device.renderer, PhysicsEnabledRenderer)
        if not self.was_enabled:
            enable_physics_on_device(self.device_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.was_enabled:
            disable_physics_on_device(self.device_name)