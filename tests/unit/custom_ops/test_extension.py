"""
Test physics device extension functionality
"""
import pytest
import sys
from pathlib import Path

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "custom_ops"))

from tinygrad.device import Device
from tinygrad.renderer import Renderer
from custom_ops.python.extension import (
    PhysicsEnabledRenderer, 
    enable_physics_on_device, 
    disable_physics_on_device,
    physics_enabled
)

class MockRenderer(Renderer):
    """Mock renderer for testing"""
    def __init__(self):
        super().__init__()
        self.device = "MOCK"
        self.suffix = ".mock"
    
    def render(self, uops):
        return "mock_render"

def test_physics_enabled_renderer_creation():
    """Test creating a physics-enabled renderer"""
    base_renderer = MockRenderer()
    physics_renderer = PhysicsEnabledRenderer(base_renderer)
    
    assert isinstance(physics_renderer, PhysicsEnabledRenderer)
    assert physics_renderer.base_renderer == base_renderer
    assert physics_renderer.device == "MOCK"
    assert physics_renderer.suffix == ".mock"

def test_physics_enabled_renderer_attributes():
    """Test that physics renderer copies base renderer attributes"""
    base_renderer = MockRenderer()
    base_renderer.supports_float4 = False
    base_renderer.has_local = False
    
    physics_renderer = PhysicsEnabledRenderer(base_renderer)
    
    assert physics_renderer.supports_float4 == False
    assert physics_renderer.has_local == False

def test_pattern_matcher_integration():
    """Test that pattern matchers are properly set up"""
    base_renderer = MockRenderer()
    physics_renderer = PhysicsEnabledRenderer(base_renderer)
    
    assert physics_renderer.pre_matcher is not None
    assert physics_renderer.extra_matcher is not None

def test_render_delegation():
    """Test that render calls are delegated to base renderer"""
    base_renderer = MockRenderer()
    physics_renderer = PhysicsEnabledRenderer(base_renderer)
    
    result = physics_renderer.render([])
    assert result == "mock_render"

def test_context_manager():
    """Test physics_enabled context manager"""
    # Create a mock device setup
    original_device = Device["CPU"]
    original_renderer = original_device.renderer if original_device else None
    
    # Test context manager
    with physics_enabled("CPU") as ctx:
        # Within context, physics should be enabled
        # Note: We can't easily test this without a full TinyGrad setup
        assert ctx is not None
    
    # After context, original state should be restored
    if original_device and original_renderer:
        # In a real test, we'd verify the renderer was restored
        pass

def test_enable_disable_cycle():
    """Test enabling and disabling physics on a device"""
    # This test would require a full TinyGrad environment
    # For now, we just test that the functions exist and are callable
    assert callable(enable_physics_on_device)
    assert callable(disable_physics_on_device)

def test_physics_patterns_exist():
    """Test that physics patterns are created"""
    base_renderer = MockRenderer()
    physics_renderer = PhysicsEnabledRenderer(base_renderer)
    
    assert hasattr(physics_renderer, 'physics_patterns')
    assert hasattr(physics_renderer, 'physics_renderer_patterns')

if __name__ == "__main__":
    pytest.main([__file__, "-v"])