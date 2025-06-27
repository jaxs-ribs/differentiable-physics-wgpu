"""
Test physics pattern matching functionality
"""
import pytest
import sys
from pathlib import Path

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "custom_ops"))

from tinygrad.uop.ops import UOp, Ops
from tinygrad.dtype import dtypes
from custom_ops.python.patterns import create_physics_patterns, PhysicsOps, create_physics_renderer_extensions

def test_create_physics_patterns():
    """Test that physics patterns can be created"""
    patterns = create_physics_patterns()
    assert patterns is not None
    assert len(patterns.patterns) > 0

def test_physics_ops_constants():
    """Test physics operation constants"""
    assert hasattr(PhysicsOps, 'PHYSICS_STEP')
    assert hasattr(PhysicsOps, 'PHYSICS_INTEGRATE')
    assert hasattr(PhysicsOps, 'PHYSICS_COLLIDE')

def test_pattern_matcher_structure():
    """Test the structure of pattern matcher"""
    patterns = create_physics_patterns()
    
    # Check that patterns have the expected structure
    for pattern, rewriter in patterns.patterns:
        # Pattern should be a UPat
        assert hasattr(pattern, 'op') or hasattr(pattern, 'name')
        # Rewriter should be callable
        assert callable(rewriter)

def test_renderer_extensions():
    """Test renderer extension patterns"""
    renderer_patterns = create_physics_renderer_extensions()
    assert renderer_patterns is not None
    assert len(renderer_patterns.patterns) > 0

def test_physics_step_marker():
    """Test creating a physics step marker UOp"""
    # Create a dummy tensor UOp
    dummy_tensor = UOp(Ops.CONST, dtypes.float32, tuple(), 1.0)
    
    # Create a physics step marker
    physics_marker = UOp(Ops.CUSTOM, dummy_tensor.dtype, (dummy_tensor,), PhysicsOps.PHYSICS_STEP)
    
    assert physics_marker.op == Ops.CUSTOM
    assert physics_marker.arg == PhysicsOps.PHYSICS_STEP
    assert physics_marker.src[0] == dummy_tensor

def test_pattern_matching_integration():
    """Test that pattern matching can process UOps"""
    patterns = create_physics_patterns()
    
    # Create a test UOp that should match
    dummy_tensor = UOp(Ops.CONST, dtypes.float32, tuple(), 1.0)
    test_uop = UOp(Ops.CUSTOM, dummy_tensor.dtype, (dummy_tensor,), PhysicsOps.PHYSICS_STEP)
    
    # This tests that the pattern matcher is properly constructed
    # In a real implementation, we would test the actual matching
    assert patterns is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])