"""Test JIT compilation functionality.

WHAT: Validates that TinyGrad's JIT compilation works correctly with 
      the physics engine and produces deterministic results.

WHY: JIT compilation is critical for performance:
     - Compiles Python/TinyGrad operations to optimized GPU/CPU code
     - Can introduce subtle bugs if tensor operations aren't JIT-safe
     - Must produce identical results to non-JIT execution
     Without working JIT, the engine would be too slow for real use.

HOW: - Forces JIT mode via environment variable
     - Runs the same simulation twice (triggering compilation on first run)
     - Compares results to ensure they're identical
     - Uses numpy.allclose to handle floating-point precision
"""
import os
import numpy as np
import pytest
from physics.engine import TensorPhysicsEngine
from physics.types import BodySchema, ShapeType

def test_jit_compilation():
    """Verify JIT compilation works correctly."""
    # Force JIT mode
    os.environ['JIT'] = '1'
    
    # Simple 2-body scene
    bodies = np.zeros((2, BodySchema.NUM_PROPERTIES), dtype=np.float32)
    
    # Two spheres
    bodies[0, BodySchema.POS_X] = -2.0
    bodies[0, BodySchema.QUAT_W] = 1.0
    bodies[0, BodySchema.SHAPE_TYPE] = ShapeType.SPHERE
    bodies[0, BodySchema.SHAPE_PARAM_1] = 0.5
    bodies[0, BodySchema.INV_MASS] = 1.0
    
    bodies[1, BodySchema.POS_X] = 2.0
    bodies[1, BodySchema.QUAT_W] = 1.0
    bodies[1, BodySchema.SHAPE_TYPE] = ShapeType.SPHERE
    bodies[1, BodySchema.SHAPE_PARAM_1] = 0.5
    bodies[1, BodySchema.INV_MASS] = 1.0
    
    # Create engine and run
    engine = TensorPhysicsEngine(bodies, dt=0.016)
    
    # First run triggers compilation
    engine.run_simulation(10)
    state1 = engine.get_state().copy()
    
    # Reset and run again - should use compiled version
    engine = TensorPhysicsEngine(bodies, dt=0.016)
    engine.run_simulation(10)
    state2 = engine.get_state()
    
    # Results should be identical
    np.testing.assert_allclose(state1, state2, rtol=1e-5)

if __name__ == "__main__":
    test_jit_compilation()
    print("JIT compilation test passed!")