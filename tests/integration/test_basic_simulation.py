"""Test basic physics simulation functionality.

WHAT: Validates core physics simulation features including gravity, 
      single-step simulation, and multi-step (JIT) simulation.

WHY: These are the fundamental operations of the physics engine:
     - Gravity must work correctly for realistic motion
     - Single-step mode is used for debugging and visualization
     - Multi-step JIT mode is used for performance
     If these basics fail, nothing else will work correctly.

HOW: Creates simple test scenes with ground and falling objects,
     runs the simulation, and verifies that:
     - Objects fall under gravity
     - Positions update correctly
     - Both single and multi-step modes produce valid results
"""
import numpy as np
import pytest
from physics.engine import TensorPhysicsEngine
from physics.types import BodySchema, ShapeType

def test_falling_sphere():
    """Test that a sphere falls under gravity."""
    # Create scene with ground and falling sphere
    bodies = np.zeros((2, BodySchema.NUM_PROPERTIES), dtype=np.float32)
    
    # Static ground box
    bodies[0, BodySchema.POS_Y] = -5.0
    bodies[0, BodySchema.QUAT_W] = 1.0
    bodies[0, BodySchema.SHAPE_TYPE] = ShapeType.BOX
    bodies[0, BodySchema.SHAPE_PARAM_1:BodySchema.SHAPE_PARAM_3+1] = [10.0, 0.5, 10.0]
    bodies[0, BodySchema.INV_MASS] = 0.0  # Static
    
    # Falling sphere
    bodies[1, BodySchema.POS_Y] = 5.0
    bodies[1, BodySchema.QUAT_W] = 1.0
    bodies[1, BodySchema.SHAPE_TYPE] = ShapeType.SPHERE
    bodies[1, BodySchema.SHAPE_PARAM_1] = 1.0
    bodies[1, BodySchema.INV_MASS] = 1.0
    bodies[1, BodySchema.INV_INERTIA_XX] = 2.5
    bodies[1, BodySchema.INV_INERTIA_YY] = 2.5
    bodies[1, BodySchema.INV_INERTIA_ZZ] = 2.5
    
    # Test single step
    engine = TensorPhysicsEngine(bodies.copy(), dt=0.016)
    initial_y = engine.get_state()[1, BodySchema.POS_Y]
    engine.step()
    final_y = engine.get_state()[1, BodySchema.POS_Y]
    
    assert final_y < initial_y, "Sphere should fall under gravity"

def test_multi_step_simulation():
    """Test N-step simulation mode."""
    bodies = np.zeros((2, BodySchema.NUM_PROPERTIES), dtype=np.float32)
    
    # Ground
    bodies[0, BodySchema.POS_Y] = -5.0
    bodies[0, BodySchema.QUAT_W] = 1.0
    bodies[0, BodySchema.SHAPE_TYPE] = ShapeType.BOX
    bodies[0, BodySchema.SHAPE_PARAM_1:BodySchema.SHAPE_PARAM_3+1] = [10.0, 0.5, 10.0]
    bodies[0, BodySchema.INV_MASS] = 0.0
    
    # Sphere
    bodies[1, BodySchema.POS_Y] = 5.0
    bodies[1, BodySchema.QUAT_W] = 1.0
    bodies[1, BodySchema.SHAPE_TYPE] = ShapeType.SPHERE
    bodies[1, BodySchema.SHAPE_PARAM_1] = 1.0
    bodies[1, BodySchema.INV_MASS] = 1.0
    
    engine = TensorPhysicsEngine(bodies, dt=0.016)
    initial_y = engine.get_state()[1, BodySchema.POS_Y]
    
    # Run 10 steps
    engine.run_simulation(10)
    final_y = engine.get_state()[1, BodySchema.POS_Y]
    
    # Should fall more in 10 steps than 1
    expected_fall = 0.5 * 9.81 * (10 * 0.016)**2
    actual_fall = initial_y - final_y
    
    assert actual_fall > 0, "Sphere should fall"
    assert actual_fall < expected_fall * 2, "Fall should be reasonable"

if __name__ == "__main__":
    test_falling_sphere()
    test_multi_step_simulation()
    print("Basic simulation tests passed!")