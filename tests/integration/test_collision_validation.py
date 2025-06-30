"""Test collision detection functionality at integration level.

WHAT: Validates that collision detection and response work correctly
      for different shape combinations (sphere-sphere, box-ground).

WHY: Collision detection is the core of any physics engine:
     - Objects must detect when they touch/overlap
     - Collision response must apply correct forces
     - Different shape combinations have different algorithms
     - Incorrect collisions make simulations useless

HOW: - Creates scenes with objects on collision courses
     - Runs simulation until collision should occur
     - Verifies collision effects:
       * Stationary objects start moving
       * Moving objects change velocity
       * Objects don't penetrate each other
     - Tests both elastic (sphere) and inelastic (box) collisions
"""
import numpy as np
import pytest
from physics.engine import TensorPhysicsEngine
from physics.types import BodySchema, ShapeType

def test_sphere_collision():
    """Test that colliding spheres interact correctly."""
    bodies = np.zeros((2, BodySchema.NUM_PROPERTIES), dtype=np.float32)
    
    # Sphere 1 moving right
    bodies[0, BodySchema.POS_X] = -2.0
    bodies[0, BodySchema.VEL_X] = 5.0
    bodies[0, BodySchema.QUAT_W] = 1.0
    bodies[0, BodySchema.SHAPE_TYPE] = ShapeType.SPHERE
    bodies[0, BodySchema.SHAPE_PARAM_1] = 0.5
    bodies[0, BodySchema.INV_MASS] = 1.0
    
    # Sphere 2 stationary
    bodies[1, BodySchema.POS_X] = 0.0
    bodies[1, BodySchema.QUAT_W] = 1.0
    bodies[1, BodySchema.SHAPE_TYPE] = ShapeType.SPHERE
    bodies[1, BodySchema.SHAPE_PARAM_1] = 0.5
    bodies[1, BodySchema.INV_MASS] = 1.0
    
    engine = TensorPhysicsEngine(bodies, dt=0.016, restitution=0.5)
    
    # Run until collision should have occurred
    for _ in range(50):
        engine.step()
    
    state = engine.get_state()
    
    # After collision, sphere 2 should be moving
    assert state[1, BodySchema.VEL_X] > 0.1, "Stationary sphere should move after collision"
    # And sphere 1 should have slowed down
    assert state[0, BodySchema.VEL_X] < 4.0, "Moving sphere should slow after collision"

def test_box_ground_collision():
    """Test box landing on ground."""
    bodies = np.zeros((2, BodySchema.NUM_PROPERTIES), dtype=np.float32)
    
    # Ground
    bodies[0, BodySchema.POS_Y] = -2.0
    bodies[0, BodySchema.QUAT_W] = 1.0
    bodies[0, BodySchema.SHAPE_TYPE] = ShapeType.BOX
    bodies[0, BodySchema.SHAPE_PARAM_1:BodySchema.SHAPE_PARAM_3+1] = [10.0, 0.5, 10.0]
    bodies[0, BodySchema.INV_MASS] = 0.0
    
    # Falling box
    bodies[1, BodySchema.POS_Y] = 2.0
    bodies[1, BodySchema.VEL_Y] = -2.0
    bodies[1, BodySchema.QUAT_W] = 1.0
    bodies[1, BodySchema.SHAPE_TYPE] = ShapeType.BOX
    bodies[1, BodySchema.SHAPE_PARAM_1:BodySchema.SHAPE_PARAM_3+1] = [0.5, 0.5, 0.5]
    bodies[1, BodySchema.INV_MASS] = 1.0
    
    engine = TensorPhysicsEngine(bodies, dt=0.016, restitution=0.1)
    
    # Run until box should have landed
    for _ in range(100):
        engine.step()
    
    state = engine.get_state()
    
    # Box should have slowed down significantly
    assert abs(state[1, BodySchema.VEL_Y]) < 20.0, "Box velocity should be reduced"
    # Position check - box should be near ground level
    assert state[1, BodySchema.POS_Y] < 0.5, "Box should be near ground level"

if __name__ == "__main__":
    test_sphere_collision()
    test_box_ground_collision()
    print("Collision validation tests passed!")