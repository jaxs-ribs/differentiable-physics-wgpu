#!/usr/bin/env python3
"""Test script to investigate position corruption issue."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from physics.engine import TensorPhysicsEngine
from physics.types import BodySchema, ShapeType

def create_test_scene():
    """Create simple scene with falling sphere."""
    bodies = np.zeros((2, BodySchema.NUM_PROPERTIES), dtype=np.float32)
    
    # Static ground
    bodies[0, BodySchema.POS_Y] = -5.0
    bodies[0, BodySchema.QUAT_W] = 1.0
    bodies[0, BodySchema.SHAPE_TYPE] = ShapeType.BOX
    bodies[0, BodySchema.SHAPE_PARAM_1:BodySchema.SHAPE_PARAM_3+1] = [10.0, 0.5, 10.0]
    bodies[0, BodySchema.INV_MASS] = 0.0
    
    # Falling sphere
    bodies[1, BodySchema.POS_X:BodySchema.POS_Z+1] = [0.0, 5.0, 0.0]
    bodies[1, BodySchema.QUAT_W] = 1.0
    bodies[1, BodySchema.SHAPE_TYPE] = ShapeType.SPHERE
    bodies[1, BodySchema.SHAPE_PARAM_1] = 1.0
    bodies[1, BodySchema.INV_MASS] = 1.0
    bodies[1, BodySchema.INV_INERTIA_XX] = 2.5
    bodies[1, BodySchema.INV_INERTIA_YY] = 2.5
    bodies[1, BodySchema.INV_INERTIA_ZZ] = 2.5
    
    return bodies

def test_single_step_positions():
    """Test positions through single-step simulation."""
    print("Testing single-step simulation positions...")
    
    bodies = create_test_scene()
    engine = TensorPhysicsEngine(bodies, dt=0.016)
    
    # Track positions for several steps
    for i in range(20):
        state = engine.get_state()
        pos = state[1, BodySchema.POS_X:BodySchema.POS_Z+1]
        vel = state[1, BodySchema.VEL_X:BodySchema.VEL_Z+1]
        
        print(f"Step {i}: pos={pos}, vel={vel}")
        
        if np.all(pos == 1.0):
            print(f"ERROR: Position corruption detected at step {i}!")
            return False
            
        engine.step()
    
    return True

def test_nstep_positions():
    """Test positions through N-step simulation."""
    print("\nTesting N-step simulation positions...")
    
    bodies = create_test_scene()
    engine = TensorPhysicsEngine(bodies, dt=0.016)
    
    initial_state = engine.get_state()
    initial_pos = initial_state[1, BodySchema.POS_X:BodySchema.POS_Z+1]
    print(f"Initial: pos={initial_pos}")
    
    # Run 20 steps as single operation
    engine.run_simulation(20)
    
    final_state = engine.get_state()
    final_pos = final_state[1, BodySchema.POS_X:BodySchema.POS_Z+1]
    print(f"Final: pos={final_pos}")
    
    if np.all(final_pos == 1.0):
        print("ERROR: Position corruption detected in N-step!")
        return False
        
    return True

def test_long_simulation():
    """Test with longer simulation to find corruption point."""
    print("\nTesting longer simulation (100 steps)...")
    
    bodies = create_test_scene()
    engine = TensorPhysicsEngine(bodies, dt=0.016)
    
    corruption_found = False
    for i in range(100):
        state = engine.get_state()
        pos = state[1, BodySchema.POS_X:BodySchema.POS_Z+1]
        
        if i % 10 == 0:
            print(f"Step {i}: pos={pos}")
        
        if np.all(pos == 1.0):
            print(f"ERROR: Position corruption detected at step {i}!")
            corruption_found = True
            break
            
        engine.step()
    
    if not corruption_found:
        final_state = engine.get_state()
        final_pos = final_state[1, BodySchema.POS_X:BodySchema.POS_Z+1]
        print(f"Final position after 100 steps: {final_pos}")
        
    return not corruption_found

def main():
    """Run all position corruption tests."""
    print("Position Corruption Investigation")
    print("=" * 60)
    
    tests = [
        ("Single-step positions", test_single_step_positions),
        ("N-step positions", test_nstep_positions),
        ("Long simulation", test_long_simulation)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Summary:")
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())