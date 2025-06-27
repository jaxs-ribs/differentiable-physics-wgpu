#!/usr/bin/env python3
"""
Position Corruption Investigation Test

This debugging test investigates potential position corruption issues in the physics
engine where body positions might unexpectedly become all 1.0 or other incorrect values.

Why this is useful:
- Helps track down subtle bugs in tensor operations
- Validates state management across simulation steps
- Tests both single-step and N-step execution modes
- Identifies when corruption occurs in the simulation pipeline
- Ensures numerical stability over extended simulations

The tests cover:
1. Single-step position tracking over multiple iterations
2. N-step batch execution validation
3. Long-running simulation stability (100+ steps)
"""

import sys
import os
# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

import numpy as np
import time
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
    print("\n" + "=" * 60)
    print("TEST: Single-Step Position Tracking")
    print("=" * 60)
    print("\n[OBJECTIVE] Track body positions through individual simulation steps")
    print("[PURPOSE] Detect if/when position corruption occurs")
    
    print("\n[SETUP] Creating test scene...")
    bodies = create_test_scene()
    engine = TensorPhysicsEngine(bodies, dt=0.016)
    
    print("  Scene configuration:")
    print("    - Static ground box at Y=-5")
    print("    - Falling sphere starting at Y=5")
    print("    - Timestep: 0.016s (60 FPS)")
    
    # Track positions for several steps
    print("\n[EXECUTION] Running 20 single steps...")
    print("Step | Position (X,Y,Z)      | Velocity (X,Y,Z)      | Status")
    print("-" * 70)
    
    corruption_detected = False
    for i in range(20):
        state = engine.get_state()
        pos = state[1, BodySchema.POS_X:BodySchema.POS_Z+1]
        vel = state[1, BodySchema.VEL_X:BodySchema.VEL_Z+1]
        
        # Check for corruption patterns
        status = "OK"
        if np.all(pos == 1.0):
            status = "CORRUPTION: All positions = 1.0!"
            corruption_detected = True
        elif np.any(np.isnan(pos)):
            status = "CORRUPTION: NaN detected!"
            corruption_detected = True
        elif np.any(np.isinf(pos)):
            status = "CORRUPTION: Infinity detected!"
            corruption_detected = True
        
        print(f"{i:4d} | {pos[0]:6.3f},{pos[1]:6.3f},{pos[2]:6.3f} | "
              f"{vel[0]:6.3f},{vel[1]:6.3f},{vel[2]:6.3f} | {status}")
        
        if corruption_detected:
            print(f"\n[ERROR] Position corruption detected at step {i}!")
            print(f"[DEBUG] Full body state:")
            print(state[1])
            return False
            
        engine.step()
    
    print("\n[VALIDATION] Checking final state...")
    final_state = engine.get_state()
    final_pos = final_state[1, BodySchema.POS_Y]
    expected_fall = 0.5 * 9.81 * (20 * 0.016)**2  # Approximate
    
    print(f"  Initial Y position: 5.0")
    print(f"  Final Y position: {final_pos:.3f}")
    print(f"  Expected fall distance: ~{expected_fall:.3f}")
    
    if final_pos < 5.0 and final_pos > -5.0:
        print("  ✓ Sphere fell as expected")
        print("\n[SUCCESS] No position corruption in single-step mode")
        return True
    else:
        print("  ✗ Unexpected final position")
        return False

def test_nstep_positions():
    """Test positions through N-step simulation."""
    print("\n" + "=" * 60)
    print("TEST: N-Step Position Validation")
    print("=" * 60)
    print("\n[OBJECTIVE] Validate positions after batch N-step execution")
    print("[PURPOSE] Check if JIT compilation introduces corruption")
    
    print("\n[SETUP] Creating fresh test scene...")
    bodies = create_test_scene()
    engine = TensorPhysicsEngine(bodies, dt=0.016)
    
    initial_state = engine.get_state()
    initial_pos = initial_state[1, BodySchema.POS_X:BodySchema.POS_Z+1]
    initial_vel = initial_state[1, BodySchema.VEL_X:BodySchema.VEL_Z+1]
    
    print("\n[INITIAL STATE]")
    print(f"  Position: X={initial_pos[0]:.3f}, Y={initial_pos[1]:.3f}, Z={initial_pos[2]:.3f}")
    print(f"  Velocity: X={initial_vel[0]:.3f}, Y={initial_vel[1]:.3f}, Z={initial_vel[2]:.3f}")
    
    # Run 20 steps as single operation
    n_steps = 20
    print(f"\n[EXECUTION] Running {n_steps} steps in batch mode...")
    print("  [INFO] This uses JIT-compiled N-step execution")
    
    start_time = time.time()
    engine.run_simulation(n_steps)
    elapsed = time.time() - start_time
    
    print(f"  [TIMING] Completed in {elapsed*1000:.2f} ms")
    
    final_state = engine.get_state()
    final_pos = final_state[1, BodySchema.POS_X:BodySchema.POS_Z+1]
    final_vel = final_state[1, BodySchema.VEL_X:BodySchema.VEL_Z+1]
    
    print("\n[FINAL STATE]")
    print(f"  Position: X={final_pos[0]:.3f}, Y={final_pos[1]:.3f}, Z={final_pos[2]:.3f}")
    print(f"  Velocity: X={final_vel[0]:.3f}, Y={final_vel[1]:.3f}, Z={final_vel[2]:.3f}")
    
    print("\n[VALIDATION] Checking for corruption patterns...")
    
    # Check various corruption patterns
    checks_passed = True
    
    if np.all(final_pos == 1.0):
        print("  ✗ CORRUPTION: All positions equal to 1.0")
        checks_passed = False
    else:
        print("  ✓ Positions are not all 1.0")
        
    if np.any(np.isnan(final_pos)):
        print("  ✗ CORRUPTION: NaN values detected")
        checks_passed = False
    else:
        print("  ✓ No NaN values")
        
    if np.any(np.isinf(final_pos)):
        print("  ✗ CORRUPTION: Infinity values detected")
        checks_passed = False
    else:
        print("  ✓ No infinity values")
    
    # Physics validation
    print("\n[PHYSICS VALIDATION]")
    y_change = final_pos[1] - initial_pos[1]
    expected_y_change = -0.5 * 9.81 * (n_steps * 0.016)**2
    
    print(f"  Y position change: {y_change:.3f}")
    print(f"  Expected (gravity only): ~{expected_y_change:.3f}")
    
    if y_change < 0 and abs(y_change - expected_y_change) < 1.0:
        print("  ✓ Sphere fell with reasonable acceleration")
    else:
        print("  ✗ Unexpected motion")
        checks_passed = False
        
    if checks_passed:
        print("\n[SUCCESS] No position corruption in N-step mode")
    else:
        print("\n[FAILURE] Position corruption or physics error detected")
        
    return checks_passed

def test_long_simulation():
    """Test with longer simulation to find corruption point."""
    print("\n" + "=" * 60)
    print("TEST: Long Simulation Stability")
    print("=" * 60)
    print("\n[OBJECTIVE] Run extended simulation to find corruption points")
    print("[PURPOSE] Test numerical stability over many iterations")
    
    print("\n[SETUP] Creating test scene for 100-step simulation...")
    bodies = create_test_scene()
    engine = TensorPhysicsEngine(bodies, dt=0.016)
    
    total_steps = 100
    print(f"\n[CONFIG]")
    print(f"  Total steps: {total_steps}")
    print(f"  Timestep: 0.016s")
    print(f"  Total time: {total_steps * 0.016:.2f}s")
    print(f"  Expected to hit ground around step ~40")
    
    print("\n[EXECUTION] Running extended simulation...")
    print("\nStep | Y Position | Y Velocity | Status")
    print("-" * 50)
    
    corruption_found = False
    collision_detected = False
    max_velocity = 0.0
    
    for i in range(total_steps):
        state = engine.get_state()
        pos = state[1, BodySchema.POS_X:BodySchema.POS_Z+1]
        vel = state[1, BodySchema.VEL_X:BodySchema.VEL_Z+1]
        
        # Track maximum velocity
        current_speed = np.linalg.norm(vel)
        max_velocity = max(max_velocity, current_speed)
        
        # Detailed logging every 10 steps
        if i % 10 == 0:
            status = "Falling"
            
            # Check if sphere hit ground
            if pos[1] < -3.5 and not collision_detected:
                status = "Near ground"
                collision_detected = True
            elif vel[1] > 0 and collision_detected:
                status = "Bouncing up"
            elif abs(vel[1]) < 0.1 and collision_detected:
                status = "Settled"
                
            print(f"{i:4d} | {pos[1]:10.3f} | {vel[1]:10.3f} | {status}")
        
        # Check for corruption
        if np.all(pos == 1.0):
            print(f"\n[ERROR] Position corruption detected at step {i}!")
            print(f"  All positions became 1.0")
            print(f"  Full position vector: {pos}")
            print(f"  Full velocity vector: {vel}")
            corruption_found = True
            break
            
        if np.any(np.isnan(pos)) or np.any(np.isinf(pos)):
            print(f"\n[ERROR] NaN/Inf detected at step {i}!")
            print(f"  Position: {pos}")
            print(f"  Velocity: {vel}")
            corruption_found = True
            break
            
        engine.step()
    
    print("\n[SUMMARY]")
    if not corruption_found:
        final_state = engine.get_state()
        final_pos = final_state[1, BodySchema.POS_X:BodySchema.POS_Z+1]
        final_vel = final_state[1, BodySchema.VEL_X:BodySchema.VEL_Z+1]
        
        print(f"  Simulation completed: {total_steps} steps")
        print(f"  Final position: X={final_pos[0]:.3f}, Y={final_pos[1]:.3f}, Z={final_pos[2]:.3f}")
        print(f"  Final velocity: {np.linalg.norm(final_vel):.3f} m/s")
        print(f"  Maximum velocity reached: {max_velocity:.3f} m/s")
        print(f"  Collision detected: {'Yes' if collision_detected else 'No'}")
        
        # Validate final state
        if final_pos[1] > -3.0:
            print("\n[WARNING] Sphere didn't reach ground - possible issue")
        elif final_pos[1] < -10.0:
            print("\n[WARNING] Sphere fell through ground - collision detection issue")
        else:
            print("\n[SUCCESS] Simulation behaved as expected")
    else:
        print("  Simulation terminated due to corruption")
        
    return not corruption_found

def main():
    """Run all position corruption tests."""
    print("\n" + "#" * 60)
    print("# POSITION CORRUPTION INVESTIGATION")
    print("#" * 60)
    print(f"\n[START] Investigation started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n[PURPOSE] Debug potential position corruption in physics engine")
    print("[CONTEXT] Looking for cases where positions become all 1.0 or NaN")
    
    tests = [
        ("Single-step positions", test_single_step_positions),
        ("N-step positions", test_nstep_positions),
        ("Long simulation", test_long_simulation)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            print(f"\n[RUNNING] {name}...")
            passed = test_func()
            results.append((name, passed))
            
            if passed:
                print(f"\n[✓] {name} completed successfully\n")
            else:
                print(f"\n[✗] {name} failed\n")
                
        except Exception as e:
            print(f"\n[EXCEPTION] Error in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "#" * 60)
    print("# INVESTIGATION SUMMARY")
    print("#" * 60)
    
    print("\n[RESULTS]")
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status} - {name}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n[CONCLUSION]")
    if all_passed:
        print("  No position corruption detected in any test mode")
        print("  The physics engine maintains numerical stability")
        print("  Both single-step and N-step modes work correctly")
    else:
        print("  Position corruption or instability detected!")
        print("  Review failed tests for specific failure modes")
        print("  Check tensor operations and JIT compilation")
    
    print(f"\n[END] Investigation completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())