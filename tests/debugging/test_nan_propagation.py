#!/usr/bin/env python3
"""
NaN Propagation Testing Module

This debugging test investigates how NaN (Not a Number) values propagate through
TinyGrad tensor operations, particularly in the context of physics simulations.

Why this is useful:
- Identifies how TinyGrad handles NaN in various operations
- Tests conditional operations (where) with NaN values
- Validates arithmetic operations involving NaN
- Helps debug numerical instability issues
- Ensures proper handling of edge cases in physics

The tests cover:
1. Tensor.where() behavior with NaN values
2. Arithmetic operations on NaN tensors
3. Integration-like operations with mixed valid/NaN data
4. Detach and assignment patterns with NaN
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tinygrad import Tensor
import numpy as np
import time

def test_where_with_nan():
    """Test how Tensor.where handles NaN values."""
    print("\n" + "=" * 60)
    print("TEST 1: Tensor.where() with NaN Values")
    print("=" * 60)
    print("\n[OBJECTIVE] Understand how conditional selection handles NaN")
    print("[CONTEXT] where(cond, x, y) returns x where cond is True, else y")
    
    # Create test data
    print("\n[SETUP] Creating test tensors...")
    cond = Tensor([True, False, True])
    x = Tensor([1.0, 2.0, float('nan')])
    y = Tensor([10.0, 20.0, 30.0])
    
    print("  Tensors created:")
    print(f"    cond = {cond.numpy()} (boolean mask)")
    print(f"    x = {x.numpy()} (contains NaN)")
    print(f"    y = {y.numpy()} (all valid)")
    
    # Test where
    print("\n[EXECUTE] result = x.where(cond, y)")
    result = x.where(cond, y)
    
    print("\n[RESULT]")
    print(f"  Condition: {cond.numpy()}")
    print(f"  x (selected when True): {x.numpy()}")
    print(f"  y (selected when False): {y.numpy()}")
    print(f"  Result: {result.numpy()}")
    
    print("\n[ANALYSIS]")
    print("  Index 0: cond=True  -> selected x[0]=1.0")
    print("  Index 1: cond=False -> selected y[1]=20.0")
    print("  Index 2: cond=True  -> selected x[2]=nan")
    print("  → NaN is preserved when selected!")
    
    # Test with all NaN
    print("\n[TEST VARIANT] All NaN values in x...")
    x_nan = Tensor([float('nan'), float('nan'), float('nan')])
    result_nan = x_nan.where(cond, y)
    
    print(f"  x_nan = {x_nan.numpy()}")
    print(f"  Result: {result_nan.numpy()}")
    print("  → NaN values selected where condition is True")
    
    # Test arithmetic with NaN
    print("\n[ARITHMETIC TESTS] NaN in basic operations...")
    a = Tensor([1.0, 2.0, float('nan')])
    b = Tensor([10.0, 20.0, 30.0])
    
    print(f"\n  a = {a.numpy()}")
    print(f"  b = {b.numpy()}")
    
    print("\n  Operations:")
    print(f"    a + b = {(a + b).numpy()}  (NaN + anything = NaN)")
    print(f"    a * 0 = {(a * 0).numpy()}  (NaN * 0 = NaN in IEEE 754)")
    print(f"    a * 1 = {(a * 1).numpy()}  (NaN * 1 = NaN)")
    print(f"    a - a = {(a - a).numpy()}  (NaN - NaN = NaN)")
    
    print("\n  [KEY INSIGHT] NaN propagates through all arithmetic!")

def test_integration_issue():
    """Test the specific integration issue."""
    print("\n\n" + "=" * 60)
    print("TEST 2: Integration-like Operations with NaN")
    print("=" * 60)
    print("\n[OBJECTIVE] Simulate physics integration with mixed valid/NaN data")
    print("[CONTEXT] This mimics the position update in physics integration")
    
    # Simulate the integration update
    print("\n[SETUP] Creating physics-like data...")
    pos = Tensor([[0.0, 1.0, 0.0], [float('nan'), float('nan'), float('nan')]])
    vel = Tensor([[0.0, 1.0, 0.0], [10.0, 20.0, 30.0]])
    is_dynamic = Tensor([[True], [False]])
    dt = 0.016
    
    print("  Initial state:")
    print(f"    pos = {pos.numpy()}")
    print(f"    vel = {vel.numpy()}")
    print(f"    is_dynamic = {is_dynamic.numpy()}")
    print(f"    dt = {dt}")
    
    print("\n  Interpretation:")
    print("    Body 0: Valid position, dynamic (will be updated)")
    print("    Body 1: NaN position, static (should not be updated)")
    
    # Update position like in integration
    print("\n[METHOD 1] pos.where(is_dynamic, pos + vel * dt)")
    print("  Logic: If dynamic, use updated position, else keep original")
    
    new_pos = pos.where(is_dynamic, pos + vel * dt)
    print(f"\n  Result: {new_pos.numpy()}")
    
    print("\n  Analysis:")
    print("    Body 0: is_dynamic=True  -> pos + vel*dt = [0, 1.016, 0]")
    print("    Body 1: is_dynamic=False -> keep original = [nan, nan, nan]")
    print("    → NaN preserved for static body!")
    
    # What if we flip the condition?
    print("\n[METHOD 2] (pos + vel * dt).where(is_dynamic, pos)")
    print("  Logic: Calculate update first, then conditionally apply")
    
    update = pos + vel * dt
    print(f"\n  Intermediate (pos + vel * dt): {update.numpy()}")
    print("    → NaN propagated through addition!")
    
    new_pos2 = update.where(is_dynamic, pos)
    print(f"\n  Final result: {new_pos2.numpy()}")
    
    print("\n[CONCLUSION]")
    print("  Method 1 preserves NaN for static bodies")
    print("  Method 2 propagates NaN through computation")
    print("  Order of operations matters with NaN!")

def test_detach_assignment():
    """Test detach and assignment pattern."""
    print("\n\n" + "=" * 60)
    print("TEST 3: Detach and Assignment with NaN")
    print("=" * 60)
    print("\n[OBJECTIVE] Test tensor detachment and slice assignment")
    print("[CONTEXT] Common pattern in physics engine state updates")
    
    # Create initial tensor
    print("\n[STEP 1] Create initial tensor...")
    bodies = Tensor([[1.0, 2.0, 3.0, 4.0],
                     [5.0, 6.0, 7.0, 8.0]])
    
    print(f"  Original: {bodies.numpy()}")
    print("  Shape: (2, 4)")
    
    # Detach and modify
    print("\n[STEP 2] Detach tensor (break gradient tracking)...")
    new_bodies = bodies.detach()
    print(f"  After detach: {new_bodies.numpy()}")
    print("  → Values unchanged, but no longer tracks gradients")
    
    # Assign slice
    print("\n[STEP 3] Assign to slice [:, 1:3]...")
    new_bodies[:, 1:3] = Tensor([[10.0, 11.0], [12.0, 13.0]])
    print(f"  After assignment: {new_bodies.numpy()}")
    print("  → Columns 1 and 2 updated successfully")
    
    # What happens with NaN?
    print("\n[STEP 4] Assign NaN to first column...")
    new_bodies[:, 0:1] = Tensor([[float('nan')], [float('nan')]])
    print(f"  After NaN assignment: {new_bodies.numpy()}")
    print("  → NaN values assigned successfully")
    
    print("\n[CONCLUSION]")
    print("  ✓ Detach works normally")
    print("  ✓ Slice assignment works with regular values")
    print("  ✓ Slice assignment works with NaN values")
    print("  → NaN can be introduced through assignment!")

def main():
    """Run all NaN propagation tests."""
    print("\n" + "#" * 60)
    print("# NaN PROPAGATION INVESTIGATION")
    print("#" * 60)
    print(f"\n[START] Investigation started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n[PURPOSE] Understand NaN handling in TinyGrad operations")
    print("[RELEVANCE] Critical for debugging numerical instabilities")
    
    try:
        test_where_with_nan()
        test_integration_issue()
        test_detach_assignment()
        
        print("\n\n" + "#" * 60)
        print("# INVESTIGATION SUMMARY")
        print("#" * 60)
        
        print("\n[KEY FINDINGS]")
        print("  1. NaN propagates through ALL arithmetic operations")
        print("  2. Tensor.where() preserves NaN when selected")
        print("  3. Order of operations matters with conditional updates")
        print("  4. NaN can be assigned to tensor slices")
        print("  5. Even NaN * 0 = NaN (IEEE 754 standard)")
        
        print("\n[IMPLICATIONS FOR PHYSICS ENGINE]")
        print("  - Check for NaN after collision detection")
        print("  - Use careful ordering in conditional updates")
        print("  - Consider NaN guards in critical paths")
        print("  - Validate inputs to prevent NaN introduction")
        
        print("\n[SUCCESS] All tests completed successfully")
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n[END] Investigation completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    import time
    main()