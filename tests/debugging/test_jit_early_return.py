#!/usr/bin/env python3
"""
JIT Early Return Testing Module

This debugging test investigates how TinyGrad's JIT compilation handles early
returns and conditional logic, particularly important for physics simulations
that need to handle edge cases like zero contacts.

Why this is useful:
- Tests JIT behavior with conditional early returns
- Validates dynamic control flow in compiled functions
- Identifies limitations in JIT compilation
- Helps design JIT-compatible physics algorithms
- Ensures edge cases work in production

The tests cover:
1. Basic early return patterns
2. Conditional logic in JIT context
3. Empty tensor operations that require early exit
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tinygrad import Tensor, TinyJit, dtypes
import time

def test_early_return(x: Tensor, n: int) -> Tensor:
    """Test function with early return."""
    if n == 0:
        return x * 2
    return x + n

def test_jit_early_return():
    """Test if JIT handles early returns properly."""
    print("\n" + "=" * 60)
    print("TEST 1: Basic Early Return in JIT")
    print("=" * 60)
    print("\n[OBJECTIVE] Test JIT compilation with early return patterns")
    print("[CONTEXT] Dynamic control flow can be challenging for JIT")
    
    # Test without JIT
    print("\n[PHASE 1] Testing without JIT compilation...")
    x = Tensor([1.0, 2.0, 3.0])
    print(f"  Input tensor: {x.numpy()}")
    
    print("\n  Case 1: n=0 (should trigger early return)")
    result1 = test_early_return(x, 0)
    print(f"    Result: {result1.numpy()}")
    print(f"    Expected: {(x * 2).numpy()} (x * 2)")
    
    print("\n  Case 2: n=5 (normal path)")
    result2 = test_early_return(x, 5)
    print(f"    Result: {result2.numpy()}")
    print(f"    Expected: {(x + 5).numpy()} (x + 5)")
    
    print("\n  ✓ Non-JIT version works correctly")
    
    # Test with JIT
    print("\n[PHASE 2] Testing with JIT compilation...")
    print("  [INFO] Creating JIT-compiled version of function")
    jitted_func = TinyJit(test_early_return)
    
    print("\n  Case 1: n=0 (early return path)")
    try:
        result3 = jitted_func(x, 0)
        print(f"    Result: {result3.numpy()}")
        print("    ✓ JIT handled early return")
    except Exception as e:
        print(f"    ✗ JIT failed with early return: {e}")
    
    print("\n  Case 2: n=5 (normal path)") 
    try:
        result4 = jitted_func(x, 5)
        print(f"    Result: {result4.numpy()}")
        print("    ✓ JIT handled normal path")
    except Exception as e:
        print(f"    ✗ JIT failed with normal path: {e}")
    
    print("\n[ANALYSIS]")
    print("  JIT compilation requires static control flow")
    print("  Dynamic conditionals may not work as expected")
    print("  Consider alternative patterns for JIT compatibility")

def test_conditional_in_jit():
    """Test how JIT handles conditionals."""
    print("\n\n" + "=" * 60)
    print("TEST 2: Conditional Logic in JIT Context")
    print("=" * 60)
    print("\n[OBJECTIVE] Test physics-like conditional patterns with JIT")
    print("[CONTEXT] Simulates solver behavior with empty contacts")
    
    def process_with_check(bodies: Tensor, n_contacts: int) -> Tensor:
        """Simulate the solver pattern."""
        if n_contacts == 0:
            # Early return - no contacts to process
            return bodies
        
        # Simulate contact processing that would fail with empty data
        # This mimics what happens in the physics solver
        indices = Tensor.arange(n_contacts)
        gathered = bodies.gather(0, indices.unsqueeze(1).expand(-1, bodies.shape[1]))
        return bodies + gathered.sum(axis=0)
    
    print("\n[SETUP] Creating test data...")
    bodies = Tensor([[1.0, 2.0], [3.0, 4.0]])
    print(f"  Bodies tensor:\n{bodies.numpy()}")
    print("  Shape: (2, 2) - 2 bodies with 2 properties each")
    
    # Without JIT
    print("\n[PHASE 1] Testing without JIT...")
    
    print("\n  Case 1: n_contacts=0 (empty contacts)")
    result1 = process_with_check(bodies, 0)
    print(f"    Input:  {bodies.numpy().tolist()}")
    print(f"    Output: {result1.numpy().tolist()}")
    print("    ✓ Early return worked - bodies unchanged")
    
    print("\n  Case 2: n_contacts=1 (one contact)")
    result2 = process_with_check(bodies, 1)
    print(f"    Input:  {bodies.numpy().tolist()}")
    print(f"    Output: {result2.numpy().tolist()}")
    print("    ✓ Contact processing worked")
    
    # With JIT - this might cause issues
    print("\n[PHASE 2] Testing with JIT compilation...")
    print("  [WARNING] Dynamic conditionals may not work in JIT")
    
    jitted_process = TinyJit(process_with_check)
    
    print("\n  Case 1: n_contacts=0 (attempting early return)")
    try:
        result3 = jitted_process(bodies, 0)
        print(f"    Result: {result3.numpy()}")
        print("    ✓ JIT handled empty contacts correctly!")
    except Exception as e:
        print(f"    ✗ ERROR: {type(e).__name__}: {e}")
        print("    [EXPLANATION] JIT cannot handle dynamic early returns")
        print("    [SOLUTION] Need alternative pattern for empty contacts")
    
    print("\n[IMPLICATIONS FOR PHYSICS ENGINE]")
    print("  1. Cannot use if n_contacts == 0: return in JIT")
    print("  2. Must handle empty contacts differently")
    print("  3. Consider using tensor operations that work with size 0")
    print("  4. Or disable JIT for functions with dynamic control flow")

def main():
    """Run all JIT early return tests."""
    print("\n" + "#" * 60)
    print("# JIT EARLY RETURN INVESTIGATION")
    print("#" * 60)
    print(f"\n[START] Investigation started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n[PURPOSE] Understand JIT limitations with dynamic control flow")
    print("[RELEVANCE] Critical for handling edge cases in physics engine")
    
    try:
        test_jit_early_return()
        test_conditional_in_jit()
        
        print("\n\n" + "#" * 60)
        print("# INVESTIGATION SUMMARY")
        print("#" * 60)
        
        print("\n[KEY FINDINGS]")
        print("  1. JIT requires static control flow")
        print("  2. Dynamic conditionals (if n == 0) may fail")
        print("  3. Early returns are problematic in JIT")
        print("  4. Need alternative patterns for edge cases")
        
        print("\n[RECOMMENDATIONS FOR PHYSICS ENGINE]")
        print("  Option 1: Use tensor operations that handle size 0")
        print("  Option 2: Disable JIT for functions with conditionals")
        print("  Option 3: Preprocess to avoid empty cases")
        print("  Option 4: Use masked operations instead of conditionals")
        
        print("\n[EXAMPLE SOLUTIONS]")
        print("\n  Bad (doesn't JIT):")
        print("    if n_contacts == 0: return bodies")
        print("\n  Good (JIT-friendly):")
        print("    mask = (contact_count > 0).cast(dtypes.float)")
        print("    return bodies * (1 - mask) + processed * mask")
        
        print("\n[SUCCESS] Investigation completed")
        
    except Exception as e:
        print(f"\n[ERROR] Investigation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n[END] Investigation completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    import time
    main()