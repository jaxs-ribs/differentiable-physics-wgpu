#!/usr/bin/env python3
"""Test if early return works in JIT context."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tinygrad import Tensor, TinyJit

def test_early_return(x: Tensor, n: int) -> Tensor:
    """Test function with early return."""
    if n == 0:
        return x * 2
    return x + n

def test_jit_early_return():
    """Test if JIT handles early returns properly."""
    print("Testing early return in JIT...")
    
    # Test without JIT
    x = Tensor([1.0, 2.0, 3.0])
    result1 = test_early_return(x, 0)
    print(f"Without JIT (n=0): {result1.numpy()}")
    
    result2 = test_early_return(x, 5)
    print(f"Without JIT (n=5): {result2.numpy()}")
    
    # Test with JIT
    jitted_func = TinyJit(test_early_return)
    
    # This might not work as expected with JIT
    result3 = jitted_func(x, 0)
    print(f"With JIT (n=0): {result3.numpy()}")
    
    result4 = jitted_func(x, 5) 
    print(f"With JIT (n=5): {result4.numpy()}")

def test_conditional_in_jit():
    """Test how JIT handles conditionals."""
    print("\n\nTesting conditionals in JIT context...")
    
    def process_with_check(bodies: Tensor, n_contacts: int) -> Tensor:
        """Simulate the solver pattern."""
        if n_contacts == 0:
            # Early return
            return bodies
        
        # Simulate some processing that would fail with empty data
        # This is what's happening in our solver
        indices = Tensor.arange(n_contacts)
        gathered = bodies.gather(0, indices.unsqueeze(1).expand(-1, bodies.shape[1]))
        return bodies + gathered.sum(axis=0)
    
    bodies = Tensor([[1.0, 2.0], [3.0, 4.0]])
    
    # Without JIT
    print("Without JIT:")
    result1 = process_with_check(bodies, 0)
    print(f"  n_contacts=0: {result1.numpy()}")
    
    result2 = process_with_check(bodies, 1)
    print(f"  n_contacts=1: {result2.numpy()}")
    
    # With JIT - this might cause issues
    print("\nWith JIT:")
    jitted_process = TinyJit(process_with_check)
    try:
        result3 = jitted_process(bodies, 0)
        print(f"  n_contacts=0: {result3.numpy()}")
    except Exception as e:
        print(f"  n_contacts=0: ERROR - {e}")

if __name__ == "__main__":
    test_jit_early_return()
    test_conditional_in_jit()