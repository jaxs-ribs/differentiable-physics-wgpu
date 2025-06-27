#!/usr/bin/env python3
"""Test boolean operations on empty tensors."""

from tinygrad import Tensor, dtypes

print("Testing empty tensor boolean operations...")

# Create empty tensors
empty_bool = Tensor.zeros((0,), dtype=dtypes.bool)
empty_float = Tensor.zeros((0,))

print(f"empty_bool dtype: {empty_bool.dtype}")
print(f"empty_float dtype: {empty_float.dtype}")

# Test comparison on empty
comp = empty_float <= 0
print(f"\nempty_float <= 0:")
print(f"  shape: {comp.shape}")
print(f"  dtype: {comp.dtype}")

# Test & operation
print("\nTesting empty_bool & comp:")
try:
    result = empty_bool & comp
    print(f"  Success! shape: {result.shape}, dtype: {result.dtype}")
except Exception as e:
    print(f"  Error: {e}")

# Test with multiplication instead
print("\nTesting multiplication instead of &:")
try:
    result = empty_bool * comp
    print(f"  Success! shape: {result.shape}, dtype: {result.dtype}")
except Exception as e:
    print(f"  Error: {e}")

# Direct test of the problematic line
print("\nTesting the exact problematic operation:")
contact_mask = Tensor.zeros((0,), dtype=dtypes.bool)
v_rel_normal = Tensor.zeros((0,))

print(f"contact_mask dtype: {contact_mask.dtype}")
print(f"v_rel_normal dtype: {v_rel_normal.dtype}")

comparison = v_rel_normal <= 0
print(f"v_rel_normal <= 0 dtype: {comparison.dtype}")

try:
    active_mask = contact_mask & comparison
    print(f"Success! active_mask shape: {active_mask.shape}")
except Exception as e:
    print(f"Error in & operation: {e}")
    print("\nTrying multiplication workaround:")
    active_mask = contact_mask * comparison
    print(f"Multiplication worked! shape: {active_mask.shape}")