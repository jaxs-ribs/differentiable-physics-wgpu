#!/usr/bin/env python3
"""Test tensor type operations."""

from tinygrad import Tensor, dtypes

# Test comparison operations
a = Tensor([1.0, 2.0, 3.0])
b = Tensor([2.0, 2.0, 1.0])

print("Testing comparison operations:")
print(f"a = {a.numpy()}")
print(f"b = {b.numpy()}")

# Comparison should return bool
comparison = a <= b
print(f"a <= b = {comparison.numpy()}")
print(f"Type of comparison: {comparison.dtype}")

# Test with empty tensors
empty = Tensor.zeros((0,))
empty_bool = Tensor.zeros((0,), dtype=dtypes.bool)

print("\nTesting empty tensors:")
print(f"Empty tensor shape: {empty.shape}")
print(f"Empty bool tensor shape: {empty_bool.shape}")

# Try comparison on empty
empty_comparison = empty <= 0
print(f"empty <= 0 shape: {empty_comparison.shape}")
print(f"empty <= 0 dtype: {empty_comparison.dtype}")

# Test boolean operations
print("\nTesting boolean operations:")
bool1 = Tensor([True, False, True], dtype=dtypes.bool)
bool2 = Tensor([True, True, False], dtype=dtypes.bool)

result = bool1 & bool2
print(f"bool1 & bool2 = {result.numpy()}")

# Test with comparison result
comp_result = a <= 2.0
print(f"\na <= 2.0 = {comp_result.numpy()}")
print(f"dtype: {comp_result.dtype}")

# Can we use & with comparison results?
try:
    combined = comp_result & bool1
    print(f"comp_result & bool1 = {combined.numpy()}")
except Exception as e:
    print(f"Error combining: {e}")