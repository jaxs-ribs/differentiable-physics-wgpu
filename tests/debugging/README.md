# Debugging Tests

This directory contains specialized debugging tests and scripts used to diagnose and fix issues in the physics engine.

## Structure

### Position Corruption Investigation
Tests that helped identify and fix the NaN/position corruption bug:

- `test_position_corruption.py` - Main test to reproduce position corruption issue
- `find_nan_source.py` - Traces simulation to find where NaN first appears
- `debug_corruption.py` - Step-by-step debugging of physics pipeline
- `debug_solver.py` - Focuses on solver behavior with edge cases
- `debug_step2.py` - Investigates specific problematic simulation step
- `full_trace.py` - Complete trace comparing JIT vs non-JIT execution
- `trace_step.py` - Detailed single-step physics pipeline trace

### Low-Level Tests
Tests for specific tensor operations and JIT behavior:

- `test_nan_propagation.py` - Tests how NaN values propagate through tensor operations
- `test_empty_contacts.py` - Tests solver behavior with empty contact arrays
- `test_jit_early_return.py` - Tests JIT compilation with early returns
- `fix_integration.py` - Tests fixes for integration NaN handling

## Purpose

These tests are not part of the regular test suite. They are:
- Used for deep debugging of specific issues
- Helpful for understanding low-level behavior
- Examples of how to trace through the physics pipeline
- Documentation of fixed bugs and their root causes

## Running

Individual tests can be run directly:
```bash
python3 tests/debugging/test_position_corruption.py
```

These tests are NOT included in the main CI as they are diagnostic tools rather than regression tests.