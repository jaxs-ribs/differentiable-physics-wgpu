# Bug Fixes Summary

## Position Corruption Issue

### Problem
The physics simulation was experiencing position corruption where sphere positions would become `[1, 1, 1]` after a few simulation steps.

### Root Causes

1. **Empty Contact Handling in Solver**
   - When the narrowphase detected 0 contacts but still called `resolve_collisions`, the solver would try to gather data with empty indices
   - This caused undefined behavior and NaN values to appear

2. **TinyGrad Tensor.where() Bug with NaN**
   - TinyGrad's `Tensor.where()` has a bug where it converts NaN values to 1.0
   - When the condition is True and the value is NaN, it returns 1.0 instead of NaN
   - This caused NaN positions to become `[1, 1, 1]`

### Solutions

1. **Fixed Empty Contact Handling** (solver.py)
   - Added check for empty contacts in `resolve_collisions`
   - When `n_contacts == 0`, create dummy data with zero mask to ensure no changes are applied
   - This prevents the gather operation from failing with empty indices

2. **Avoided Tensor.where() in Integration** (integration.py)
   - Replaced `Tensor.where()` calls with multiplication-based masking
   - Instead of `vel.where(is_dynamic, vel + gravity * dt)`, use `vel + gravity * dt * is_dynamic`
   - This avoids the NaN conversion bug entirely

### Testing
- Created comprehensive test scripts to reproduce and verify the fix
- All CI tests pass after the fixes
- The simulation now runs stably without position corruption

### Files Modified
- `physics/solver.py`: Added empty contact handling
- `physics/integration.py`: Replaced Tensor.where() with multiplication masking