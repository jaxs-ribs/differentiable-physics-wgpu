# Debug Viz Implementation Summary

## What Was Implemented

The `debug_viz` tool has been refactored from a simple diff renderer into a **flexible state visualizer** with three distinct operating modes:

### 1. DEFAULT/DEMO MODE (No Arguments)
```bash
cargo run --features viz --bin debug_viz
```
- Displays a hardcoded demo scene (3 white spheres)
- No file dependencies required
- Perfect for testing the renderer itself

### 2. INSPECT MODE (Single File)
```bash
cargo run --features viz --bin debug_viz -- --oracle tests/oracle_dump.npy
# OR
cargo run --features viz --bin debug_viz -- --gpu tests/gpu_dump.npy
```
- Loads and displays a single state file
- Renders in neutral white color
- Ideal for visualizing Python oracle behavior

### 3. DIFF MODE (Both Files)
```bash
cargo run --features viz --bin debug_viz -- \
  --oracle tests/failures/test_name/cpu_state.npy \
  --gpu tests/failures/test_name/gpu_state.npy
```
- Compares two states side-by-side
- Oracle rendered in green/transparent
- GPU rendered in red/opaque
- Shows divergences between implementations

## Key Changes Made

1. **Updated Argument Parsing** (`src/bin/debug_viz.rs`)
   - Made both `--oracle` and `--gpu` arguments optional
   - Added mode detection logic based on provided arguments
   - Clear console output showing which mode is active

2. **Added Demo Scene Function**
   - `create_default_demo_scene()` returns 3 hardcoded spheres
   - Ground sphere (large, static)
   - Two smaller spheres positioned above

3. **Enhanced Color Logic** (`src/viz/dual_renderer.rs`)
   - Diff mode: Green/transparent vs Red/opaque
   - Inspect mode: White/opaque for single state
   - Dynamic color updates based on mode

4. **Documentation Updates**
   - Updated `README.md` with all three modes
   - Updated `AGENTS.md` with detailed mode descriptions
   - Clear usage examples for each mode

## Testing

To verify the implementation works correctly:

```bash
# Test Mode 1: Default/Demo
cargo run --features viz --bin debug_viz

# Test Mode 2: Inspect (requires NPY file)
cd tests
python3 create_test_dump.py  # Creates oracle_dump.npy
cargo run --features viz --bin debug_viz -- --oracle tests/oracle_dump.npy

# Test Mode 3: Diff (using same file for both)
cargo run --features viz --bin debug_viz -- \
  --oracle tests/oracle_dump.npy \
  --gpu tests/oracle_dump.npy
```

## Benefits

1. **Immediate Utility**: Can visualize Python oracle states without GPU implementation
2. **Progressive Enhancement**: Tool is useful at every stage of development
3. **Debugging Power**: Easily spot divergences when they occur
4. **Self-Contained Testing**: Demo mode requires no external dependencies

This implementation fulfills all requirements from the refined plan, making `debug_viz` a versatile tool for both immediate oracle inspection and future GPU-CPU diff visualization.