# Binary Overview

This directory contains executable programs for benchmarking, testing, and demonstrating the physics engine.

## Production Binaries

### Benchmarks
- **`benchmark`** - Performance benchmark for physics simulation
  ```bash
  cargo run --release --bin benchmark -- [body_count]
  ```
  Runs a physics simulation benchmark with specified number of bodies (default: 10,000).

- **`benchmark_full`** - Comprehensive benchmark suite
  ```bash
  cargo run --release --bin benchmark_full
  ```
  Runs multiple benchmark configurations and outputs detailed performance metrics.

### Demos
- **`demo_simple`** - Basic physics simulation with console output
  ```bash
  cargo run --release --bin demo_simple
  ```
  Demonstrates basic physics simulation, outputs state to console.

- **`demo_ascii`** - ASCII art visualization of physics simulation
  ```bash
  cargo run --release --bin demo_ascii
  ```
  Shows physics simulation as ASCII art in the terminal.

- **`demo_viz`** - 3D wireframe visualization demo
  ```bash
  cargo run --release --features viz --bin demo_viz
  ```
  Real-time 3D visualization of physics simulation with camera controls.

### Debug Tools
- **`debug_viz`** - Physics trace animation player
  ```bash
  cargo run --release --features viz --bin debug_viz -- --oracle path/to/trace.npy
  ```
  Plays back recorded physics simulations from NPY trace files with interactive controls.

## Test Binaries

### GPU Tests
- **`test_broadphase`** - Tests broadphase collision detection
  ```bash
  cargo run --bin test_broadphase
  ```
  Validates GPU broadphase collision detection implementation.

- **`test_broadphase_grid`** - Tests grid-based broadphase
  ```bash
  cargo run --bin test_broadphase_grid
  ```
  Tests uniform grid broadphase collision detection.

- **`test_contact_solver`** - Tests collision resolution
  ```bash
  cargo run --bin test_contact_solver
  ```
  Validates contact solver and collision response.

- **`test_sdf`** - Tests Signed Distance Functions
  ```bash
  cargo run --bin test_sdf
  ```
  Validates SDF calculations for all shape types.

- **`test_runner`** - Runs GPU integration tests
  ```bash
  cargo run --bin test_runner
  ```
  Executes a suite of GPU physics tests.

## Notes

- Binaries requiring visualization must be run with `--features viz`
- Release builds (`--release`) are recommended for benchmarks
- Debug binaries may produce verbose output for troubleshooting