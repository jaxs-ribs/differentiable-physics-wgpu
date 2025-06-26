# Python Tests Overview

This directory contains the Python reference implementation and test suite for validating the GPU physics engine.

## Core Reference Implementation

- **`reference.py`** - Complete physics engine reference implementation
  ```bash
  # Import as module
  from reference import PhysicsEngine, Body, ShapeType
  ```
  The golden standard implementation used to validate GPU results.

- **`reference_simple.py`** - Simplified reference for basic tests
  ```bash
  # Import for simple tests
  from reference_simple import PhysicsEngine
  ```
  Minimal physics implementation for quick validation.

## Physics Tests

### Collision Detection
- **`test_sdf.py`** - Comprehensive SDF testing
  ```bash
  python3 test_sdf.py
  ```
  Tests all shape-to-shape distance calculations.

- **`test_sdf_quick.py`** - Quick SDF validation
  ```bash
  python3 test_sdf_quick.py
  ```
  Fast subset of SDF tests for rapid iteration.

- **`test_sdf_fuzz.py`** - Property-based SDF testing
  ```bash
  python3 test_sdf_fuzz.py
  ```
  Uses Hypothesis to test SDF mathematical properties.

- **`test_broadphase.py`** - Broadphase collision tests
  ```bash
  python3 test_broadphase.py
  ```
  Tests spatial partitioning for collision detection.

- **`test_broadphase_sap.py`** - Sweep and Prune tests
  ```bash
  python3 test_broadphase_sap.py
  ```
  Tests the Sweep and Prune broadphase algorithm.

### Dynamics & Integration
- **`test_integrator.py`** - Integration method tests
  ```bash
  python3 test_integrator.py
  ```
  Validates semi-implicit Euler integration.

- **`test_dynamics.py`** - Rotational dynamics tests
  ```bash
  python3 test_dynamics.py
  ```
  Tests angular momentum and torque calculations.

- **`test_energy.py`** - Energy conservation tests
  ```bash
  python3 test_energy.py
  ```
  Validates energy conservation in collisions.

### Collision Resolution
- **`test_contact_solver.py`** - CPU contact solver tests
  ```bash
  python3 test_contact_solver.py
  ```
  Tests impulse-based collision resolution.

- **`test_contact_solver_gpu.py`** - GPU contact solver validation
  ```bash
  python3 test_contact_solver_gpu.py
  ```
  Compares GPU contact solver against CPU reference.

### Integration Tests
- **`test_implementations.py`** - Basic implementation tests
  ```bash
  python3 test_implementations.py
  ```
  Quick validation of core functionality.

- **`test_stability_stress.py`** - Stability stress tests
  ```bash
  python3 test_stability_stress.py
  ```
  Tests simulation stability under extreme conditions.

- **`test_minimal.py`** - Minimal functionality test
  ```bash
  python3 test_minimal.py
  ```
  Bare minimum test to verify setup.

## GPU Validation Tests

- **`test_sdf_gpu.py`** - GPU SDF validation
  ```bash
  python3 test_sdf_gpu.py
  ```
  Compares GPU SDF calculations against reference.

- **`test_new_features.py`** - New feature validation
  ```bash
  python3 test_new_features.py
  ```
  Tests recently added functionality.

## Utilities

- **`plot_energy.py`** - Energy conservation plotter
  ```bash
  python3 plot_energy.py [--steps N] [--output file.png]
  ```
  Visualizes total system energy over time to detect drift.

- **`create_test_dump.py`** - Generate physics trace
  ```bash
  python3 create_test_dump.py
  ```
  Creates NPY trace file for debug_viz playback.

- **`conftest.py`** - PyTest configuration
  ```bash
  # Automatically loaded by pytest
  ```
  Configures test discovery and fixtures.

## Running Test Suites

```bash
# Run all quick tests
python3 test_sdf_quick.py && python3 test_implementations.py && python3 test_minimal.py

# Run comprehensive tests
python3 test_sdf.py && python3 test_energy.py && python3 test_dynamics.py

# Run property-based tests (slower)
python3 test_sdf_fuzz.py

# Create and view a simulation
python3 create_test_dump.py
cd ..
cargo run --features viz --bin debug_viz -- --oracle tests/oracle_dump.npy
```

## Dependencies

- NumPy
- Matplotlib (optional, for plotting)
- Hypothesis (for property-based tests)
- PyTest (for test discovery)