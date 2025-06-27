# Physics Engine Test Suite

This comprehensive test suite ensures the robustness, correctness, and performance of our differentiable physics engine. The tests are organized hierarchically and cover everything from low-level math utilities to full integration tests.

## Overview

The test suite is structured into three main categories:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test the full physics pipeline and emergent behaviors
- **Benchmarks**: Measure performance to prevent regressions

## Directory Structure

```
tests/
├── conftest.py              # Shared pytest fixtures
├── unit/
│   └── physics/
│       ├── test_broadphase.py    # Differentiable broadphase tests
│       └── test_math_utils.py    # Quaternion and matrix operations
├── integration/
│   ├── test_energy_conservation.py  # Physical invariants
│   └── test_simulation_stability.py  # Numerical stability
└── benchmarks/
    └── test_physics_step_performance.py  # Performance metrics
```

## Running the Tests

### Install Dependencies

```bash
pip install pytest pytest-benchmark numpy tinygrad
```

### Run All Tests

```bash
# From the physics_core directory
pytest tests/
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Benchmarks only
pytest tests/benchmarks/ -m benchmark
```

### Run with Verbose Output

```bash
pytest tests/ -v
```

### Run with Coverage Report

```bash
pytest tests/ --cov=physics --cov-report=html
```

## Test Descriptions

### Unit Tests

#### `test_math_utils.py`
Tests the correctness of quaternion operations and rotation matrices:
- Quaternion multiplication
- Quaternion to rotation matrix conversion
- Vector rotation by quaternion
- Validation of rotation matrix properties (orthogonality, determinant)

#### `test_broadphase.py`
Tests the differentiable all-pairs broadphase collision detection:
- Correct generation of all N*(N-1)/2 unique pairs
- AABB collision detection for spheres and boxes
- Handling of rotated shapes
- Edge cases (empty scene, single body)

### Integration Tests

#### `test_energy_conservation.py`
Verifies that the simulation conserves energy appropriately:
- Total kinetic energy doesn't spontaneously increase
- Energy is properly dissipated (not created) in collisions
- Tests with various scene configurations

#### `test_simulation_stability.py`
Ensures the simulation remains stable under various conditions:
- Box stacks settle without exploding
- High-velocity collisions remain bounded
- Random chaotic scenes don't diverge
- Zero timestep handling

### Benchmarks

#### `test_physics_step_performance.py`
Measures performance of key operations:
- Full physics step with 10, 50, and 200 bodies
- Differentiable broadphase scaling
- Narrowphase performance with many collisions

## Example Test Runs

### Quick Smoke Test
```bash
pytest tests/unit/physics/test_math_utils.py::TestQuaternionOperations::test_quat_to_rotmat_identity -v
```

### Stability Test with Output
```bash
pytest tests/integration/test_simulation_stability.py::TestSimulationStability::test_box_stack_is_stable -v -s
```

### Performance Benchmark
```bash
pytest tests/benchmarks/test_physics_step_performance.py::TestPhysicsPerformance::test_physics_step_performance_medium -v -s
```

## Writing New Tests

### Using Fixtures

The `conftest.py` file provides reusable test scenes:

```python
def test_my_physics_feature(two_body_scene):
    """Test using the two-body collision fixture."""
    engine = two_body_scene
    # Your test code here
```

Available fixtures:
- `two_body_scene`: Sphere moving toward a box
- `multi_body_stack_scene`: Stack of 5 boxes
- `random_bodies_scene`: 20 random spheres and boxes

### Adding New Test Categories

To add a new test category (e.g., for a renderer):

1. Create a new directory under `tests/unit/` or `tests/integration/`
2. Add an `__init__.py` file
3. Follow the naming convention: `test_*.py`
4. Use descriptive test function names starting with `test_`

## Continuous Integration

These tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run Physics Engine Tests
  run: |
    pip install -r requirements.txt
    pytest tests/ --cov=physics --cov-report=xml
```

## Tips for Debugging Failed Tests

1. Run with `-v -s` to see print statements and detailed output
2. Use `--pdb` to drop into debugger on failure
3. Run individual tests to isolate issues
4. Check energy conservation tests for numerical precision issues
5. For stability tests, visualize the simulation to spot issues

## Performance Testing

To track performance over time:

```bash
# Run benchmarks and save results
pytest tests/benchmarks/ -m benchmark --benchmark-json=benchmark_results.json

# Compare with previous results
pytest tests/benchmarks/ -m benchmark --benchmark-compare=benchmark_results.json
```