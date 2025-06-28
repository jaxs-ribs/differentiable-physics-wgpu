# Physics Engine Test Suite

## Organization

```
tests/
├── unit/                    # Unit tests for individual components
│   ├── physics/            # Physics-specific unit tests
│   │   ├── bouncing/       # Bounce behavior tests
│   │   ├── collision/      # Collision detection/response tests
│   │   ├── impulse/        # Impulse calculation tests
│   │   ├── test_broadphase.py
│   │   ├── test_math_utils.py
│   │   └── ...             # Other physics component tests
│   └── custom_ops/         # Custom operations tests
├── integration/            # Integration and end-to-end tests
│   ├── test_energy_conservation.py
│   ├── test_simulation_stability.py
│   ├── test_fuzzing_stability.py
│   └── ...
├── benchmarks/             # Performance benchmarks
│   └── test_physics_step_performance.py
├── debugging/              # Debug utilities and temporary test files
└── run_ci.py              # Main CI test runner
```

## Quick Start

### Run CI Suite
```bash
# From physics_core directory
python3 tests/run_ci.py
```

### Run All Tests
```bash
python -m pytest tests/
```

### Run Specific Categories
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Benchmarks only
pytest tests/benchmarks/ -m benchmark

# Specific component
pytest tests/unit/physics/collision/
```

## Test Categories

### Unit Tests (`unit/`)
Test individual functions and classes in isolation:
- **physics/**: Core physics components
  - **bouncing/**: Restitution and bounce behavior
  - **collision/**: Contact detection and response
  - **impulse/**: Impulse calculations
  - **test_broadphase.py**: Differentiable broadphase
  - **test_math_utils.py**: Quaternion and matrix ops
- **custom_ops/**: Custom C extensions

### Integration Tests (`integration/`)
Test complete simulation scenarios:
- **test_energy_conservation.py**: Verify physical invariants
- **test_simulation_stability.py**: Numerical stability checks
- **test_fuzzing_stability.py**: Property-based testing with random scenes

### Benchmarks (`benchmarks/`)
Performance testing and profiling:
- **test_physics_step_performance.py**: Step performance with various body counts

### Debugging (`debugging/`)
Temporary test files for investigating issues:
- Should be cleaned up regularly
- Not part of CI pipeline
- Used for reproducing and fixing specific bugs

## Running Tests

### With Coverage
```bash
pytest tests/ --cov=physics --cov-report=html
```

### With Verbose Output
```bash
pytest tests/ -v -s
```

### Debug on Failure
```bash
pytest tests/ --pdb
```

## Writing New Tests

### Using Fixtures
Available in `conftest.py`:
```python
def test_my_feature(two_body_scene):
    engine = two_body_scene
    # Test code here
```

### Naming Conventions
- Test files: `test_*.py`
- Test functions: `test_*`
- Group related tests in subdirectories

## Continuous Integration

The `run_ci.py` script runs essential tests:
1. Import verification
2. Basic simulation
3. JIT compilation
4. NumPy-free core
5. Main script execution
6. Collision detection
7. Performance baseline