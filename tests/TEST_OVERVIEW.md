# Physics Engine Test Overview

## Test Structure

The physics engine test suite is organized into several categories:

### 1. Unit Tests (`tests/unit/`)

#### Physics Core Tests
- **test_math_utils.py** - Tests quaternion operations, rotation matrices (pytest)
- **test_broadphase.py** - Tests broadphase collision detection (pytest)
- **test_angular_terms.py** - Tests angular physics calculations (standalone)
- **test_effective_restitution.py** - Tests restitution calculations (standalone)
- **test_resolve_count.py** - Tests collision resolution counting (standalone)

#### Specialized Physics Tests
- **bouncing/test_bounce_behavior.py** - Comprehensive bounce physics tests (standalone)
- **collision/test_collision_detection.py** - Collision detection tests (standalone)
- **impulse/test_impulse_resolution.py** - Impulse calculation tests (standalone)

#### Custom Ops Tests (Optional)
- **custom_ops/test_extension.py** - Device extension tests (pytest)
- **custom_ops/test_integration.py** - Custom ops integration (pytest)
- **custom_ops/test_patterns.py** - Pattern matching tests (pytest)

### 2. Integration Tests (`tests/integration/`)

- **test_minimal.py** - Basic integration test (standalone)
- **test_every_step.py** - Step-by-step collision tracking (standalone)
- **test_index_order.py** - Tests collision pair indexing (standalone)
- **test_large_timestep.py** - Tests with large dt values (standalone)
- **test_simulation_stability.py** - Tests simulation doesn't explode (pytest)
- **test_energy_conservation.py** - Tests energy conservation (pytest)
- **test_fuzzing_stability.py** - Fuzz testing for edge cases (pytest)

### 3. CI Tests (`tests/run_ci.py`)

The main CI runner includes:
- Import tests
- Basic simulation test
- JIT compilation test
- NumPy-free core verification
- Main script testing
- Collision detection test
- Performance benchmarks (not in quick mode)

## Running Tests

### Run all tests (with conda/pytest):
```bash
./ci
```

### Run quick tests (skip slow/optional tests):
```bash
./ci --quick
```

### Run individual test:
```bash
python tests/unit/physics/test_angular_terms.py
```

### Run pytest tests:
```bash
python -m pytest tests/unit/physics/test_math_utils.py -v
```

## CI Optimizations

When `CI=true` is set in the environment:
- Tests use reduced iteration counts
- Long-running simulations are shortened
- Timeouts are enforced on all tests

## Test Requirements

### Core Tests (Required for Merge)
- All unit physics tests must pass
- Basic integration tests must pass
- No NaN/Inf values in simulations
- Energy conservation within tolerance
- Collision detection working correctly

### Optional Tests
- Custom ops tests (experimental feature)
- Performance benchmarks
- Extended fuzzing tests