# Physics Engine Test Structure

This directory contains all tests for the XPBD physics engine. Tests are organized by their purpose and scope.

## Test Categories

### 1. Unit Tests (`tests/unit/`)
Tests for individual components in isolation.

#### XPBD Phases (`tests/unit/xpbd/`)
- `test_integration.py` - Forward prediction/integration step
- `test_broadphase.py` - Spatial hash collision detection  
- `test_narrowphase.py` - Contact generation
- `test_solver.py` - Constraint solving
- `test_velocity_update.py` - Velocity reconciliation
- `test_prediction.py` - Detailed forward prediction tests

#### Utilities (`tests/unit/utilities/`)
- `test_math_utils.py` - Math helper functions (quaternions, etc.)
- `test_scene_builder.py` - Scene construction utilities

#### Types (`tests/unit/types/`)
- `test_physics_types.py` - Physics data types and conversions

### 2. Integration Tests (`tests/integration/`)
Tests for component interactions and full pipeline.

- `test_full_pipeline.py` - Complete XPBD step execution
- `test_collision_pipeline.py` - Broadphase + narrowphase integration

### 3. Scaffolding Tests (`tests/scaffolding/`)
Basic tests to ensure the engine can be created and called.

- `test_engine_creation.py` - Engine instantiation and initialization
- `test_basic_operations.py` - Basic engine operations (step, get/set state)

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test category
pytest tests/unit/
pytest tests/integration/
pytest tests/scaffolding/

# Run tests for specific XPBD phase
pytest tests/unit/xpbd/test_broadphase.py

# Run with verbose output
pytest tests/ -v

# Run specific test function
pytest tests/unit/xpbd/test_broadphase.py::test_compute_cell_ids
```

## Test Fixtures

Common test fixtures are defined in `conftest.py`:
- `two_body_scene` - Simple scene with two spheres
- `multi_body_stack_scene` - Stack of boxes
- `random_bodies_scene` - Random configuration of bodies

## Adding New Tests

When adding new tests:

1. **Unit tests** - Test individual functions/methods in isolation
   - Place in appropriate subdirectory under `tests/unit/`
   - Focus on edge cases and error conditions
   - Mock dependencies when necessary

2. **Integration tests** - Test interactions between components
   - Place in `tests/integration/`
   - Use realistic scenarios
   - Verify end-to-end behavior

3. **Scaffolding tests** - Ensure basic functionality
   - Place in `tests/scaffolding/`
   - Should be fast and simple
   - Focus on "does it crash?" level testing

## Test Naming Conventions

- Test files: `test_<component>.py`
- Test functions: `test_<feature>_<scenario>()`
- Test classes: `Test<Component>` (when grouping related tests)

## Performance Considerations

- Unit tests should be fast (< 0.1s each)
- Integration tests can be slower but should complete in < 5s
- Use smaller scenes/fewer steps for tests
- Consider using `@pytest.mark.slow` for expensive tests