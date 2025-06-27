# TinyGrad Physics Engine

A pure TinyGrad differentiable rigid-body physics engine with JIT compilation support.

## Quick Start

```bash
# Run the physics simulation
python3 -m physics.main --steps 200 --output artifacts/simulation.npy

# Run all tests
python3 tests/run_ci.py
```

## Key Features

- **Pure Tensor Operations** - No NumPy in core physics modules
- **JIT Compilable** - Entire N-step simulations run as single GPU kernels
- **Differentiable** - Gradients flow through collision detection and resolution
- **Multi-Backend** - Runs on any TinyGrad backend (WebGPU, Metal, CUDA, etc.)

## Project Structure

```
physics_core/
├── physics/              # Core physics modules
│   ├── engine.py        # Main physics engine with JIT support
│   ├── broadphase_tensor.py  # AABB collision detection
│   ├── narrowphase.py   # Sphere-sphere and sphere-box collisions
│   ├── solver.py        # Impulse-based collision resolution
│   ├── integration.py   # Semi-implicit Euler integration
│   ├── math_utils.py    # Quaternion and matrix operations
│   └── main.py          # Entry point for simulations
├── tests/               # Comprehensive test suite
│   ├── run_ci.py       # Run all CI tests (7 tests)
│   └── ...             # Unit, integration, and benchmarks
└── artifacts/           # Output directory for simulations
```

## Usage

### Running Simulations

```bash
# Basic simulation with default parameters
python3 -m physics.main

# Custom simulation parameters
python3 -m physics.main --steps 500 --dt 0.01 --output my_simulation.npy

# N-step JIT mode (faster for long simulations)
python3 -m physics.main --mode nstep --steps 1000

# Single-step mode (for debugging)
python3 -m physics.main --mode single --steps 100
```

### Using as a Library

```python
import numpy as np
from physics.engine import TensorPhysicsEngine
from physics.types import create_body_array, ShapeType

# Create bodies
bodies = []
bodies.append(create_body_array(
    position=np.array([0, 5, 0]),
    mass=1.0,
    shape_type=ShapeType.SPHERE,
    shape_params=np.array([0.5, 0, 0])  # radius=0.5
))

# Initialize engine
engine = TensorPhysicsEngine(np.stack(bodies), dt=0.016)

# Run simulation
engine.run_simulation(num_steps=100)  # N-step JIT
# or
engine.step()  # Single step

# Get results
final_state = engine.get_state()
```

## Development

### Testing

```bash
# Quick CI test suite (recommended)
python3 tests/run_ci.py

# Run specific test categories
python3 -m pytest tests/unit/
python3 -m pytest tests/integration/
python3 -m pytest tests/benchmarks/
```

### Key Implementation Details

1. **NumPy-Free Core** - All physics computations use TinyGrad tensors
2. **JIT Compilation** - Uses `@TinyJit` for automatic kernel fusion
3. **Collision Detection** - Differentiable AABB broadphase, sphere/box narrowphase
4. **Solver** - Impulse-based with Baumgarte stabilization
5. **Integration** - Semi-implicit Euler with quaternion normalization

### Known Issues & Workarounds

- TinyGrad's `Tensor.where()` has a bug with NaN values (converts to 1.0)
  - Workaround: Use multiplication-based masking instead
- Empty contact arrays require special handling for JIT compatibility
  - Workaround: Create dummy data with zero mask

## Performance

On a typical GPU:
- 100 steps with 10 bodies: ~170 steps/second
- N-step JIT compilation: ~1.2s overhead, then very fast execution

## Status

**Phase 1 Complete** ✓ - Python oracle fully implemented and tested

Ready for:
- Phase 2: WGSL kernel implementation
- Phase 3: Ops.CUSTOM integration
- Phase 4: Backward pass for full differentiability

## Documentation

- `AGENTS.md` - Project vision and status
- `BUG_FIXES.md` - Documented fixes and workarounds
- `tests/README.md` - Test suite documentation
- `physics/README.md` - Physics module details