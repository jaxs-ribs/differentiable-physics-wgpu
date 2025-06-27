# TinyGrad Physics Engine

A pure TinyGrad differentiable rigid-body physics engine with JIT compilation support.

## Installation

```bash
# Install dependencies
pip install tinygrad numpy

# Clone and navigate to the project
cd physics_core/
```

## Quick Start

```bash
# Run the physics simulation (creates artifacts/oracle_dump.npy)
python3 -m physics.main --steps 200

# Run all tests (7 comprehensive tests)
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
├── physics/                  # Core physics modules
│   ├── engine.py            # Main physics engine with JIT support
│   ├── broadphase_tensor.py # Differentiable AABB collision detection
│   ├── narrowphase.py       # Sphere-sphere and sphere-box collisions
│   ├── solver.py            # Impulse-based collision resolution
│   ├── integration.py       # Semi-implicit Euler integration
│   ├── math_utils.py        # Quaternion and matrix operations
│   ├── types.py             # Body schema and shape types
│   └── main.py              # Entry point for simulations
├── tests/                   # Comprehensive test suite
│   ├── run_ci.py           # Main CI runner (7 tests)
│   ├── unit/               # Component tests
│   ├── integration/        # System tests
│   ├── benchmarks/         # Performance tests
│   └── debugging/          # Diagnostic tools
├── artifacts/               # Simulation outputs (.npy files)
├── README.md               # This file
├── AGENTS.md               # Project vision and phases
└── BUG_FIXES.md            # Documented issues and solutions
```

## Usage

### Running Simulations

```bash
# Basic simulation with default parameters
python3 -m physics.main

# Custom simulation parameters
python3 -m physics.main --steps 500 --dt 0.01 --output artifacts/my_simulation.npy

# N-step JIT mode (default - faster for long simulations)
python3 -m physics.main --mode nstep --steps 1000

# Single-step mode (for debugging)
python3 -m physics.main --mode single --steps 100

# Save all intermediate frames (single-step mode only)
python3 -m physics.main --mode single --steps 100 --save-intermediate
```

### Using as a Library

```python
import numpy as np
from physics.engine import TensorPhysicsEngine
from physics.types import create_body_array, ShapeType

# Create bodies
bodies = []

# Add a static ground box
bodies.append(create_body_array(
    position=np.array([0, -2, 0]),
    mass=1e8,  # Very large mass = effectively static
    shape_type=ShapeType.BOX,
    shape_params=np.array([10, 0.5, 10])  # 20x1x20 ground
))

# Add a falling sphere
bodies.append(create_body_array(
    position=np.array([0, 5, 0]),
    velocity=np.array([0, 0, 0]),
    mass=1.0,
    shape_type=ShapeType.SPHERE,
    shape_params=np.array([0.5, 0, 0])  # radius=0.5
))

# Initialize engine
engine = TensorPhysicsEngine(np.stack(bodies), dt=0.016)

# Run simulation
engine.run_simulation(num_steps=100)  # N-step JIT (fast)
# or
engine.step()  # Single step (for debugging)

# Get results
final_state = engine.get_state()  # Returns numpy array
print(f"Sphere final position: {final_state[1, 0:3]}")
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

### Supported Features

- **Shapes**: Spheres and boxes (more coming in future phases)
- **Collisions**: Sphere-sphere and sphere-box
- **Constraints**: Contact constraints with restitution
- **Forces**: Gravity (extensible to other forces)
- **Output**: NumPy arrays with full state (position, velocity, orientation, etc.)

### Output Format

The simulation outputs NumPy arrays with shape `(num_frames, num_bodies, 27)` where:
- `num_frames`: 2 (initial and final) or `steps+1` (with --save-intermediate)
- `num_bodies`: Number of rigid bodies in the scene
- `27`: Properties per body (see `physics/types.py:BodySchema` for details)

Key properties include:
- `[0:3]`: Position (x, y, z)
- `[3:6]`: Velocity (vx, vy, vz)
- `[6:10]`: Orientation quaternion (w, x, y, z)
- `[10:13]`: Angular velocity
- `[13]`: Inverse mass (0 for static bodies)
- `[14:17]`: Inverse inertia tensor diagonal
- `[17]`: Shape type (0=sphere, 1=box)
- `[18:21]`: Shape parameters (radius for sphere, half-extents for box)

### Known Issues & Workarounds

- TinyGrad's `Tensor.where()` has a bug with NaN values (converts to 1.0)
  - Workaround: Use multiplication-based masking instead
- Empty contact arrays require special handling for JIT compatibility
  - Workaround: Create dummy data with zero mask

## Performance

On a typical GPU:
- 100 steps with 10 bodies: ~170 steps/second
- N-step JIT compilation: ~1.2s overhead, then very fast execution
- Scaling: O(n²) for n bodies due to all-pairs collision detection

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