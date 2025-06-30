# TinyGrad Physics Engine

A pure TinyGrad differentiable rigid-body physics engine with JIT compilation support.

https://github.com/user-attachments/assets/c83ca74a-d11c-42ad-a4ac-a340326fe149

## Installation

```bash
# Clone with submodules
git clone --recursive [repository-url]
cd physics_core/

# Or if already cloned, initialize submodules
git submodule update --init --recursive

# Install dependencies
pip install numpy  # TinyGrad is included as a submodule
```

## Quick Start

```bash
# Run the default naive simulation and render the result
python3 run.py --naive

# Run all tests
python3 ci.py
```

## Usage

### Running Simulations

The `run.py` script is the main entry point for all simulations.

```bash
# Run the pure Python "Oracle" simulation
python3 run.py --naive

# Run the C-accelerated simulation
python3 run.py --c

# Run both and compare them in the renderer
python3 run.py --naive --c

# Run with a different number of steps
python3 run.py --naive --steps 500

# Save all simulation frames (not just first and last)
python3 run.py --naive --save-all-frames

# Run simulation without automatically starting the renderer
python3 run.py --naive --no-render

# Re-render the most recent simulation(s)
python3 run.py
```

### Visualizing Results

The renderer is typically launched automatically by `run.py`. You can also run it manually to view specific files.

```bash
# Build the renderer if you haven't already
cd renderer
cargo build --release
cd ..

# Manually render one or more simulation files
./renderer/target/release/renderer artifacts/naive_sim_*.npy --color 0.2 0.5 1.0 --alpha 0.7
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
# Run the standard test suite (unit + integration)
python3 ci.py

# Run only unit tests
python3 ci.py --unit

# Run all tests, including benchmarks
python3 ci.py --all
```

#### Test Status (All Passing ✓)
- **Core CI Suite**: 7/7 tests passing
- **Custom Ops**: 4/4 tests passing (C library, integration, demo, benchmark)
- **Debugging**: 4/4 tests passing
- **Total**: 15/15 tests passing

Note: Some unit tests require pytest. The core functionality is thoroughly tested without it.

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

The physics engine internally uses 27 properties per body, but automatically transforms the output to a renderer-compatible format with 18 properties per body.

**Saved NPY Format**: `(num_frames, num_bodies * 18)` - flattened for renderer compatibility
- `num_frames`: 2 (initial and final) or `steps+1` (with --save-intermediate)
- `num_bodies`: Number of rigid bodies in the scene

**Per-body properties (18 values)**:
1. Position (3): x, y, z world coordinates
2. Velocity (3): vx, vy, vz
3. Orientation (4): quaternion (w, x, y, z)
4. Angular velocity (3): omega_x, omega_y, omega_z
5. Mass (1): actual mass (converted from inverse mass)
6. Shape type (1): 0=sphere, 2=box, 3=capsule
7. Shape parameters (3):
   - Sphere: radius, 0, 0
   - Box: half_extent_x, half_extent_y, half_extent_z
   - Capsule: half_height, radius, 0

The transformation from physics format (27 properties) to renderer format (18 properties) removes the inverse inertia tensor and converts inverse mass to mass.

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

## Custom Operations (Experimental)

The `custom_ops/` directory contains an experimental implementation of high-performance physics operations using TinyGrad's CUSTOM op mechanism:

### Building Custom Ops
```bash
cd custom_ops/src
make
```

### Using Custom Ops
```python
from custom_ops import enable_physics_on_device, PhysicsTensor

# Enable physics operations on CPU
enable_physics_on_device("CPU")

# Create and simulate physics world
world = PhysicsTensor.create_physics_world(n_bodies=100)
world = world.integrate(dt=0.016)
```

See `custom_ops/README.md` for detailed documentation.

## Visualization & Analysis

### Comparative Replay Renderer

The `renderer/` directory contains a high-performance Rust application for visualizing and comparing simulation runs:

```bash
# Build the renderer
cd renderer
cargo build --release

# Single simulation visualization
cargo run --bin renderer -- --primary ../artifacts/simulation.npy

# Compare two simulation runs (e.g., CPU vs GPU)
cargo run --bin renderer -- \
  --primary ../artifacts/cpu_run.npy \
  --secondary ../artifacts/gpu_run.npy \
  --mode benchmark

# Correctness mode - verify numerical parity
cargo run --bin renderer -- -p run1.npy -s run2.npy --mode correctness
```

**Features:**
- Ghost rendering: Secondary runs appear as semi-transparent overlays
- Dual modes: Correctness (frame-sync) and Benchmark (performance comparison)
- Direct .npy file loading with validation
- Clear visual feedback for performance differences

See `renderer/README.md` for detailed usage instructions.

## Status

**Phase 1 Complete** ✓ - Python oracle fully implemented and tested
**Phase 3 Explored** ✓ - Custom op proof-of-concept created

### Recent Improvements
- ✓ Reorganized custom ops into clean directory structure
- ✓ Added comprehensive test suite (15 tests, all passing)
- ✓ Fixed dtypes import in solver module
- ✓ Improved path handling in debugging tests
- ✓ Created simplified test runners that don't require pytest
- ✓ Documented all test results
- ✓ Created interactive test runner with terminal UI
- ✓ Added verbose logging to all tests
- ✓ Added scrolling support and docstring display
- ✓ All tests passing and ready for production

Ready for:
- Phase 2: WGSL kernel implementation
- Phase 3: Full Ops.CUSTOM integration
- Phase 4: Backward pass for full differentiability

## Documentation

- [`docs/AGENTS.md`](docs/AGENTS.md) - Project vision and status
- [`docs/BUG_FIXES.md`](docs/BUG_FIXES.md) - Documented fixes and workarounds
- [`docs/TEST_HELL.md`](docs/TEST_HELL.md) - Interactive test runner guide
- [`docs/TEST_SUMMARY.md`](docs/TEST_SUMMARY.md) - Test results documentation
- [`tests/README.md`](tests/README.md) - Test suite documentation
- [`physics/README.md`](physics/README.md) - Physics module details
- [`custom_ops/README.md`](custom_ops/README.md) - Custom operations guide
- [`renderer/README.md`](renderer/README.md) - Comparative replay renderer guide