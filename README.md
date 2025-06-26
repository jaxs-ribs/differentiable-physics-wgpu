# WebGPU Physics Engine

A high-performance rigid-body physics engine designed for GPU execution via WebGPU. Features semi-implicit Euler integration, SDF-based collision detection, and support for spheres, capsules, and boxes.

## Core Features

- **GPU Compute Pipeline:** Physics steps execute entirely on the GPU using WGSL compute shaders
- **Semi-implicit Euler integration** for stable motion
- **SDF collision detection** supporting spheres, capsules, and boxes
- **Penalty-based contact resolution** 
- **Uniform grid broadphase** for efficient collision culling
- **Comprehensive test suite** with Python reference implementation
- **3D wireframe visualization** (optional)

## Performance

Benchmarks run on a single consumer-grade GPU, measuring throughput in simulated bodies multiplied by simulation steps per second.

| Body Count | Throughput (body-steps/s) |
| :--- | :--- |
| 1,000      | 28,000,000+               |
| 10,000     | 302,000,000+              |
| 20,000     | **630,000,000+**              |

## Quick Start

### Prerequisites

- Rust toolchain (`rustup`)
- Python 3.x

### Usage

```bash
# Clone the repository
git clone <repository-url>
cd physicsengine/physics_core

# Run all tests
./pc-test                                          # Run all tests (recommended)
# Or run individual test suites:
cargo test --lib                                   # Rust unit tests
cd tests
python3 test_sdf_quick.py                          # Quick SDF validation
python3 test_energy.py                             # Energy conservation
python3 test_broadphase_sap.py                     # Sweep and Prune broadphase
python3 test_dynamics.py                           # Rotational dynamics
./run_all_tests.sh                                 # Run all Python tests

# Run benchmarks
./pc-bench                                         # Quick benchmark (default: 10k bodies)
./pc-bench 20000                                   # Benchmark with 20k bodies
cargo run --release --bin benchmark_full           # Full benchmark suite

# Run demos
./pc-demo viz                                      # 3D wireframe visualization
./pc-demo simple                                   # Console output
./pc-demo ascii                                    # ASCII visualization

# Debug tools
cargo run --features viz --bin debug_viz -- \      # Visual diff debugger
  --oracle tests/failures/test_name/cpu_state.npy \
  --gpu tests/failures/test_name/gpu_state.npy

python3 tests/plot_energy.py                       # Plot energy conservation
```

## Technical Details

- **Zero-copy GPU execution:** Simulation state remains on GPU throughout execution
- **Consistent memory layout:** `Body` structure matches exactly between Rust and WGSL
- **Validated correctness:** GPU implementation tested against Python/NumPy reference

## Development

All changes should be verified against the existing test suite.

```bash
# Run comprehensive test suite
cargo test --lib                                    # Rust unit tests
python3 tests/test_integrator.py                    # Python reference tests  
python3 tests/test_sdf.py
python3 tests/test_sdf_quick.py                     # Quick SDF validation
python3 tests/test_implementations.py                # Test broadphase + dynamics
python3 tests/test_contact_solver.py                # Additional Python tests
python3 tests/test_contact_solver_gpu.py
python3 tests/test_sdf_gpu.py
python3 tests/test_energy.py                        # Comprehensive tests
python3 tests/test_sdf_fuzz.py
python3 tests/test_stability_stress.py
cargo run --bin test_sdf                           # GPU integration tests
cargo run --bin test_contact_solver
cargo run --bin test_broadphase_grid
cargo run --bin test_runner

# Check code quality
cargo check                                         # Without viz features
cargo check --features viz                          # With viz features
```

## Debug and Analysis Tools

### Smart Test Failure System

When GPU-CPU parity tests fail, the system automatically saves both states:

```bash
# Run tests with automatic failure dumps
pytest tests/test_gpu_cpu_parity.py

# Files created on failure:
tests/failures/<test_name>/
├── cpu_state.npy      # CPU physics state
├── gpu_state.npy      # GPU physics state
└── debug_info.txt     # Test details and traceback

# Keep old failures (default cleans up before each test)
pytest tests/test_gpu_cpu_parity.py --keep-failures
```

### Visual Diff Debugger

Visualize divergence between CPU and GPU simulations:

```bash
# Launch the visual debugger
cargo run --features viz --bin debug_viz -- \
  --oracle tests/failures/test_name/cpu_state.npy \
  --gpu tests/failures/test_name/gpu_state.npy

# Controls:
# - B: Toggle AABB visualization
# - C: Toggle contact points
# - ESC: Exit

# Visual indicators:
# - Green/transparent: Oracle (CPU) state
# - Red/opaque: GPU state
```

### Energy Conservation Analysis

Track system energy over time to detect instabilities:

```bash
# Plot energy conservation
cd tests
python3 plot_energy.py --scene scenes/stack.json --steps 1000

# Outputs: energy_drift.png
# Shows total energy (kinetic + potential) over time
# Warns if drift exceeds 5%

# Custom scene
python3 plot_energy.py --scene my_scene.json --output my_energy.png
```

## License

This project is licensed under the MIT License.