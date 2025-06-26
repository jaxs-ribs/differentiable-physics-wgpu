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
cargo test --lib                                    # Rust unit tests
python3 tests/test_integrator.py                    # Python reference tests
python3 tests/test_sdf.py
python3 tests/test_sdf_quick.py                    # Quick SDF validation
python3 tests/test_energy.py                        # Energy conservation
python3 tests/test_implementations.py               # Test new features (broadphase + dynamics)
cargo run --bin test_sdf                           # GPU integration tests
cargo run --bin test_contact_solver
cargo run --bin test_broadphase_grid

# Run benchmarks
cargo run --release --bin benchmark                # Performance benchmarks
cargo run --release --bin benchmark_full           # Full benchmark suite

# Run demos
cargo run --features viz --bin demo_viz             # 3D wireframe visualization
cargo run --bin demo_simple                         # Console output
cargo run --bin demo_ascii                          # ASCII visualization

# Get help
cargo run --bin physics_core                        # List all available commands
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

## License

This project is licensed under the MIT License.