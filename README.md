# Differentiable Physics Engine

A rigid-body physics engine designed for GPU execution via WebGPU (`wgpu`). The primary goal is to serve as a high-performance, differentiable custom operator within machine learning frameworks like `tinygrad`.

This project is currently at the end of **Phase 1**, resulting in a functional, stand-alone Rust/WGSL physics engine.

## Core Components

- **GPU Compute Pipeline:** Physics steps execute entirely on the GPU using WGSL compute shaders.
- **Physics Primitives:**
- Semi-implicit Euler integration
- Signed Distance Function (SDF) collision detection for sphere, capsule, and box shapes.
- A penalty-based contact model for resolving collisions.
- A uniform grid broad-phase to minimize collision checks.
- **Testing:** The GPU implementation is validated against a Python/NumPy reference implementation. The test suite includes property-based fuzzing for SDFs and multi-body stability tests.
- **Debug Viewer:** An optional `winit`-based wireframe viewer renders body AABBs directly from GPU memory.

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
python3 tests/test_broadphase.py
python3 tests/test_energy.py
python3 tests/test_sdf_fuzz.py
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

## Project Technical Goals

1.  **Correctness via Testing:** The behavior of the GPU kernels is verified against a CPU-based NumPy reference implementation.
2.  **GPU-Centric Execution:** The core simulation loop avoids CPU-GPU data transfer. State remains on the GPU until explicitly retrieved.
3.  **Consistent Memory Layout:** The `Body` data structure is identical between Rust (`#[repr(C)]`) and WGSL to allow for zero-copy buffer mapping.

## Technical Roadmap

-   **Phase 1 (Complete): Stand-Alone Engine**
    -   **Outcome:** A functional Rust/WGSL physics simulator with a comprehensive test suite and validated performance.

-   **Phase 2 (Current): `tinygrad` Integration**
    -   **Objective:** Expose the WGSL physics kernels as custom operations in `tinygrad`.
    -   **Tasks:**
        -   Patch `tinygrad`'s WebGPU runtime to accept raw WGSL code.
        -   Write a Python wrapper to dispatch the physics step as a tensor operation.
        -   Verify the integration by training a simple policy for a control task (e.g., cart-pole).

-   **Phase 3: Differentiable Backward Pass**
    -   **Objective:** Implement an efficient, analytic backward pass for the physics simulation.
    -   **Tasks:**
        -   Write WGSL kernels for the analytic Jacobians of the integrator and contact solver.
        -   Validate the gradients against finite-difference approximations.
        -   Use the backward pass in a Quality-Diversity (QD) optimization loop to find novel solutions to physical tasks.

-   **Phase 4: Telemetry and Interaction**
    -   **Objective:** Add capabilities for live monitoring and external control of the simulation.
    -   **Tasks:**
        -   Implement a WebSocket server to stream simulation state as JSON data.
        -   Build a minimal web client to display the telemetry.

-   **Phase 5: Packaging and Documentation**
    -   **Objective:** Produce a distributable artifact and a technical summary.
    -   **Tasks:**
        -   Package the simulation and UI into a container for simplified deployment.
        -   Write a technical report detailing the methods, performance, and results.

## Development

All changes should be verified against the existing test suite.

```bash
# Run comprehensive test suite
cargo test --lib                                    # Rust unit tests
python3 tests/test_integrator.py                    # Python reference tests  
python3 tests/test_sdf.py
python3 tests/test_broadphase.py
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