# Differentiable Physics Engine

A WebGPU-first, batch-differentiable rigid-body simulator designed for high-throughput reinforcement learning and evolutionary robotics. This engine runs entirely on the GPU, enabling massive parallelism for simulations involving tens of thousands of objects.

This project is currently in **Phase 1: Complete**, resulting in a high-performance, stand-alone Rust and WGSL physics engine.

## Core Features

- **Massively Parallel Architecture:** Built from the ground up for the GPU using WebGPU (via `wgpu`) and WGSL shaders.
- **High Performance:** Achieves over **600 million** body-steps per second on consumer hardware. See [Performance](#performance) for details.
- **Core Physics Primitives:**
    - Semi-implicit Euler integration
    - Analytic Signed Distance Function (SDF) collision detection (Sphere, Capsule, Box)
    - Penalty-based contact resolution
    - Uniform grid broad-phase for efficient collision pair pruning
- **Comprehensive Test Suite:** Behavior is locked in against a Python/NumPy golden reference, with property-based fuzz testing and stability stress tests.
- **Minimal Wireframe Visualization:** An optional, lightweight `winit`-based viewer for debugging simulation state directly from GPU memory.

## Performance

The engine is memory-bandwidth limited and scales linearly with the number of bodies. Benchmarks were run on a consumer-grade GPU.

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

The project is controlled via a unified command-line interface.

```bash
# Clone the repository
git clone <repository-url>
cd physicsengine/physics_core

# Run the benchmark suite
./physics_core benchmark

# Run the wireframe visualization demo
./physics_core demo viz

# Run the full test suite (Rust and Python)
./run_all_tests.sh
```

## Project Philosophy

1.  **Correctness, Then Speed:** Behavior is locked with an exhaustive test suite before any micro-optimization.
2.  **No Hidden Host Hops:** The core simulation loop runs entirely on the GPU, with zero data transfer back to the CPU until explicitly needed.
3.  **Single Source of Truth:** A single, GPU-resident buffer holds the state of all bodies, ensuring identical memory layouts between Rust and WGSL.

## Roadmap

This project is planned in several phases, moving from a stand-alone engine to a fully differentiable system integrated with machine learning frameworks.

-   **Phase 1 (Complete):** Build and validate the stand-alone Rust + WGSL engine.
-   **Phase 2 (Next):** Integrate with `tinygrad` by exposing WGSL kernels as custom operations, enabling a full autograd pipeline.
-   **Phase 3:** Implement analytic Jacobians for a differentiable backward pass and connect to world models like DreamerV3 for novel research applications.
-   **Future Work:** Explore telemetry dashboards, LLM-driven simulation control, soft-body physics, and differentiable fluids.

## Development

To contribute, ensure you can run the full test suite. It is the primary guardrail for correctness.

```bash
# Run all Rust and Python tests
./run_all_tests.sh
```

## License

This project is licensed under the MIT License.