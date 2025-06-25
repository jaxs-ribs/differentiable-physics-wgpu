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

This project follows an ambitious, phased plan to evolve from a physics engine into a platform for cutting-edge AI research.

-   **Phase 1: Stand-Alone Engine (Complete)**
    -   **Status:** A high-performance, stand-alone Rust + WGSL simulator.
    -   **Outcome:** Validated with an exhaustive test suite and benchmarked at over 600M body-steps/s.

-   **Phase 2: Tinygrad Integration & Autograd (Next)**
    -   **Goal:** Achieve a full, end-to-end differentiable pipeline.
    -   **Key Steps:**
        -   Expose raw WGSL kernels as custom operations within `tinygrad`.
        -   Implement a Python shim for dispatching physics steps as tensor operations.
        -   Validate the gradient flow with a simple reinforcement learning task.

-   **Phase 3: Differentiable Evolution & World Models**
    -   **Goal:** Enable novel morphology and policy co-evolution.
    -   **Key Steps:**
        -   Implement an analytic backward pass (Jacobians) in WGSL for high-performance gradient calculation.
        -   Integrate with pre-trained world models (e.g., DreamerV3) by using latent vectors (`z_t`) as the observation space.
        -   Build a Quality-Diversity (QD) outer loop to search for novel morphologies driven by task reward and latent-space novelty.

-   **Phase 4: Live Telemetry & Interactive Control**
    -   **Goal:** Create a "live control center" for monitoring and interacting with simulations.
    -   **Key Steps:**
        -   Develop a WebSocket server to stream real-time telemetry from the simulation.
        -   Build a minimal web-based dashboard for live visualization.
        -   Prototype interactive control, allowing external agents (including LLMs) to pause, mutate, and resume simulations.

-   **Phase 5: Flagship Showcase & Publication**
    -   **Goal:** Deliver a shippable, public demonstration and publish the results.
    -   **Key Steps:**
        -   Create a flagship demo showcasing emergent "ideal body" evolution.
        -   Package the project for one-click deployment (e.g., Docker).
        -   Publish an arXiv paper detailing the methods, performance, and findings.

## Development

To contribute, ensure you can run the full test suite. It is the primary guardrail for correctness.

```bash
# Run all Rust and Python tests
./run_all_tests.sh
```

## License

This project is licensed under the MIT License.