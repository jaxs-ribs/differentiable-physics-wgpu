# Physics Core - WebGPU Rigid Body Physics Engine

A high-performance, GPU-accelerated rigid body physics engine built with Rust and WebGPU. Designed for massively parallel simulation of thousands of bodies entirely on the GPU.

## 🚀 Performance

TBD

## 🏗️ Architecture Overview

### Core Components

```
physics_core/
├── src/
│   ├── lib.rs                 # Library root
│   ├── body.rs               # Rigid body data structure (112 bytes, GPU-aligned)
│   ├── gpu.rs                # WebGPU context and initialization
│   ├── physics/              # Physics simulation modules
│   │   ├── mod.rs           # Module exports
│   │   ├── buffer_manager.rs # GPU buffer management
│   │   ├── compute_executor.rs # Compute shader execution
│   │   ├── pipeline_builder.rs # Pipeline construction
│   │   └── shader_loader.rs  # WGSL shader loading
│   ├── shaders/              # WGSL compute shaders
│   │   ├── physics_step.wgsl # Main physics integration
│   │   ├── broadphase.wgsl   # Collision detection broadphase
│   │   ├── narrowphase.wgsl  # Detailed collision detection
│   │   └── contact_solver.wgsl # Collision resolution
│   ├── test_utils/           # Testing utilities
│   │   ├── gpu_helpers.rs    # GPU testing helpers
│   │   └── harness.rs        # Test harness
│   └── viz/                  # Visualization (optional feature)
│       ├── camera.rs         # 3D camera controls
│       ├── dual_renderer.rs  # Debug visualization renderer
│       ├── uniforms.rs       # GPU uniform buffers
│       ├── window.rs         # Window management
│       └── wireframe_geometry.rs # Wireframe generation
```

### Key Design Principles

1. **GPU-First Architecture**: All physics computations happen on GPU via compute shaders
2. **Structure of Arrays (SoA)**: Optimized memory layout for GPU parallelism
3. **Lock-Step Parity**: GPU implementation validated against Python/NumPy reference
4. **Zero-Copy Execution**: Data stays on GPU throughout simulation

## 🛠️ Quick Start

### Prerequisites

- Rust toolchain (via [rustup](https://rustup.rs/))
- Python 3.x (for reference tests)
- WebGPU-capable GPU

### Basic Usage

```bash
# Clone and build
git clone <repository>
cd physics_core
cargo build --release

# Run benchmarks
cargo run --release --bin benchmark -- 10000

# Run visual demo (requires 'viz' feature)
cargo run --release --features viz --bin demo_viz

# Run physics animation player
cargo run --release --features viz --bin debug_viz -- --oracle tests/oracle_dump.npy
```

### Running Tests

```bash
# Run all Rust tests
cargo test

# Run Python reference tests
cd tests
python3 test_sdf_quick.py      # Quick validation
python3 test_energy.py          # Energy conservation
python3 test_implementations.py # Implementation tests

# Create and visualize a physics trace
python3 create_test_dump.py
cd ..
cargo run --features viz --bin debug_viz -- --oracle tests/oracle_dump.npy
```

## 📦 Features

### Core Physics
- Semi-implicit Euler integration
- Sphere, Box, and Capsule collision shapes
- Signed Distance Function (SDF) based collision detection
- Impulse-based collision resolution
- Broadphase collision culling

### Visualization (`viz` feature)
- Real-time 3D wireframe rendering
- Physics state inspection and debugging
- Animation playback from trace files
- Camera controls (mouse drag to rotate, scroll to zoom)

### Debug Tools
- `debug_viz`: Animated physics trace player
- `plot_energy.py`: Energy conservation analysis
- State comparison tools for GPU/CPU parity testing

## 🔧 Building

```bash
# Basic build
cargo build --release

# With visualization
cargo build --release --features viz

# Run specific binary
cargo run --release --features viz --bin <binary_name>
```

## 📊 Benchmarking

```bash
# Quick benchmark (10k bodies)
cargo run --release --bin benchmark

# Full benchmark suite
cargo run --release --bin benchmark_full

# Custom body count
cargo run --release --bin benchmark -- 20000
```

## 🧪 Testing Philosophy

The project maintains correctness through extensive testing against a Python/NumPy reference implementation:

1. **Reference Implementation** (`tests/reference.py`): Golden standard for physics behavior
2. **Property-Based Testing**: Validates mathematical properties (symmetry, conservation laws)
3. **GPU-CPU Parity**: Ensures GPU results match CPU reference exactly

## 📝 Project Status

This is an active research project exploring GPU-accelerated physics simulation. The codebase is designed for experimentation and may undergo significant changes.

## 🔗 Related Documentation

- [AGENTS.md](AGENTS.md) - Detailed development history and roadmap
- [Overview files](src/bin/overview.md) - Descriptions of all binaries and tests

## 📄 License

MIT License - See LICENSE file for details.