# Physics Renderer

A high-performance comparative replay renderer for physics simulations. This standalone Rust application visualizes and compares multiple simulation runs simultaneously, rendering secondary runs as semi-transparent "ghosts" for direct visual comparison.

## Overview

The renderer serves two primary purposes:
- **Benchmarking**: Visually compare execution performance between different simulation backends
- **Validation**: Verify numerical parity and correctness between simulation runs

## Features

- **Dual-mode operation**: Correctness mode for frame-synchronized comparison, Benchmark mode for performance visualization
- **Ghost rendering**: Secondary simulation runs appear as semi-transparent overlays
- **Efficient data loading**: Direct loading of NumPy (.npy) trajectory files
- **Command-line interface**: Simple and intuitive CLI for all operations

## Installation

### Prerequisites
- Rust 1.70 or later
- Cargo package manager

### Building
```bash
cd renderer
cargo build --release
```

## Usage

### Basic Usage

```bash
# Single simulation visualization
cargo run --bin renderer -- --primary ../artifacts/simulation.npy

# Dual simulation comparison
cargo run --bin renderer -- --primary ../artifacts/cpu_run.npy --secondary ../artifacts/gpu_run.npy

# Benchmark mode with verbose logging
cargo run --bin renderer -- -p ../artifacts/cpu_run.npy -s ../artifacts/gpu_run.npy -m benchmark -v
```

### Command-Line Options

- `-p, --primary <FILE>`: Primary simulation .npy file (required)
- `-s, --secondary <FILE>`: Secondary simulation .npy file for comparison (optional)
- `-m, --mode <MODE>`: Rendering mode: `correctness` (default) or `benchmark`
- `-v, --verbose`: Enable verbose logging

### Rendering Modes

#### Correctness Mode (Default)
All simulation replays are perfectly synchronized frame-by-frame. The renderer continues until the longest simulation completes. This mode verifies that different backends produce identical results.

```bash
cargo run --bin renderer -- -p run1.npy -s run2.npy --mode correctness
```

#### Benchmark Mode
The renderer identifies the fastest run and caps playback duration at this time. Slower runs appear frozen mid-animation when time expires, visually demonstrating performance differences.

```bash
cargo run --bin renderer -- -p run1.npy -s run2.npy --mode benchmark
```

## Data Format

The renderer expects .npy files with the following structure:
- **Shape**: `(num_frames, num_bodies, 27)`
- **Type**: `float32`
- **Body data layout** (27 floats per body):
  - Position (3 floats)
  - Velocity (3 floats)
  - Orientation quaternion (4 floats)
  - Angular velocity (3 floats)
  - Force (3 floats)
  - Torque (3 floats)
  - Mass properties (4 floats)
  - Shape properties (4 floats)

## Architecture

### Core Components

- **`body.rs`**: Defines the 108-byte Body struct matching the physics engine format
- **`loader.rs`**: Handles .npy file loading and trajectory data management
- **`main.rs`**: CLI interface and application entry point

### Project Structure
```
renderer/
├── Cargo.toml
├── README.md
└── src/
    ├── body.rs      # Body data structure (27 floats)
    ├── loader.rs    # .npy file loading
    └── main.rs      # Application entry point
```

## Development

### Running Tests
```bash
cargo test
```

### Running Specific Tests
```bash
# Test body struct size
cargo test body::tests::test_body_size

# Test loader functionality
cargo test loader::tests
```

## Current Status

✅ **Implemented**:
- Body struct with exact 108-byte layout
- .npy file loading with validation
- Command-line interface
- Dual-mode logic (correctness/benchmark)
- Data integrity verification

🚧 **In Progress**:
- Actual rendering pipeline
- Ghost transparency rendering
- Frame synchronization
- On-screen HUD

## Future Enhancements

- WebGPU rendering pipeline
- Interactive camera controls
- Real-time performance metrics
- Video export capabilities
- Web platform support

## Contributing

When adding new features:
1. Ensure all tests pass: `cargo test`
2. Follow Rust naming conventions
3. Add unit tests for new functionality
4. Update this README as needed

## License

See the main project license file.