# SDF Physics Renderer

A minimal SDF (Signed Distance Field) raymarching renderer for physics simulations, built with Rust and WebGPU. This renderer uses brute-force raymarching to precisely visualize spheres, boxes, and capsules with support for headless rendering and video recording.

## Features

- **SDF Raymarching**: Precise visualization using signed distance fields
- **Shape Support**: Spheres, boxes, and capsules with proper rotations
- **Headless Mode**: Render to PNG without a window for verification
- **Real-time 3D Visualization**: Interactive camera controls
- **Video Recording**: Export simulation playback as MP4 videos
- **GPU-accelerated**: Brute-force raymarching on the GPU via WebGPU
- **NPY Data Loading**: Support for physics simulation trajectory files

## Installation

### Prerequisites

- Rust 1.70 or later
- FFmpeg (for video recording functionality)
- GPU with WebGPU support

### Building

```bash
cd renderer
cargo build --release
```

## Usage

### Basic Visualization

```bash
# Visualize a single simulation
cargo run -- --oracle path/to/simulation.npy

# Run without data (shows test scene with sphere, box, capsule)
cargo run
```

### Headless Rendering

```bash
# Render a single frame to PNG without opening a window
cargo run -- --save-frame output.png
```

### Video Recording

```bash
# Record a 10-second video at 60 FPS
cargo run -- --oracle simulation.npy --record output.mp4 --duration 10 --fps 60

# Record comparison of two simulations
cargo run -- --oracle cpu.npy --gpu gpu.npy --record comparison.mp4 --duration 5
```

### Command-Line Options

- `--oracle <FILE>`: Path to the oracle (CPU) simulation file (.npy format)
- `--gpu <FILE>`: Path to the GPU simulation file for comparison (optional)
- `--record <FILE>`: Output video file path (enables recording mode)
- `--duration <SECONDS>`: Recording duration in seconds (default: 5)
- `--fps <NUMBER>`: Frames per second for recording (default: 30)
- `--save-frame <FILE>`: Render a single frame to PNG in headless mode

### Interactive Controls

- **Left Mouse Drag**: Rotate camera around the scene
- **Mouse Wheel**: Zoom in/out
- **Q/Escape**: Exit application

## Data Format

The renderer expects trajectory data in NPY format with the following structure:
- **Shape**: `(num_frames, num_bodies * 18)`
- **Type**: `float32`
- **Body data layout** (18 floats per body):
  - Position (3 floats)
  - Velocity (3 floats)
  - Orientation quaternion (4 floats)
  - Angular velocity (3 floats)
  - Mass (1 float)
  - Shape type (1 float: 0=sphere, 2=box)
  - Shape parameters (3 floats: radius for spheres, half-extents for boxes)

## Architecture

The renderer follows clean architecture principles with clear separation of concerns:

```
src/
├── main.rs              # Application entry point and event loop
├── lib.rs               # Core renderer implementation
├── body.rs              # Physics body data structures
├── camera.rs            # 3D camera system with orbit controls
├── gpu.rs               # GPU context and device management
├── loader.rs            # NPY trajectory file loading
├── mesh.rs              # Wireframe geometry generation
├── video.rs             # Video recording and frame export
└── shaders/
    └── wireframe.wgsl   # WGSL shaders for rendering
```

### Core Components

- **Renderer** (`lib.rs`): Main rendering engine managing dual scenes, GPU pipelines, and frame capture
- **Body** (`body.rs`): 112-byte aligned structure matching physics engine format
- **Camera** (`camera.rs`): Orbit camera with spherical coordinates and smooth controls
- **GpuContext** (`gpu.rs`): WebGPU device and queue management
- **TrajectoryLoader** (`loader.rs`): Efficient NPY file parsing and frame extraction
- **WireframeGeometry** (`mesh.rs`): Generates colored wireframe meshes for physics bodies
- **Video** (`video.rs`): FFmpeg integration for MP4 video export

## Performance

- **GPU Acceleration**: All rendering performed on GPU via WebGPU
- **Efficient Geometry**: Triangle-based wireframe rendering for optimal GPU utilization
- **Batched Updates**: Vertex data uploaded in single buffer writes
- **Off-screen Rendering**: Video recording uses dedicated render targets

## Testing

The renderer includes comprehensive test coverage:

```bash
# Run all tests
cargo test

# Run specific test suites
cargo test camera_tests      # Camera mathematics and controls
cargo test body_tests        # Body structure and alignment
cargo test mesh_tests        # Geometry generation
cargo test loader_tests      # NPY file loading
cargo test integration_tests # Full pipeline tests
cargo test video_tests       # Video encoding (requires FFmpeg)
```

### Test Categories

- **Unit Tests**: Individual component functionality
- **Integration Tests**: Full rendering pipeline with various scenes
- **Shader Tests**: WGSL validation and correctness
- **Performance Tests**: Large body count handling (9000+ bodies)

## Development

### Project Structure
```
renderer/
├── Cargo.toml          # Dependencies and metadata
├── README.md           # This file
├── src/
│   ├── main.rs         # Entry point
│   ├── lib.rs          # Core renderer
│   ├── body.rs         # Data structures
│   ├── camera.rs       # Camera system
│   ├── gpu.rs          # GPU management
│   ├── loader.rs       # File loading
│   ├── mesh.rs         # Geometry
│   ├── video.rs        # Recording
│   └── shaders/        # GPU shaders
└── tests/              # Test suites
```

### Adding Features

1. Follow Rust naming conventions and idioms
2. Add comprehensive tests for new functionality
3. Update documentation as needed
4. Ensure all tests pass: `cargo test`
5. Run `cargo clippy` for linting

## Troubleshooting

### Common Issues

1. **"Failed to find suitable adapter"**: Ensure your GPU supports WebGPU
2. **"FFmpeg encoding failed"**: Install FFmpeg and ensure it's in PATH
3. **Window sizing issues**: The renderer defaults to 800x600 for video recording

### Debug Mode

The wireframe generator includes debug visualization modes:
- Set `DEBUG_TRIANGLES = true` in `mesh.rs` for triangle-based debugging
- Console output shows frame counts and vertex statistics

## Dependencies

- `wgpu`: WebGPU implementation
- `winit`: Cross-platform windowing
- `glam`: Linear algebra for 3D math
- `bytemuck`: Safe transmutation for GPU buffers
- `clap`: Command-line parsing
- `image`: PNG encoding for video frames
- `npyz`: NPY file format support
- `pollster`: Async runtime for wgpu
- `futures`: Async utilities
- `env_logger`: Logging support

## License

This project is part of the physics engine simulation toolkit.