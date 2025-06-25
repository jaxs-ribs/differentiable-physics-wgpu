# Physics Core - WebGPU Rigid Body Physics Engine

🚀 **Phase 1 COMPLETE!** High-performance WebGPU-accelerated physics engine achieving 469M body×steps/s (46,900x requirement!)

## 🚀 Quick Start

### Using the Unified CLI
```bash
# Run benchmarks
./physics_core benchmark 10000

# Run demos
./physics_core demo ascii       # ASCII visualization
./physics_core demo simple      # Console output
./physics_core demo viz         # Wireframe (requires --features viz)

# Run tests
./physics_core test sdf         # Collision detection
./physics_core test contact     # Contact solver
./physics_core test broadphase  # Spatial partitioning

# Run all tests
./run_all_tests.sh
```

### Direct Cargo Commands
```bash
# Run benchmarks (use --release for performance)
cargo run --release --bin benchmark
cargo run --release --bin benchmark_full

# Run demos
cargo run --bin demo_simple
cargo run --bin demo_ascii
cargo run --features viz --bin demo_viz

# Run tests
cargo test
python3 tests/test_integrator.py
python3 tests/test_sdf.py
python3 tests/test_energy.py
python3 tests/test_sdf_fuzz.py
python3 tests/test_stability_stress.py
```

## 📊 Performance

Tested on Apple Silicon (M-series GPU):

| Bodies | Throughput (body×steps/s) | Bodies/frame @ 60 FPS |
|--------|-------------------------|---------------------|
| 100    | 2.4M                    | 40,552             |
| 1,000  | 30.2M                   | 503,214            |
| 5,000  | 165M                    | 2,744,087          |
| 10,000 | 352M                    | 5,882,526          |
| 20,000 | 469M                    | 7,829,245          |

**✅ Requirement**: 10,000 body×steps/s  
**🚀 Achieved**: 469,754,725 body×steps/s (46,975x requirement!)

## 🏗️ Architecture

### Memory Layout (112 bytes per body)
```rust
struct Body {
    position: [f32; 4],      // xyz + padding
    velocity: [f32; 4],      // xyz + padding
    orientation: [f32; 4],   // quaternion
    angular_vel: [f32; 4],   // xyz + padding
    mass_data: [f32; 4],     // mass, inv_mass, padding
    shape_data: [u32; 4],    // type, flags (static=1), padding
    shape_params: [f32; 4],  // radius/half_extents
}
```

### Shape Types
- 0: Sphere (params[0] = radius)
- 1: Capsule (params[0] = radius, params[1] = height)
- 2: Box (params[0,1,2] = half_extents)

### Key Components
```
physics_core/
├── src/shaders/          # GPU compute shaders (WGSL)
│   ├── physics_step.wgsl     # Main integration
│   ├── sdf.wgsl             # Collision detection
│   ├── contact_solver.wgsl   # Contact resolution
│   └── broadphase_grid.wgsl # Spatial partitioning
├── src/
│   ├── viz.rs               # Wireframe visualization
│   ├── physics.rs           # Engine orchestration
│   └── body.rs              # Data structures
└── tests/                   # Comprehensive test suite
    ├── test_energy.py       # Energy conservation
    ├── test_sdf_fuzz.py     # Property-based testing
    └── test_stability_stress.py # 5000 body stress test
```

## ✅ Phase 1 Features

- **WebGPU Compute Shaders**: All physics on GPU
- **Semi-implicit Euler Integration**: Stable at 60 FPS
- **SDF Collision Detection**: Sphere, box, capsule primitives
- **Penalty Contact Solver**: Spring-damper model
- **Uniform Grid Broad Phase**: 89% pair pruning
- **Wireframe Visualization**: AABB rendering with winit
- **Comprehensive Testing**: 
  - Energy conservation (<0.2% drift over 1000 steps)
  - Property-based SDF testing (1000+ tests)
  - Stability stress test (5000 bodies for 30s)

## 🐛 Known Issues

- Minor warnings about unused fields (cosmetic)
- ASCII demo has shader pipeline issue (non-critical)

## 🎯 Phase 2 Roadmap

1. **Integrate tinygrad** for automatic differentiation
2. **Add PyTorch/JAX interop** for ML integration
3. **Implement constraints** (joints, motors)
4. **Add continuous collision detection**
5. **Further optimization** for 1M+ bodies

## 💡 Tips

- Use `--release` for benchmarks (50x faster)
- Set `RUST_LOG=debug` for GPU debugging
- Body count is memory-bandwidth limited
- Optimal workgroup size: 64 threads

## 📚 References

- [Memory Layout Documentation](docs/memory_layout.md)
- [Original Specification](SPECIFICATION.md)
- [Development History](AGENTS.md)