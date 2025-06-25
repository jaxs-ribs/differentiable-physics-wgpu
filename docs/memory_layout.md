# Memory Layout

## Decision: Array of Structures (AoS)

We chose AoS over SoA for Phase 1 because:
1. Simpler to map between CPU and GPU
2. Each body's data is contiguous, better for small simulations
3. Easier to extend with per-body properties
4. Phase 2 can refactor to SoA if needed for better GPU performance

## Body Structure (112 bytes aligned)

```rust
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Body {
    position: [f32; 4],      // 16 bytes
    velocity: [f32; 4],      // 16 bytes
    orientation: [f32; 4],   // 16 bytes: quaternion wxyz
    angular_vel: [f32; 4],   // 16 bytes
    mass_data: [f32; 4],     // 16 bytes: mass, inv_mass, padding, padding
    shape_data: [u32; 4],    // 16 bytes: shape_type, flags, padding, padding
    shape_params: [f32; 4],  // 16 bytes: radius, height, etc
}
```

Total: 112 bytes per body (7 × 16 bytes)

## WGSL Mapping

```wgsl
struct Body {
    position: vec4<f32>,
    velocity: vec4<f32>,
    orientation: vec4<f32>,
    angular_vel: vec4<f32>,
    mass_data: vec4<f32>,      // mass, inv_mass, padding, padding
    shape_data: vec4<u32>,     // shape_type, flags, padding, padding
    shape_params: vec4<f32>,
}
```

## Buffer Layout

Bodies are stored in a single buffer:
- Buffer binding: 0
- Access: read_write
- Size: num_bodies × 112 bytes
- Alignment: 16 bytes (vec4 aligned)