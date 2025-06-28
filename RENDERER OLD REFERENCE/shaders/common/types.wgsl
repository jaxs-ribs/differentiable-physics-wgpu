// Common type definitions for all shaders
// This file is meant to be included/concatenated with other shader files

struct Body {
    position: vec4<f32>,
    velocity: vec4<f32>,
    orientation: vec4<f32>,
    angular_vel: vec4<f32>,
    mass_data: vec4<f32>,      // mass, inv_mass, padding, padding
    shape_data: vec4<u32>,     // shape_type, flags, padding, padding
    shape_params: vec4<f32>,
}

struct SimParams {
    dt: f32,
    gravity_x: f32,
    gravity_y: f32,
    gravity_z: f32,
    num_bodies: u32,
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
}

struct Contact {
    body_a: u32,
    body_b: u32,
    normal: vec3<f32>,
    depth: f32,
    point: vec3<f32>,
    _padding0: f32,
}

struct BroadphaseParams {
    num_bodies: u32,
    cell_size: f32,
    grid_size: u32,
    _padding: f32,
}

struct SolverParams {
    num_iterations: u32,
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
}

// Common constants
const SHAPE_SPHERE: u32 = 0u;
const SHAPE_CAPSULE: u32 = 1u;
const SHAPE_BOX: u32 = 2u;

const FLAG_STATIC: u32 = 1u;
const FLAG_KINEMATIC: u32 = 2u;