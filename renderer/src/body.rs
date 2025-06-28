//! Physics body representation for GPU-based simulations.
//!
//! This module defines the `Body` struct, which represents a physical object in the simulation.
//! The struct is carefully designed to be GPU-compatible with specific memory alignment requirements
//! for efficient data transfer between CPU and GPU. Each body contains position, velocity,
//! orientation, angular velocity, mass properties, and shape information.
//!
//! The 112-byte aligned structure ensures optimal GPU memory access patterns and matches
//! the layout expected by the physics compute shaders.
//!
//! # Memory Layout
//! - Total size: 112 bytes (7 × 16-byte aligned chunks)
//! - Alignment: 16 bytes (GPU requirement)
//! - Pod + Zeroable: Safe for direct GPU buffer mapping
//!
//! # Shape Types
//! - 0: Sphere (radius in shape_params[0])
//! - 2: Box (half-extents in shape_params[0..3])
//!
//! # Flags (shape_data[1])
//! - 0: Dynamic body (affected by physics)
//! - 1: Static body (infinite mass, no movement)

use bytemuck::{Pod, Zeroable};

#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Body {
    pub position: [f32; 4],      // 16 bytes
    pub velocity: [f32; 4],      // 16 bytes
    pub orientation: [f32; 4],   // 16 bytes
    pub angular_vel: [f32; 4],   // 16 bytes
    pub mass_data: [f32; 4],     // 16 bytes: mass, inv_mass, padding, padding
    pub shape_data: [u32; 4],    // 16 bytes: shape_type, flags, padding, padding
    pub shape_params: [f32; 4],  // 16 bytes
    // Total: 112 bytes
}

impl Body {
    pub fn new_sphere(position: [f32; 3], radius: f32, mass: f32) -> Self {
        Self {
            position: [position[0], position[1], position[2], 0.0],
            velocity: [0.0; 4],
            orientation: [1.0, 0.0, 0.0, 0.0], // identity quaternion
            angular_vel: [0.0; 4],
            mass_data: [mass, if mass > 0.0 { 1.0 / mass } else { 0.0 }, 0.0, 0.0],
            shape_data: [0, 0, 0, 0], // shape_type=0 (sphere), flags=0
            shape_params: [radius, 0.0, 0.0, 0.0],
        }
    }
    
    pub fn new_static_sphere(position: [f32; 3], radius: f32) -> Self {
        Self {
            position: [position[0], position[1], position[2], 0.0],
            velocity: [0.0; 4],
            orientation: [1.0, 0.0, 0.0, 0.0],
            angular_vel: [0.0; 4],
            mass_data: [0.0, 0.0, 0.0, 0.0],
            shape_data: [0, 1, 0, 0], // shape_type=0 (sphere), flags=1 (static)
            shape_params: [radius, 0.0, 0.0, 0.0],
        }
    }
    
    pub fn new_box(position: [f32; 3], half_extents: [f32; 3], mass: f32) -> Self {
        Self {
            position: [position[0], position[1], position[2], 0.0],
            velocity: [0.0; 4],
            orientation: [1.0, 0.0, 0.0, 0.0],
            angular_vel: [0.0; 4],
            mass_data: [mass, if mass > 0.0 { 1.0 / mass } else { 0.0 }, 0.0, 0.0],
            shape_data: [2, 0, 0, 0], // shape_type=2 (box), flags=0 (dynamic)
            shape_params: [half_extents[0], half_extents[1], half_extents[2], 0.0],
        }
    }
    
    pub fn new_static_box(position: [f32; 3], half_extents: [f32; 3]) -> Self {
        Self {
            position: [position[0], position[1], position[2], 0.0],
            velocity: [0.0; 4],
            orientation: [1.0, 0.0, 0.0, 0.0],
            angular_vel: [0.0; 4],
            mass_data: [0.0, 0.0, 0.0, 0.0],
            shape_data: [2, 1, 0, 0], // shape_type=2 (box), flags=1 (static)
            shape_params: [half_extents[0], half_extents[1], half_extents[2], 0.0],
        }
    }
}

// Size check is done at runtime for now

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_body_size() {
        assert_eq!(std::mem::size_of::<Body>(), 112);
        assert_eq!(std::mem::align_of::<Body>(), 16);
    }

    #[test]
    fn test_body_creation() {
        let body = Body::new_sphere([1.0, 2.0, 3.0], 0.5, 1.0);
        assert_eq!(body.position[0], 1.0);
        assert_eq!(body.position[1], 2.0);
        assert_eq!(body.position[2], 3.0);
        assert_eq!(body.shape_params[0], 0.5); // radius
        assert_eq!(body.mass_data[0], 1.0); // mass
        assert_eq!(body.shape_data[0], 0); // sphere type
        assert_eq!(body.shape_data[1], 0); // dynamic
    }

    #[test]
    fn test_static_body_creation() {
        let body = Body::new_static_sphere([0.0, 0.0, 0.0], 1.0);
        assert_eq!(body.shape_data[1], 1); // static flag
        assert_eq!(body.mass_data[0], 0.0); // no mass for static
    }

    #[test]
    fn test_box_creation() {
        let body = Body::new_static_box([0.0, -1.0, 0.0], [10.0, 1.0, 10.0]);
        assert_eq!(body.shape_data[0], 2); // box type
        assert_eq!(body.shape_data[1], 1); // static flag
        assert_eq!(body.shape_params[0], 10.0); // half extent x
        assert_eq!(body.shape_params[1], 1.0); // half extent y
        assert_eq!(body.shape_params[2], 10.0); // half extent z
    }
}