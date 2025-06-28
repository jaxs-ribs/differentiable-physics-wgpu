//! GPU-compatible physics body representation.
//! 
//! 112-byte aligned structure for efficient GPU transfer.
//! Supports spheres, boxes, and capsules with static/dynamic flags.

use bytemuck::{Pod, Zeroable};

/// Physics body with GPU-aligned memory layout (112 bytes).
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Body {
    pub position: [f32; 4],      // xyz + padding
    pub velocity: [f32; 4],      // xyz + padding
    pub orientation: [f32; 4],   // quaternion wxyz
    pub angular_vel: [f32; 4],   // xyz + padding
    pub mass_data: [f32; 4],     // [mass, inv_mass, _, _]
    pub shape_data: [u32; 4],    // [shape_type, flags, _, _]
    pub shape_params: [f32; 4],  // shape-specific parameters
}

// Shape types
const SHAPE_SPHERE: u32 = 0;
const SHAPE_BOX: u32 = 2;
const SHAPE_CAPSULE: u32 = 3;

// Body flags
const FLAG_STATIC: u32 = 1;

impl Body {
    pub fn new_sphere(position: [f32; 3], radius: f32, mass: f32) -> Self {
        Self::create_sphere(position, radius, mass, false)
    }
    
    pub fn new_static_sphere(position: [f32; 3], radius: f32) -> Self {
        Self::create_sphere(position, radius, 0.0, true)
    }
    
    pub fn new_box(position: [f32; 3], half_extents: [f32; 3], mass: f32) -> Self {
        Self::create_box(position, half_extents, mass, false)
    }
    
    pub fn new_static_box(position: [f32; 3], half_extents: [f32; 3]) -> Self {
        Self::create_box(position, half_extents, 0.0, true)
    }
    
    pub fn new_capsule(position: [f32; 3], half_height: f32, radius: f32, mass: f32) -> Self {
        Self::create_capsule(position, half_height, radius, mass, false)
    }
    
    pub fn new_static_capsule(position: [f32; 3], half_height: f32, radius: f32) -> Self {
        Self::create_capsule(position, half_height, radius, 0.0, true)
    }
    
    fn create_sphere(position: [f32; 3], radius: f32, mass: f32, is_static: bool) -> Self {
        Self::new_body(
            position,
            mass,
            SHAPE_SPHERE,
            if is_static { FLAG_STATIC } else { 0 },
            [radius, 0.0, 0.0, 0.0],
        )
    }
    
    fn create_box(position: [f32; 3], half_extents: [f32; 3], mass: f32, is_static: bool) -> Self {
        Self::new_body(
            position,
            mass,
            SHAPE_BOX,
            if is_static { FLAG_STATIC } else { 0 },
            [half_extents[0], half_extents[1], half_extents[2], 0.0],
        )
    }
    
    fn create_capsule(position: [f32; 3], half_height: f32, radius: f32, mass: f32, is_static: bool) -> Self {
        Self::new_body(
            position,
            mass,
            SHAPE_CAPSULE,
            if is_static { FLAG_STATIC } else { 0 },
            [half_height, radius, 0.0, 0.0],
        )
    }
    
    fn new_body(
        position: [f32; 3],
        mass: f32,
        shape_type: u32,
        flags: u32,
        shape_params: [f32; 4],
    ) -> Self {
        Self {
            position: pad_vec3(position),
            velocity: [0.0; 4],
            orientation: identity_quaternion(),
            angular_vel: [0.0; 4],
            mass_data: compute_mass_data(mass),
            shape_data: [shape_type, flags, 0, 0],
            shape_params,
        }
    }
}

fn pad_vec3(v: [f32; 3]) -> [f32; 4] {
    [v[0], v[1], v[2], 0.0]
}

fn identity_quaternion() -> [f32; 4] {
    [1.0, 0.0, 0.0, 0.0]
}

fn compute_mass_data(mass: f32) -> [f32; 4] {
    let inverse_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
    [mass, inverse_mass, 0.0, 0.0]
}


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
        assert_eq!(body.shape_data[0], SHAPE_SPHERE);
        assert_eq!(body.shape_data[1], 0); // dynamic
    }

    #[test]
    fn test_static_body_creation() {
        let body = Body::new_static_sphere([0.0, 0.0, 0.0], 1.0);
        assert_eq!(body.shape_data[1], FLAG_STATIC);
        assert_eq!(body.mass_data[0], 0.0); // no mass for static
    }

    #[test]
    fn test_box_creation() {
        let body = Body::new_static_box([0.0, -1.0, 0.0], [10.0, 1.0, 10.0]);
        assert_eq!(body.shape_data[0], SHAPE_BOX);
        assert_eq!(body.shape_data[1], FLAG_STATIC);
        assert_eq!(body.shape_params[0], 10.0); // half extent x
        assert_eq!(body.shape_params[1], 1.0); // half extent y
        assert_eq!(body.shape_params[2], 10.0); // half extent z
    }
}