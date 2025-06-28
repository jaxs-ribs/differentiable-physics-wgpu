//! Interactive 3D camera system with orbit controls.
//!
//! This module provides a spherical coordinate-based camera that orbits around a target point.
//! The camera supports smooth rotation via mouse drag and zoom via scroll wheel, making it
//! ideal for exploring 3D physics simulations from different angles.
//!
//! # Features
//! - Orbit camera with spherical coordinates (radius, theta, phi)
//! - Mouse-based rotation controls
//! - Scroll wheel zoom
//! - Automatic view-projection matrix calculation
//! - Aspect ratio updates for window resizing
//!
//! # Coordinate System
//! - Y-up right-handed coordinate system
//! - Theta: Azimuthal angle (horizontal rotation)
//! - Phi: Polar angle (vertical rotation, clamped to prevent gimbal lock)
//! - Default view: 45° elevation angle looking along negative Z axis

use glam::{Mat4, Vec3};

const DEFAULT_FOV_DEGREES: f32 = 60.0;  // Wider FOV
const NEAR_PLANE: f32 = 0.1;
const FAR_PLANE: f32 = 1000.0;
const UP_VECTOR: Vec3 = Vec3::Y;

pub struct Camera {
    target: Vec3,
    radius: f32,
    theta: f32, // Azimuthal angle (horizontal)
    phi: f32,   // Polar angle (vertical)
    aspect_ratio: f32,
    view_projection_matrix: Mat4,
}

impl Camera {
    pub fn new(aspect_ratio: f32) -> Self {
        let mut camera = Self {
            target: Vec3::new(0.0, 5.0, 0.0),
            radius: 30.0,  // Closer to the scene
            theta: -std::f32::consts::FRAC_PI_2, // Start looking along the Z axis
            phi: std::f32::consts::FRAC_PI_4,   // Start looking from a 45 degree angle
            aspect_ratio,
            view_projection_matrix: Mat4::IDENTITY,
        };
        camera.recalculate_view_projection();
        camera
    }

    pub fn rotate(&mut self, delta_x: f32, delta_y: f32) {
        self.theta += delta_x;
        self.phi = (self.phi + delta_y).clamp(0.1, std::f32::consts::PI - 0.1);
        self.recalculate_view_projection();
    }

    pub fn zoom(&mut self, delta: f32) {
        self.radius = (self.radius - delta).max(1.0);
        self.recalculate_view_projection();
    }
    
    pub fn update_aspect_ratio(&mut self, aspect_ratio: f32) {
        self.aspect_ratio = aspect_ratio;
        self.recalculate_view_projection();
    }
    
    pub fn view_projection_matrix_transposed(&self) -> [[f32; 4]; 4] {
        self.view_projection_matrix.to_cols_array_2d()
    }

    fn recalculate_view_projection(&mut self) {
        let position = Vec3::new(
            self.target.x + self.radius * self.phi.sin() * self.theta.cos(),
            self.target.y + self.radius * self.phi.cos(),
            self.target.z + self.radius * self.phi.sin() * self.theta.sin(),
        );

        let view_matrix = Mat4::look_at_rh(position, self.target, UP_VECTOR);
        let projection_matrix = Mat4::perspective_rh(
            DEFAULT_FOV_DEGREES.to_radians(),
            self.aspect_ratio,
            NEAR_PLANE,
            FAR_PLANE,
        );
        self.view_projection_matrix = projection_matrix * view_matrix;
    }
}

