//! Orbit camera with spherical coordinates.
//! 
//! Y-up coordinate system, mouse drag for rotation, scroll for zoom.
//! Theta = horizontal, Phi = vertical (clamped to prevent gimbal lock).

use glam::{Mat4, Vec3};

const DEFAULT_FOV_DEGREES: f32 = 60.0;
const NEAR_PLANE: f32 = 0.1;
const FAR_PLANE: f32 = 1000.0;
const UP_VECTOR: Vec3 = Vec3::Y;
const MIN_PHI: f32 = 0.1;
const MAX_PHI: f32 = std::f32::consts::PI - 0.1;
const MIN_RADIUS: f32 = 1.0;

/// Orbit camera around a target point.
pub struct Camera {
    target: Vec3,
    radius: f32,
    theta: f32,  // Horizontal angle
    phi: f32,    // Vertical angle
    aspect_ratio: f32,
    view_projection_matrix: Mat4,
}

impl Camera {
    pub fn new(aspect_ratio: f32) -> Self {
        let mut camera = Self {
            target: Vec3::new(0.0, 5.0, 0.0),
            radius: 30.0,
            theta: -std::f32::consts::FRAC_PI_2,  // Looking along Z
            phi: std::f32::consts::FRAC_PI_4,     // 45° elevation
            aspect_ratio,
            view_projection_matrix: Mat4::IDENTITY,
        };
        camera.update_matrices();
        camera
    }

    pub fn rotate(&mut self, delta_x: f32, delta_y: f32) {
        self.theta += delta_x;
        self.phi = (self.phi + delta_y).clamp(MIN_PHI, MAX_PHI);
        self.update_matrices();
    }

    pub fn zoom(&mut self, delta: f32) {
        self.radius = (self.radius - delta).max(MIN_RADIUS);
        self.update_matrices();
    }
    
    pub fn update_aspect_ratio(&mut self, aspect_ratio: f32) {
        self.aspect_ratio = aspect_ratio;
        self.update_matrices();
    }
    
    pub fn view_projection_matrix_transposed(&self) -> [[f32; 4]; 4] {
        self.view_projection_matrix.to_cols_array_2d()
    }

    fn update_matrices(&mut self) {
        let position = self.calculate_eye_position();
        let view = Mat4::look_at_rh(position, self.target, UP_VECTOR);
        let projection = Mat4::perspective_rh(
            DEFAULT_FOV_DEGREES.to_radians(),
            self.aspect_ratio,
            NEAR_PLANE,
            FAR_PLANE,
        );
        self.view_projection_matrix = projection * view;
    }
    
    fn calculate_eye_position(&self) -> Vec3 {
        let sin_phi = self.phi.sin();
        Vec3::new(
            self.target.x + self.radius * sin_phi * self.theta.cos(),
            self.target.y + self.radius * self.phi.cos(),
            self.target.z + self.radius * sin_phi * self.theta.sin(),
        )
    }
}

