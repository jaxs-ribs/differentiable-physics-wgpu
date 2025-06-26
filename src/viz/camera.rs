use glam::{Mat4, Vec3};

const DEFAULT_FOV_DEGREES: f32 = 45.0;
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
            radius: 40.0,
            theta: -std::f32::consts::FRAC_PI_2, // Start looking along the Z axis
            phi: std::f32::consts::FRAC_PI_3,   // Start looking from a 60 degree angle
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

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / len, v[1] / len, v[2] / len]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn matrix_multiply(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut result = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

fn transpose_matrix(m: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    [
        [m[0][0], m[1][0], m[2][0], m[3][0]],
        [m[0][1], m[1][1], m[2][1], m[3][1]],
        [m[0][2], m[1][2], m[2][2], m[3][2]],
        [m[0][3], m[1][3], m[2][3], m[3][3]],
    ]
}