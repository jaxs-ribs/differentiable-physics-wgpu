const DEFAULT_FOV_DEGREES: f32 = 45.0;
const NEAR_PLANE: f32 = 0.1;
const FAR_PLANE: f32 = 1000.0;
const CAMERA_POSITION: [f32; 3] = [0.0, 20.0, 50.0];
const CAMERA_TARGET: [f32; 3] = [0.0, 5.0, 0.0];
const UP_VECTOR: [f32; 3] = [0.0, 1.0, 0.0];

pub struct Camera {
    view_projection_matrix: [[f32; 4]; 4],
}

impl Camera {
    pub fn new(aspect_ratio: f32) -> Self {
        let view_projection_matrix = Self::create_view_projection_matrix(aspect_ratio);
        Self { view_projection_matrix }
    }
    
    pub fn update_aspect_ratio(&mut self, aspect_ratio: f32) {
        self.view_projection_matrix = Self::create_view_projection_matrix(aspect_ratio);
    }
    
    pub fn view_projection_matrix(&self) -> [[f32; 4]; 4] {
        self.view_projection_matrix
    }
    
    pub fn view_projection_matrix_transposed(&self) -> [[f32; 4]; 4] {
        transpose_matrix(&self.view_projection_matrix)
    }
    
    fn create_view_projection_matrix(aspect_ratio: f32) -> [[f32; 4]; 4] {
        let projection = Self::create_perspective_matrix(aspect_ratio);
        let view = Self::create_look_at_matrix();
        matrix_multiply(&projection, &view)
    }
    
    fn create_perspective_matrix(aspect_ratio: f32) -> [[f32; 4]; 4] {
        let fov_radians = DEFAULT_FOV_DEGREES.to_radians();
        let f = 1.0 / (fov_radians / 2.0).tan();
        
        [
            [f / aspect_ratio, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, FAR_PLANE / (FAR_PLANE - NEAR_PLANE), 1.0],
            [0.0, 0.0, -(FAR_PLANE * NEAR_PLANE) / (FAR_PLANE - NEAR_PLANE), 0.0],
        ]
    }
    
    fn create_look_at_matrix() -> [[f32; 4]; 4] {
        let forward = normalize([
            CAMERA_TARGET[0] - CAMERA_POSITION[0],
            CAMERA_TARGET[1] - CAMERA_POSITION[1],
            CAMERA_TARGET[2] - CAMERA_POSITION[2],
        ]);
        let right = normalize(cross(forward, UP_VECTOR));
        let up = cross(right, forward);
        
        [
            [right[0], up[0], -forward[0], 0.0],
            [right[1], up[1], -forward[1], 0.0],
            [right[2], up[2], -forward[2], 0.0],
            [-dot(right, CAMERA_POSITION), -dot(up, CAMERA_POSITION), dot(forward, CAMERA_POSITION), 1.0],
        ]
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