pub struct MatrixOperations;

impl MatrixOperations {
    pub fn perspective_matrix(fov: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
        let f = 1.0 / (fov / 2.0).tan();
        
        [
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, far / (far - near), 1.0],
            [0.0, 0.0, -(far * near) / (far - near), 0.0],
        ]
    }
    
    pub fn look_at_matrix(eye: [f32; 3], center: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
        let forward = Self::normalize([
            center[0] - eye[0],
            center[1] - eye[1],
            center[2] - eye[2],
        ]);
        let right = Self::normalize(Self::cross(forward, up));
        let up = Self::cross(right, forward);
        
        [
            [right[0], up[0], -forward[0], 0.0],
            [right[1], up[1], -forward[1], 0.0],
            [right[2], up[2], -forward[2], 0.0],
            [-Self::dot(right, eye), -Self::dot(up, eye), Self::dot(forward, eye), 1.0],
        ]
    }
    
    pub fn create_view_projection_matrix(aspect_ratio: f32) -> [[f32; 4]; 4] {
        let projection = Self::perspective_matrix(45.0_f32.to_radians(), aspect_ratio, 0.1, 1000.0);
        let view = Self::look_at_matrix(
            [0.0, 20.0, 50.0], // eye
            [0.0, 5.0, 0.0],   // center
            [0.0, 1.0, 0.0],   // up
        );
        Self::matrix_multiply(&projection, &view)
    }
    
    pub fn matrix_multiply(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
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
    
    pub fn transpose_matrix(m: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
        [
            [m[0][0], m[1][0], m[2][0], m[3][0]],
            [m[0][1], m[1][1], m[2][1], m[3][1]],
            [m[0][2], m[1][2], m[2][2], m[3][2]],
            [m[0][3], m[1][3], m[2][3], m[3][3]],
        ]
    }
    
    pub fn transform_point(matrix: &[[f32; 4]; 4], point: [f32; 3]) -> [f32; 3] {
        let w = matrix[3][0] * point[0] + matrix[3][1] * point[1] + matrix[3][2] * point[2] + matrix[3][3];
        
        [
            (matrix[0][0] * point[0] + matrix[0][1] * point[1] + matrix[0][2] * point[2] + matrix[0][3]) / w,
            (matrix[1][0] * point[0] + matrix[1][1] * point[1] + matrix[1][2] * point[2] + matrix[1][3]) / w,
            (matrix[2][0] * point[0] + matrix[2][1] * point[1] + matrix[2][2] * point[2] + matrix[2][3]) / w,
        ]
    }
    
    pub fn transform_point_homogeneous(matrix: &[[f32; 4]; 4], point: [f32; 3]) -> [f32; 4] {
        let p4 = [point[0], point[1], point[2], 1.0];
        let mut result = [0.0; 4];
        
        for i in 0..4 {
            for j in 0..4 {
                result[i] += matrix[i][j] * p4[j];
            }
        }
        
        result
    }
    
    pub fn normalize(v: [f32; 3]) -> [f32; 3] {
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / len, v[1] / len, v[2] / len]
    }
    
    pub fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }
    
    pub fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }
}

// Convenience functions for common operations
pub fn print_matrix(m: &[[f32; 4]; 4]) {
    for i in 0..4 {
        println!("  [{:8.3} {:8.3} {:8.3} {:8.3}]", m[i][0], m[i][1], m[i][2], m[i][3]);
    }
}

pub fn print_matrix_with_label(label: &str, m: &[[f32; 4]; 4]) {
    println!("{}:", label);
    print_matrix(m);
}