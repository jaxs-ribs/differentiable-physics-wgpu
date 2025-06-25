#[cfg(not(feature = "viz"))]
compile_error!("This test requires the 'viz' feature");

fn main() {
    // Test column-major matrix multiplication
    
    // Identity matrix
    let identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    
    // Translation matrix (translate by (5, 0, 0))
    // In column-major order:
    let translation = [
        [1.0, 0.0, 0.0, 0.0],  // column 0
        [0.0, 1.0, 0.0, 0.0],  // column 1
        [0.0, 0.0, 1.0, 0.0],  // column 2
        [5.0, 0.0, 0.0, 1.0],  // column 3 (translation)
    ];
    
    // Test point
    let point = [0.0, 0.0, 0.0, 1.0];
    
    // Apply translation manually
    let transformed = [
        translation[0][0] * point[0] + translation[1][0] * point[1] + translation[2][0] * point[2] + translation[3][0] * point[3],
        translation[0][1] * point[0] + translation[1][1] * point[1] + translation[2][1] * point[2] + translation[3][1] * point[3],
        translation[0][2] * point[0] + translation[1][2] * point[1] + translation[2][2] * point[2] + translation[3][2] * point[3],
        translation[0][3] * point[0] + translation[1][3] * point[1] + translation[2][3] * point[2] + translation[3][3] * point[3],
    ];
    
    println!("Column-major matrix test:");
    println!("Point: {:?}", point);
    println!("Expected after translation: [5.0, 0.0, 0.0, 1.0]");
    println!("Actual: {:?}", transformed);
    
    // Test our perspective matrix
    let aspect = 1.33;
    let fov = 45.0_f32.to_radians();
    let near = 0.1;
    let far = 100.0;
    
    let proj = perspective_matrix(fov, aspect, near, far);
    
    // Test a point at the center of the near plane
    let near_center = [0.0, 0.0, -near, 1.0];
    let proj_near = transform_vec4(&proj, &near_center);
    let ndc_near = [
        proj_near[0] / proj_near[3],
        proj_near[1] / proj_near[3],
        proj_near[2] / proj_near[3],
    ];
    
    println!("\nPerspective projection test:");
    println!("Point at near plane center: {:?}", near_center);
    println!("Projected: {:?}", proj_near);
    println!("NDC: {:?} (should be close to [0, 0, 0])", ndc_near);
    
    // Test with our actual camera setup
    let view_proj = create_view_proj_matrix(aspect);
    let test_point = [0.0, 5.0, 0.0, 1.0]; // A point at the camera target
    let transformed = transform_vec4(&view_proj, &test_point);
    let ndc = [
        transformed[0] / transformed[3],
        transformed[1] / transformed[3],
        transformed[2] / transformed[3],
    ];
    
    println!("\nFull view-projection test:");
    println!("World point at camera target: {:?}", test_point);
    println!("Clip space: {:?}", transformed);
    println!("NDC: {:?}", ndc);
    println!("In frustum: {}", ndc[0] >= -1.0 && ndc[0] <= 1.0 && ndc[1] >= -1.0 && ndc[1] <= 1.0 && ndc[2] >= 0.0 && ndc[2] <= 1.0);
}

fn create_view_proj_matrix(aspect_ratio: f32) -> [[f32; 4]; 4] {
    let proj = perspective_matrix(45.0_f32.to_radians(), aspect_ratio, 0.1, 1000.0);
    let view = look_at_matrix(
        [0.0, 20.0, 50.0], // eye
        [0.0, 5.0, 0.0],   // center
        [0.0, 1.0, 0.0],   // up
    );
    matrix_multiply(&proj, &view)
}

fn perspective_matrix(fov: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (fov / 2.0).tan();
    // Column-major perspective matrix
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
        [0.0, 0.0, 1.0, 0.0],
    ]
}

fn look_at_matrix(eye: [f32; 3], center: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
    let f = normalize([
        center[0] - eye[0],
        center[1] - eye[1],
        center[2] - eye[2],
    ]);
    let s = normalize(cross(f, up));
    let u = cross(s, f);
    
    // Column-major view matrix
    [
        [s[0], s[1], s[2], 0.0],
        [u[0], u[1], u[2], 0.0],
        [-f[0], -f[1], -f[2], 0.0],
        [-dot(s, eye), -dot(u, eye), dot(f, eye), 1.0],
    ]
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
                result[j][i] += a[j][k] * b[k][i];
            }
        }
    }
    result
}

fn transform_vec4(mat: &[[f32; 4]; 4], vec: &[f32; 4]) -> [f32; 4] {
    [
        mat[0][0] * vec[0] + mat[1][0] * vec[1] + mat[2][0] * vec[2] + mat[3][0] * vec[3],
        mat[0][1] * vec[0] + mat[1][1] * vec[1] + mat[2][1] * vec[2] + mat[3][1] * vec[3],
        mat[0][2] * vec[0] + mat[1][2] * vec[1] + mat[2][2] * vec[2] + mat[3][2] * vec[3],
        mat[0][3] * vec[0] + mat[1][3] * vec[1] + mat[2][3] * vec[2] + mat[3][3] * vec[3],
    ]
}