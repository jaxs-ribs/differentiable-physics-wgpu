#[cfg(not(feature = "viz"))]
compile_error!("This test requires the 'viz' feature. Run with: cargo run --features viz --bin test_viz_debug");

use physics_core::{body::Body, gpu::GpuContext};
use pollster::block_on;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    let gpu = block_on(GpuContext::new())?;
    
    // Create test scene
    let bodies = vec![
        Body::new_sphere([0.0, 5.0, 0.0], 0.5, 1.0),
        Body::new_static_box([0.0, -1.0, 0.0], [10.0, 1.0, 10.0]),
    ];
    
    println!("Test Scene:");
    println!("===========");
    for (i, body) in bodies.iter().enumerate() {
        println!("Body {}: pos=({:.2}, {:.2}, {:.2}), type={}, static={}", 
            i, 
            body.position[0], body.position[1], body.position[2],
            body.shape_data[0],
            body.shape_data[1] == 1
        );
    }
    
    // Test camera matrices
    let aspect_ratio = 1024.0 / 768.0;
    let view_proj = create_view_proj_matrix(aspect_ratio);
    
    println!("\nView-Projection Matrix:");
    for row in &view_proj {
        println!("  [{:.3}, {:.3}, {:.3}, {:.3}]", row[0], row[1], row[2], row[3]);
    }
    
    // Test vertex transformation
    println!("\nTest Vertex Transformations:");
    let test_points = [
        [0.0, 5.0, 0.0, 1.0],   // Sphere center
        [0.0, -1.0, 0.0, 1.0],  // Ground center
        [0.0, 0.0, 0.0, 1.0],   // Origin
    ];
    
    for (i, point) in test_points.iter().enumerate() {
        let transformed = transform_point(&view_proj, point);
        let ndc = [
            transformed[0] / transformed[3],
            transformed[1] / transformed[3],
            transformed[2] / transformed[3],
        ];
        println!("  Point {}: ({:.2}, {:.2}, {:.2}) -> clip=({:.2}, {:.2}, {:.2}, {:.2}) -> ndc=({:.2}, {:.2}, {:.2})",
            i, point[0], point[1], point[2],
            transformed[0], transformed[1], transformed[2], transformed[3],
            ndc[0], ndc[1], ndc[2]
        );
    }
    
    // Check if vertices are in view frustum
    println!("\nFrustum Check:");
    for (i, point) in test_points.iter().enumerate() {
        let transformed = transform_point(&view_proj, point);
        let ndc = [
            transformed[0] / transformed[3],
            transformed[1] / transformed[3],
            transformed[2] / transformed[3],
        ];
        let in_frustum = ndc[0] >= -1.0 && ndc[0] <= 1.0 &&
                         ndc[1] >= -1.0 && ndc[1] <= 1.0 &&
                         ndc[2] >= 0.0 && ndc[2] <= 1.0;
        println!("  Point {}: {}", i, if in_frustum { "IN VIEW" } else { "OUT OF VIEW" });
    }
    
    Ok(())
}

fn create_view_proj_matrix(aspect_ratio: f32) -> [[f32; 4]; 4] {
    let proj = perspective_matrix(45.0_f32.to_radians(), aspect_ratio, 0.1, 100.0);
    let view = look_at_matrix(
        [0.0, 10.0, 20.0], // eye
        [0.0, 5.0, 0.0],   // center
        [0.0, 1.0, 0.0],   // up
    );
    matrix_multiply(&proj, &view)
}

fn perspective_matrix(fov: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (fov / 2.0).tan();
    // WebGPU/Vulkan clip space: x,y in [-1,1], z in [0,1]
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, far / (far - near), 1.0],
        [0.0, 0.0, -(far * near) / (far - near), 0.0],
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
    
    [
        [s[0], u[0], -f[0], 0.0],
        [s[1], u[1], -f[1], 0.0],
        [s[2], u[2], -f[2], 0.0],
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
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

fn transform_point(matrix: &[[f32; 4]; 4], point: &[f32; 4]) -> [f32; 4] {
    let mut result = [0.0; 4];
    for i in 0..4 {
        for j in 0..4 {
            result[i] += matrix[i][j] * point[j];
        }
    }
    result
}