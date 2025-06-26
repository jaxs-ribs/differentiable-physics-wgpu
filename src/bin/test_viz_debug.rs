/*
 * Visualization debug analysis test - validates matrix transformations and coordinate space conversions
 * for physics body positioning. Tests view-projection matrix calculations and vertex transformations
 * without full rendering. Essential for diagnosing physics visualization issues as it isolates
 * mathematical transforms from GPU rendering, helping identify incorrect body positioning or camera bugs.
 */

#[cfg(not(feature = "viz"))]
compile_error!("This test requires the 'viz' feature. Run with: cargo run --features viz --bin test_viz_debug");

use physics_core::{
    body::Body, 
    gpu::GpuContext,
    test_utils::math::MatrixOperations,
};
use pollster::block_on;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    let _gpu = block_on(GpuContext::new())?;
    
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
        ([0.0, 5.0, 0.0], "Sphere center"),
        ([0.0, -1.0, 0.0], "Ground center"),
        ([0.0, 0.0, 0.0], "Origin"),
    ];
    
    for (i, (point, desc)) in test_points.iter().enumerate() {
        let transformed = MatrixOperations::transform_point_homogeneous(&view_proj, *point);
        let ndc = [
            transformed[0] / transformed[3],
            transformed[1] / transformed[3],
            transformed[2] / transformed[3],
        ];
        println!("  Point {}: {} ({:.2}, {:.2}, {:.2}) -> clip=({:.2}, {:.2}, {:.2}, {:.2}) -> ndc=({:.2}, {:.2}, {:.2})",
            i, desc, point[0], point[1], point[2],
            transformed[0], transformed[1], transformed[2], transformed[3],
            ndc[0], ndc[1], ndc[2]
        );
    }
    
    // Check if vertices are in view frustum
    println!("\nFrustum Check:");
    for (i, (point, desc)) in test_points.iter().enumerate() {
        let transformed = MatrixOperations::transform_point_homogeneous(&view_proj, *point);
        let ndc = [
            transformed[0] / transformed[3],
            transformed[1] / transformed[3],
            transformed[2] / transformed[3],
        ];
        let in_frustum = ndc[0] >= -1.0 && ndc[0] <= 1.0 &&
                         ndc[1] >= -1.0 && ndc[1] <= 1.0 &&
                         ndc[2] >= 0.0 && ndc[2] <= 1.0;
        println!("  Point {} ({}): {}", i, desc, if in_frustum { "IN VIEW" } else { "OUT OF VIEW" });
    }
    
    Ok(())
}

fn create_view_proj_matrix(aspect_ratio: f32) -> [[f32; 4]; 4] {
    let proj = MatrixOperations::perspective_matrix(45.0_f32.to_radians(), aspect_ratio, 0.1, 100.0);
    let view = MatrixOperations::look_at_matrix(
        [0.0, 10.0, 20.0], // eye
        [0.0, 5.0, 0.0],   // center
        [0.0, 1.0, 0.0],   // up
    );
    MatrixOperations::matrix_multiply(&proj, &view)
}