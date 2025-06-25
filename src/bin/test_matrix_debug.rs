#[cfg(not(feature = "viz"))]
compile_error!("This test requires the 'viz' feature. Run with: cargo run --features viz --bin test_matrix_debug");

use physics_core::{body::Body, gpu::GpuContext, viz::Visualizer};
use pollster::block_on;
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
};

// Copy the matrix functions from viz.rs to debug them
fn create_view_proj_matrix(aspect_ratio: f32) -> [[f32; 4]; 4] {
    let proj = perspective_matrix(45.0_f32.to_radians(), aspect_ratio, 0.1, 1000.0);
    let view = look_at_matrix(
        [0.0, 20.0, 50.0], // eye
        [0.0, 5.0, 0.0],   // center
        [0.0, 1.0, 0.0],   // up
    );
    
    println!("\nProjection Matrix:");
    print_matrix(&proj);
    
    println!("\nView Matrix:");
    print_matrix(&view);
    
    let view_proj = matrix_multiply(&proj, &view);
    println!("\nView-Projection Matrix:");
    print_matrix(&view_proj);
    
    view_proj
}

fn perspective_matrix(fov: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (fov / 2.0).tan();
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

fn transpose_matrix(m: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    [
        [m[0][0], m[1][0], m[2][0], m[3][0]],
        [m[0][1], m[1][1], m[2][1], m[3][1]],
        [m[0][2], m[1][2], m[2][2], m[3][2]],
        [m[0][3], m[1][3], m[2][3], m[3][3]],
    ]
}

fn print_matrix(m: &[[f32; 4]; 4]) {
    for i in 0..4 {
        println!("  [{:8.3} {:8.3} {:8.3} {:8.3}]", m[i][0], m[i][1], m[i][2], m[i][3]);
    }
}

fn transform_point(m: &[[f32; 4]; 4], p: [f32; 3]) -> [f32; 4] {
    let p4 = [p[0], p[1], p[2], 1.0];
    let mut result = [0.0; 4];
    
    for i in 0..4 {
        for j in 0..4 {
            result[i] += m[i][j] * p4[j];
        }
    }
    
    result
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    println!("Matrix Debug Test");
    println!("=================");
    
    let aspect = 1024.0 / 768.0;
    let view_proj = create_view_proj_matrix(aspect);
    
    // Test some points
    let test_points = [
        ([0.0, 0.0, 0.0], "Origin"),
        ([0.0, 5.0, 0.0], "Camera look-at target"),
        ([5.0, 5.0, 0.0], "Right of center"),
        ([-5.0, 5.0, 0.0], "Left of center"),
        ([0.0, 10.0, 0.0], "Above center"),
        ([0.0, 0.0, -10.0], "Behind center"),
        ([0.0, 0.0, 10.0], "In front of center"),
    ];
    
    println!("\nTransforming test points:");
    println!("==========================");
    
    for (point, desc) in test_points.iter() {
        let transformed = transform_point(&view_proj, *point);
        let clip_space = [
            transformed[0] / transformed[3],
            transformed[1] / transformed[3],
            transformed[2] / transformed[3],
        ];
        
        println!("\n{}: ({:.1}, {:.1}, {:.1})", desc, point[0], point[1], point[2]);
        println!("  Transformed: ({:.3}, {:.3}, {:.3}, {:.3})", 
            transformed[0], transformed[1], transformed[2], transformed[3]);
        println!("  Clip space: ({:.3}, {:.3}, {:.3})",
            clip_space[0], clip_space[1], clip_space[2]);
        
        // Check if in view frustum
        let in_frustum = clip_space[0] >= -1.0 && clip_space[0] <= 1.0 &&
                        clip_space[1] >= -1.0 && clip_space[1] <= 1.0 &&
                        clip_space[2] >= 0.0 && clip_space[2] <= 1.0;
        println!("  In frustum: {}", in_frustum);
    }
    
    // Test with transposed matrix (for GPU)
    println!("\n\nTransposed Matrix Test:");
    println!("=======================");
    let view_proj_t = transpose_matrix(&view_proj);
    println!("Transposed View-Projection Matrix:");
    print_matrix(&view_proj_t);
    
    // Now let's test what happens to a typical AABB vertex
    let body_pos = [0.0, 5.0, 0.0];
    let box_half = [5.0, 5.0, 5.0];
    let aabb_min = [body_pos[0] - box_half[0], body_pos[1] - box_half[1], body_pos[2] - box_half[2]];
    let aabb_max = [body_pos[0] + box_half[0], body_pos[1] + box_half[1], body_pos[2] + box_half[2]];
    
    println!("\n\nTesting AABB corners for box at ({:.1}, {:.1}, {:.1}):", 
        body_pos[0], body_pos[1], body_pos[2]);
    println!("AABB: min=({:.1}, {:.1}, {:.1}), max=({:.1}, {:.1}, {:.1})",
        aabb_min[0], aabb_min[1], aabb_min[2],
        aabb_max[0], aabb_max[1], aabb_max[2]);
    
    let corners = [
        [aabb_min[0], aabb_min[1], aabb_min[2]],
        [aabb_max[0], aabb_min[1], aabb_min[2]],
        [aabb_max[0], aabb_max[1], aabb_min[2]],
        [aabb_min[0], aabb_max[1], aabb_min[2]],
        [aabb_min[0], aabb_min[1], aabb_max[2]],
        [aabb_max[0], aabb_min[1], aabb_max[2]],
        [aabb_max[0], aabb_max[1], aabb_max[2]],
        [aabb_min[0], aabb_max[1], aabb_max[2]],
    ];
    
    let mut any_visible = false;
    for (i, corner) in corners.iter().enumerate() {
        let transformed = transform_point(&view_proj, *corner);
        let clip_space = [
            transformed[0] / transformed[3],
            transformed[1] / transformed[3],
            transformed[2] / transformed[3],
        ];
        
        let in_frustum = clip_space[0] >= -1.0 && clip_space[0] <= 1.0 &&
                        clip_space[1] >= -1.0 && clip_space[1] <= 1.0 &&
                        clip_space[2] >= 0.0 && clip_space[2] <= 1.0;
        
        if in_frustum {
            any_visible = true;
            println!("  Corner {}: ({:.1}, {:.1}, {:.1}) -> clip ({:.3}, {:.3}, {:.3}) [VISIBLE]",
                i, corner[0], corner[1], corner[2],
                clip_space[0], clip_space[1], clip_space[2]);
        }
    }
    
    println!("\nAny corners visible: {}", any_visible);
    
    // Also launch the visualizer to compare
    let event_loop = EventLoop::new()?;
    let gpu = block_on(GpuContext::new())?;
    let mut viz = block_on(Visualizer::new(&event_loop, &gpu))?;
    
    // Create a single test body
    let mut bodies = vec![
        Body::new_box([0.0, 5.0, 0.0], [5.0, 5.0, 5.0], 1.0),
    ];
    
    event_loop.run(move |event, control_flow| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == viz.window().id() => match event {
                WindowEvent::CloseRequested => {
                    control_flow.exit();
                },
                WindowEvent::Resized(physical_size) => {
                    viz.resize(&gpu, *physical_size);
                },
                WindowEvent::RedrawRequested => {
                    viz.update_bodies(&gpu, &bodies);
                    let vertex_count = bodies.len() as u32 * 24;
                    
                    match viz.render(&gpu, vertex_count) {
                        Ok(_) => {},
                        Err(wgpu::SurfaceError::Lost) => viz.resize(&gpu, viz.window().inner_size()),
                        Err(wgpu::SurfaceError::OutOfMemory) => control_flow.exit(),
                        Err(e) => eprintln!("Render error: {:?}", e),
                    }
                    
                    viz.window().request_redraw();
                },
                _ => {}
            },
            _ => {}
        }
    })?;
    
    Ok(())
}