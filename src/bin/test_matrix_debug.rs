/*
Interactive Matrix Math Debug Tool

This visualization tool allows interactive debugging of matrix transformations and camera operations
in the 3D rendering pipeline. It helps identify transformation errors, projection issues, and
coordinate system problems by providing real-time visual feedback. Critical for debugging rendering
problems and ensuring correct spatial transformations in the visualization system.
*/

#[cfg(not(feature = "viz"))]
compile_error!("This test requires the 'viz' feature. Run with: cargo run --features viz --bin test_matrix_debug");

use physics_core::{
    body::Body, 
    gpu::GpuContext, 
    viz::Visualizer,
    test_utils::math::{MatrixOperations, print_matrix, print_matrix_with_label},
};
use pollster::block_on;
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
};

fn create_view_proj_matrix(aspect_ratio: f32) -> [[f32; 4]; 4] {
    let proj = MatrixOperations::perspective_matrix(45.0_f32.to_radians(), aspect_ratio, 0.1, 1000.0);
    let view = MatrixOperations::look_at_matrix(
        [0.0, 20.0, 50.0], // eye
        [0.0, 5.0, 0.0],   // center
        [0.0, 1.0, 0.0],   // up
    );
    
    print_matrix_with_label("\nProjection Matrix", &proj);
    print_matrix_with_label("\nView Matrix", &view);
    
    let view_proj = MatrixOperations::matrix_multiply(&proj, &view);
    print_matrix_with_label("\nView-Projection Matrix", &view_proj);
    
    view_proj
}

fn test_point_transformation(view_proj: &[[f32; 4]; 4], point: [f32; 3], description: &str) {
    let transformed = MatrixOperations::transform_point_homogeneous(view_proj, point);
    let clip_space = [
        transformed[0] / transformed[3],
        transformed[1] / transformed[3],
        transformed[2] / transformed[3],
    ];
    
    println!("\n{}: ({:.1}, {:.1}, {:.1})", description, point[0], point[1], point[2]);
    println!("  Transformed: ({:.3}, {:.3}, {:.3}, {:.3})", 
        transformed[0], transformed[1], transformed[2], transformed[3]);
    println!("  Clip space: ({:.3}, {:.3}, {:.3})",
        clip_space[0], clip_space[1], clip_space[2]);
    
    let in_frustum = clip_space[0] >= -1.0 && clip_space[0] <= 1.0 &&
                    clip_space[1] >= -1.0 && clip_space[1] <= 1.0 &&
                    clip_space[2] >= 0.0 && clip_space[2] <= 1.0;
    println!("  In frustum: {}", in_frustum);
}

fn test_aabb_visibility(view_proj: &[[f32; 4]; 4], body_pos: [f32; 3], box_half: [f32; 3]) {
    let aabb_min = [
        body_pos[0] - box_half[0], 
        body_pos[1] - box_half[1], 
        body_pos[2] - box_half[2]
    ];
    let aabb_max = [
        body_pos[0] + box_half[0], 
        body_pos[1] + box_half[1], 
        body_pos[2] + box_half[2]
    ];
    
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
        let transformed = MatrixOperations::transform_point_homogeneous(view_proj, *corner);
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
        test_point_transformation(&view_proj, *point, desc);
    }
    
    // Test with transposed matrix (for GPU)
    println!("\n\nTransposed Matrix Test:");
    println!("=======================");
    let view_proj_t = MatrixOperations::transpose_matrix(&view_proj);
    print_matrix_with_label("Transposed View-Projection Matrix", &view_proj_t);
    
    // Test AABB visibility
    test_aabb_visibility(&view_proj, [0.0, 5.0, 0.0], [5.0, 5.0, 5.0]);
    
    // Also launch the visualizer to compare
    let event_loop = EventLoop::new()?;
    let gpu = block_on(GpuContext::new())?;
    let mut viz = block_on(Visualizer::new(&event_loop, &gpu))?;
    
    // Create a single test body
    let bodies = vec![
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