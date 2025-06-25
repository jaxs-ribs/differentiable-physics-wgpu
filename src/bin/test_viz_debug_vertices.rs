#[cfg(not(feature = "viz"))]
compile_error!("This test requires the 'viz' feature. Run with: cargo run --features viz --bin test_viz_debug_vertices");

use physics_core::{body::Body, gpu::GpuContext, viz::Visualizer};
use pollster::block_on;
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    let event_loop = EventLoop::new()?;
    let gpu = block_on(GpuContext::new())?;
    let mut viz = block_on(Visualizer::new(&event_loop, &gpu))?;
    
    println!("Test: Debug Vertex Generation");
    println!("=============================");
    
    // Create test bodies at positions that should definitely be visible
    let mut bodies = vec![
        // A large box at the origin - should be very visible
        Body::new_box([0.0, 0.0, 0.0], [5.0, 5.0, 5.0], 1.0),
        
        // Some spheres around it
        Body::new_sphere([-10.0, 0.0, 0.0], 2.0, 1.0),
        Body::new_sphere([10.0, 0.0, 0.0], 2.0, 1.0),
        Body::new_sphere([0.0, 10.0, 0.0], 2.0, 1.0),
        
        // A static box as ground
        Body::new_static_box([0.0, -10.0, 0.0], [20.0, 1.0, 20.0]),
    ];
    
    println!("Camera: eye=(0, 20, 50), looking at (0, 5, 0)");
    println!("Bodies created:");
    for (i, body) in bodies.iter().enumerate() {
        println!("  Body {}: pos=({:.1}, {:.1}, {:.1}), type={}, static={}",
            i, 
            body.position[0], body.position[1], body.position[2],
            if body.shape_data[0] == 0 { "sphere" } else { "box" },
            body.shape_data[1] == 1);
    }
    
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
                    // Update visualization with our test bodies
                    viz.update_bodies(&gpu, &bodies);
                    
                    // 5 bodies * 24 vertices each = 120 vertices
                    let vertex_count = bodies.len() as u32 * 24;
                    
                    println!("Rendering {} vertices for {} bodies", vertex_count, bodies.len());
                    
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