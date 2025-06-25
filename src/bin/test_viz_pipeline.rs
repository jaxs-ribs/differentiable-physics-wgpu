#[cfg(not(feature = "viz"))]
compile_error!("This test requires the 'viz' feature. Run with: cargo run --features viz --bin test_viz_pipeline");

use physics_core::{gpu::GpuContext, viz::Visualizer};
use pollster::block_on;
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
};
use wgpu::util::DeviceExt;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    let event_loop = EventLoop::new()?;
    let gpu = block_on(GpuContext::new())?;
    let mut viz = block_on(Visualizer::new(&event_loop, &gpu))?;
    
    println!("Test: Direct Vertex Pipeline");
    println!("============================");
    
    // Create test vertices directly - a large triangle that should be visible
    let test_vertices: Vec<f32> = vec![
        // Position (x, y, z), Color (r, g, b)
        -10.0, 0.0, 0.0,   1.0, 0.0, 0.0,  // Red vertex at left
        10.0, 0.0, 0.0,    0.0, 1.0, 0.0,  // Green vertex at right
        0.0, 10.0, 0.0,    0.0, 0.0, 1.0,  // Blue vertex at top
        
        // Add more lines to form a visible shape
        -10.0, 0.0, 0.0,   1.0, 1.0, 0.0,  // Yellow
        0.0, 10.0, 0.0,    1.0, 1.0, 0.0,
        
        10.0, 0.0, 0.0,    0.0, 1.0, 1.0,  // Cyan
        0.0, 10.0, 0.0,    0.0, 1.0, 1.0,
    ];
    
    println!("Test vertices:");
    for i in (0..test_vertices.len()).step_by(6) {
        println!("  Vertex {}: pos=({:.1}, {:.1}, {:.1}), color=({:.1}, {:.1}, {:.1})",
            i/6,
            test_vertices[i], test_vertices[i+1], test_vertices[i+2],
            test_vertices[i+3], test_vertices[i+4], test_vertices[i+5]);
    }
    
    // Create a line buffer and upload vertices
    let line_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Test Line Buffer"),
        contents: bytemuck::cast_slice(&test_vertices),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    });
    
    // We need to manually set up the render pipeline since we can't access viz internals
    // For now, let's just test if the window opens and clears properly
    
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
                    // For this test, just render with 6 vertices (3 lines)
                    match viz.render(&gpu, 6) {
                        Ok(_) => {
                            println!("Frame rendered successfully");
                        },
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