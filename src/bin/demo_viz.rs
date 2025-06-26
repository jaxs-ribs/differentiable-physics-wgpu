/*
3D Wireframe Physics Visualization Demo

This demo provides real-time 3D wireframe visualization of physics simulation using GPU-accelerated
rendering. It showcases the complete integration between physics computation and visualization
systems, allowing users to observe complex multi-body dynamics in an interactive 3D environment.
Essential for debugging collision detection, validating physics behavior, and demonstrating engine capabilities.
*/

#[cfg(not(feature = "viz"))]
compile_error!("This demo requires the 'viz' feature. Run with: cargo run --features viz --bin demo_viz");

use physics_core::{body::Body, gpu::GpuContext, viz::Visualizer};
use pollster::block_on;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct SimParams {
    dt: f32,
    gravity_x: f32,
    gravity_y: f32,
    gravity_z: f32,
    num_bodies: u32,
    _padding: [f32; 3],
}

fn create_demo_scene() -> Vec<Body> {
    let mut bodies = Vec::new();
    
    // Create falling spheres in a grid
    for i in 0..3 {
        for j in 0..3 {
            let x = (i as f32 - 1.0) * 2.0;
            let z = (j as f32 - 1.0) * 2.0;
            let y = 10.0 + (i + j) as f32 * 0.5;
            bodies.push(Body::new_sphere([x, y, z], 0.5, 1.0));
        }
    }
    
    // Add some static obstacles
    bodies.push(Body::new_static_sphere([-3.0, 5.0, 0.0], 0.5));
    bodies.push(Body::new_static_sphere([3.0, 5.0, 0.0], 0.5));
    
    // Ground plane
    bodies.push(Body::new_static_box([0.0, -1.0, 0.0], [10.0, 1.0, 10.0]));
    
    bodies
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    let event_loop = EventLoop::new()?;
    let gpu = block_on(GpuContext::new())?;
    let mut viz = block_on(Visualizer::new(&event_loop, &gpu))?;
    
    // Create scene
    let mut bodies = create_demo_scene();
    let num_bodies = bodies.len() as u32;
    
    println!("Physics Engine Wireframe Visualization");
    println!("=====================================");
    println!("Showing {} bodies", num_bodies);
    println!("Green: Dynamic bodies");
    println!("Gray: Static bodies");
    
    // Create buffers
    let bodies_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Bodies Buffer"),
        contents: bytemuck::cast_slice(&bodies),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    });
    
    let sim_params = SimParams {
        dt: 0.016,
        gravity_x: 0.0,
        gravity_y: -9.81,
        gravity_z: 0.0,
        num_bodies,
        _padding: [0.0; 3],
    };
    
    let params_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Params Buffer"),
        contents: bytemuck::cast_slice(&[sim_params]),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    
    // Load shader
    let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Physics Step Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/physics_step.wgsl").into()),
    });
    
    // Create pipeline
    let bind_group_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Physics Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Physics Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: bodies_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });
    
    let pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Physics Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    
    let pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Physics Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("physics_step"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    
    // Create staging buffer for reading back
    let staging_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: (std::mem::size_of::<Body>() * bodies.len()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
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
                    // Run physics step
                    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Physics Encoder"),
                    });
                    
                    {
                        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("Physics Pass"),
                            timestamp_writes: None,
                        });
                        
                        compute_pass.set_pipeline(&pipeline);
                        compute_pass.set_bind_group(0, &bind_group, &[]);
                        
                        let workgroups = (num_bodies + 63) / 64;
                        compute_pass.dispatch_workgroups(workgroups, 1, 1);
                    }
                    
                    // Copy buffer for reading
                    encoder.copy_buffer_to_buffer(&bodies_buffer, 0, &staging_buffer, 0, staging_buffer.size());
                    
                    gpu.queue.submit(Some(encoder.finish()));
                    gpu.device.poll(wgpu::Maintain::Wait);
                    
                    // Read back positions
                    let buffer_slice = staging_buffer.slice(..);
                    let (tx, rx) = futures::channel::oneshot::channel();
                    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                        tx.send(result).unwrap();
                    });
                    gpu.device.poll(wgpu::Maintain::Wait);
                    block_on(rx).unwrap().unwrap();
                    
                    let data = buffer_slice.get_mapped_range();
                    bodies.copy_from_slice(bytemuck::cast_slice(&data));
                    drop(data);
                    staging_buffer.unmap();
                    
                    // Update visualization
                    viz.update_bodies(&gpu, &bodies);
                    
                    // Calculate vertex count (12 lines * 2 vertices per AABB)
                    let vertex_count = bodies.len() as u32 * 24;
                    
                    // Render
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