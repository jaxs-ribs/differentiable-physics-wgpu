/*
Simple Console Physics Demonstration

This demo showcases the basic functionality of the physics engine by running a simple simulation
and outputting key metrics to the console. It serves as a minimal example for users learning the
API and validates that core physics systems are working correctly. Essential for documentation,
tutorials, and quick verification that the engine installation is functional.
*/

use physics_core::{body::Body, gpu::GpuContext};
use pollster::block_on;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use std::time::Instant;

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
    
    // Create falling spheres
    println!("Creating demo scene:");
    for i in 0..5 {
        let x = (i as f32 - 2.0) * 2.0;
        println!("  Sphere {} at x={}", i, x);
        bodies.push(Body::new_sphere([x, 10.0, 0.0], 0.5, 1.0));
    }
    
    // Ground plane
    println!("  Ground plane at y=-1");
    bodies.push(Body::new_static_box([0.0, -1.0, 0.0], [20.0, 1.0, 20.0]));
    
    bodies
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    println!("Physics Engine Console Demo");
    println!("===========================\n");
    
    let gpu = block_on(GpuContext::new())?;
    
    // Create scene
    let mut bodies = create_demo_scene();
    let num_bodies = bodies.len() as u32;
    
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
        num_bodies: bodies.len() as u32,
        _padding: [0.0; 3],
    };
    
    println!("SimParams size: {} bytes", std::mem::size_of::<SimParams>());
    println!("SimParams: dt={}, gravity=({}, {}, {}), num_bodies={}", 
        sim_params.dt, sim_params.gravity_x, sim_params.gravity_y, sim_params.gravity_z, sim_params.num_bodies);
    
    let params_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Params Buffer"),
        contents: bytemuck::cast_slice(&[sim_params]),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    
    // Load shader - use fixed version
    let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Physics Step Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/physics_step_fixed.wgsl").into()),
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
    
    println!("\nRunning physics simulation for 3 seconds...\n");
    
    // Debug: print initial body properties
    for (i, body) in bodies.iter().enumerate() {
        if i < 5 {
            println!("Body {}: mass={}, is_static={}", i, body.mass_data[0], body.shape_data[1]);
        }
    }
    
    let start_time = Instant::now();
    let mut step_count = 0;
    let dt = 0.016; // 60 FPS
    
    while start_time.elapsed().as_secs_f32() < 3.0 {
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
        
        // Read back positions every 0.5 seconds
        if step_count % 30 == 0 {
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
            
            println!("Time: {:.1}s", step_count as f32 * dt);
            for (i, body) in bodies.iter().enumerate() {
                if body.shape_data[1] == 0 { // Dynamic bodies only
                    println!("  Body {}: y = {:.2} m, vy = {:.2} m/s", 
                        i, body.position[1], body.velocity[1]);
                }
            }
            println!();
        }
        
        step_count += 1;
        
        // Sleep to maintain 60 FPS
        std::thread::sleep(std::time::Duration::from_millis(16));
    }
    
    println!("Simulation complete!");
    println!("Total steps: {}", step_count);
    println!("Average FPS: {:.1}", step_count as f32 / 3.0);
    
    Ok(())
}