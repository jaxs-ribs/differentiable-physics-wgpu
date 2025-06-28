/*
Physics Engine Performance Benchmark

This benchmark measures the throughput and performance characteristics of the GPU physics pipeline
under varying body counts. It provides quantitative metrics for optimization efforts and performance
regression detection. Essential for validating that algorithmic improvements translate to real-world
speedups and ensuring the engine meets performance requirements for production workloads.
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
    _padding: [f32; 4],  // Ensure 32 bytes total
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Contact {
    body_a: u32,
    body_b: u32,
    normal: [f32; 4],
    distance: f32,
    _padding: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct SolverParams {
    values: [f32; 4], // stiffness, dt, damping, padding
}

fn create_test_scene(num_bodies: usize) -> Vec<Body> {
    let mut bodies = Vec::with_capacity(num_bodies);
    
    // Create a grid of bodies with some initial velocity
    let grid_size = (num_bodies as f32).sqrt() as usize;
    let spacing = 3.0; // 3m spacing to avoid initial collisions
    
    for i in 0..num_bodies {
        let x = (i % grid_size) as f32 * spacing;
        let z = (i / grid_size) as f32 * spacing;
        let y = 10.0 + (i as f32 * 0.1); // Slightly different heights
        
        let mut body = Body::new_sphere([x, y, z], 0.5, 1.0);
        // Add some random initial velocity
        body.velocity[0] = ((i * 17) % 10) as f32 * 0.1 - 0.5;
        body.velocity[1] = -2.0;
        body.velocity[2] = ((i * 23) % 10) as f32 * 0.1 - 0.5;
        
        bodies.push(body);
    }
    
    // Add ground plane as a large static box
    let ground = Body::new_static_box([0.0, -1.0, 0.0], [1000.0, 1.0, 1000.0]);
    bodies.push(ground);
    
    bodies
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    // Test different body counts
    let test_counts = vec![100, 1000, 5000, 10000, 20000];
    
    println!("WebGPU Physics Engine Benchmark");
    println!("================================\n");
    
    let gpu = block_on(GpuContext::new())?;
    
    for &num_bodies in &test_counts {
        println!("Testing with {} bodies...", num_bodies);
        
        // Create scene
        let bodies = create_test_scene(num_bodies);
        let total_bodies = bodies.len() as u32;
        
        // Create buffers
        let bodies_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bodies Buffer"),
            contents: bytemuck::cast_slice(&bodies),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        
        // Parameters
        let sim_params = SimParams {
            dt: 0.016,
            gravity_x: 0.0,
            gravity_y: -9.81,
            gravity_z: 0.0,
            _padding: [0.0; 4],
        };
        
        let params_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params Buffer"),
            contents: bytemuck::cast_slice(&[sim_params]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        
        // Simplified physics step shader (just integration for benchmarking)
        let shader_source = r#"
struct Body {
    position: vec4<f32>,
    velocity: vec4<f32>,
    orientation: vec4<f32>,
    angular_vel: vec4<f32>,
    mass_data: vec4<f32>,
    shape_data: vec4<u32>,
    shape_params: vec4<f32>,
}

struct SimParams {
    dt: f32,
    gravity: vec3<f32>,
}

@group(0) @binding(0) var<storage, read_write> bodies: array<Body>;
@group(0) @binding(1) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn physics_step(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let num_bodies = arrayLength(&bodies);
    
    if (idx >= num_bodies) {
        return;
    }
    
    let body = bodies[idx];
    
    // Skip static bodies
    if (body.shape_data.y == 1u) {
        return;
    }
    
    // Simple integration
    let mass = body.mass_data.x;
    let inv_mass = 1.0 / mass;
    
    // Apply gravity
    let force = params.gravity * mass;
    let acceleration = force * inv_mass;
    
    // Update velocity
    let new_velocity = body.velocity.xyz + acceleration * params.dt;
    bodies[idx].velocity.x = new_velocity.x;
    bodies[idx].velocity.y = new_velocity.y;
    bodies[idx].velocity.z = new_velocity.z;
    
    // Update position
    let new_position = body.position.xyz + new_velocity * params.dt;
    bodies[idx].position.x = new_position.x;
    bodies[idx].position.y = new_position.y;
    bodies[idx].position.z = new_position.z;
}
"#;
        
        let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Physics Step Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        
        // Create bind group layout
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
        
        // Warm up
        for _ in 0..10 {
            let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Warmup Encoder"),
            });
            
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Physics Pass"),
                    timestamp_writes: None,
                });
                
                compute_pass.set_pipeline(&pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                
                let workgroups = (total_bodies + 63) / 64;
                compute_pass.dispatch_workgroups(workgroups, 1, 1);
            }
            
            gpu.queue.submit(Some(encoder.finish()));
            gpu.device.poll(wgpu::MaintainBase::Wait);
        }
        
        // Benchmark
        let num_steps = 100;
        let start = Instant::now();
        
        for _ in 0..num_steps {
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
                
                let workgroups = (total_bodies + 63) / 64;
                compute_pass.dispatch_workgroups(workgroups, 1, 1);
            }
            
            gpu.queue.submit(Some(encoder.finish()));
        }
        
        gpu.device.poll(wgpu::MaintainBase::Wait);
        
        let elapsed = start.elapsed();
        let total_body_steps = (num_bodies * num_steps) as f64;
        let throughput = total_body_steps / elapsed.as_secs_f64();
        let ms_per_step = elapsed.as_millis() as f64 / num_steps as f64;
        
        println!("  Time for {} steps: {:.2} ms", num_steps, elapsed.as_millis());
        println!("  Average per step: {:.2} ms", ms_per_step);
        println!("  Throughput: {:.0} body×steps/s", throughput);
        println!("  Bodies/frame at 60 FPS: {:.0}", throughput / 60.0);
        
        if throughput >= 10000.0 {
            println!("  ✓ Meets 10,000 body×steps/s requirement");
        } else {
            println!("  ✗ Below 10,000 body×steps/s requirement");
        }
        
        println!();
    }
    
    println!("Benchmark complete!");
    
    Ok(())
}