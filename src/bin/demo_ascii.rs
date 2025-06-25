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
    _padding: [f32; 4],
}

fn create_demo_scene() -> Vec<Body> {
    let mut bodies = Vec::new();
    
    // Create falling spheres in a row
    for i in 0..5 {
        let x = (i as f32 - 2.0) * 3.0;
        bodies.push(Body::new_sphere([x, 15.0 + i as f32, 0.0], 0.5, 1.0));
    }
    
    // Add some static obstacles
    bodies.push(Body::new_static_sphere([-3.0, 8.0, 0.0], 0.5));
    bodies.push(Body::new_static_sphere([3.0, 8.0, 0.0], 0.5));
    
    // Ground plane
    bodies.push(Body::new_static_box([0.0, 0.0, 0.0], [20.0, 0.5, 20.0]));
    
    bodies
}

fn render_ascii(bodies: &[Body], width: usize, height: usize) {
    // Clear screen (ANSI escape code)
    print!("\x1B[2J\x1B[1;1H");
    
    // Create ASCII buffer
    let mut buffer = vec![vec![' '; width]; height];
    
    // World to screen transformation
    let scale = 2.0; // 2 chars per meter
    let offset_x = width as f32 / 2.0;
    let offset_y = height as f32 - 2.0;
    
    // Draw bodies
    for (i, body) in bodies.iter().enumerate() {
        let x = (body.position[0] * scale + offset_x) as i32;
        let y = (offset_y - body.position[1] * scale) as i32;
        
        if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
            let symbol = if body.shape_data[1] == 1 {
                // Static body
                if body.shape_data[0] == 2 { '=' } else { '*' }
            } else {
                // Dynamic body - use number
                char::from_digit(i as u32, 10).unwrap_or('o')
            };
            buffer[y as usize][x as usize] = symbol;
        }
    }
    
    // Draw frame
    println!("┌{}┐", "─".repeat(width));
    for row in buffer {
        print!("│");
        for ch in row {
            print!("{}", ch);
        }
        println!("│");
    }
    println!("└{}┘", "─".repeat(width));
    
    // Legend
    println!("\nPhysics Engine ASCII Demo");
    println!("0-4: Dynamic spheres, *: Static spheres, =: Ground");
    println!("Press Ctrl+C to exit");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
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
        _padding: [0.0; 4],
    };
    
    let _params_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
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
    
    println!("Starting ASCII physics visualization...\n");
    std::thread::sleep(std::time::Duration::from_secs(1));
    
    let start_time = Instant::now();
    let dt = 0.016; // 60 FPS
    
    loop {
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
        
        // Render
        render_ascii(&bodies, 80, 24);
        
        // Show time and first body info
        let elapsed = start_time.elapsed().as_secs_f32();
        println!("\nTime: {:.1}s", elapsed);
        println!("Body 0: y={:.2}m, vy={:.2}m/s", bodies[0].position[1], bodies[0].velocity[1]);
        
        // Sleep to maintain frame rate
        std::thread::sleep(std::time::Duration::from_millis(100)); // 10 FPS for visibility
        
        // Exit after 10 seconds
        if elapsed > 10.0 {
            println!("\nDemo complete!");
            break;
        }
    }
    
    Ok(())
}