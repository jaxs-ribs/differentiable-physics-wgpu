use physics_core::{body::Body, gpu::GpuContext, contact::Contact};
use pollster::block_on;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct SolverParams {
    values: [f32; 4],      // dt, stiffness, damping, restitution
    counts: [u32; 4],      // num_contacts, 0, 0, 0
    _padding: [f32; 4],
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    println!("Size of Contact: {} bytes", std::mem::size_of::<Contact>());
    println!("Size of SolverParams: {} bytes", std::mem::size_of::<SolverParams>());
    
    // Check if padding is correctly sized
    let expected_solver_size = 16 + 4 + 12; // vec4 + u32 + vec3<u32>
    println!("Expected SolverParams size: {} bytes", expected_solver_size);
    
    // Test case: Two spheres in collision
    let mut bodies = vec![
        Body::new_sphere([0.0, 1.0, 0.0], 0.5, 1.0),   // Falling sphere
        Body::new_static_sphere([0.0, 0.0, 0.0], 0.5), // Ground sphere
    ];
    
    // Set initial velocity for falling sphere
    bodies[0].velocity[1] = -2.0; // Falling down
    
    // Create a contact representing penetration
    // Normal should point from A to B (from falling sphere to ground)
    let contact = Contact {
        body_a: 0,
        body_b: 1,
        distance: -0.1, // 0.1 units of penetration
        _padding1: 0.0,
        normal: [0.0, -1.0, 0.0, 0.0], // Normal pointing down (from A to B)
        point: [0.0, 0.45, 0.0, 0.0], // Contact point (vec4)
    };
    
    let gpu = block_on(GpuContext::new())?;
    
    // Create buffers
    let bodies_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Bodies Buffer"),
        contents: bytemuck::cast_slice(&bodies),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    });
    
    let contacts_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Contacts Buffer"),
        contents: bytemuck::cast_slice(&[contact]),
        usage: wgpu::BufferUsages::STORAGE,
    });
    
    let params = SolverParams {
        values: [0.016, 1000.0, 10.0, 0.5], // dt, stiffness, damping, restitution
        counts: [1, 0, 0, 0], // num_contacts
        _padding: [0.0; 4],
    };
    
    let params_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Solver Params Buffer"),
        contents: bytemuck::cast_slice(&[params]),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    
    // Load shader
    let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Contact Solver Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/contact_solver.wgsl").into()),
    });
    
    // Create bind group layout
    let bind_group_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Solver Bind Group Layout"),
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
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
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
    
    // Create bind group
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Solver Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: bodies_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: contacts_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });
    
    // Create pipeline
    let pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Solver Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    
    let pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Solver Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("solve_contacts"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    
    // Run the solver
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Solver Encoder"),
    });
    
    // Copy initial state for comparison
    let initial_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Initial Staging"),
        size: bodies_buffer.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    encoder.copy_buffer_to_buffer(&bodies_buffer, 0, &initial_staging, 0, bodies_buffer.size());
    
    // Run solver
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Solver Pass"),
            timestamp_writes: None,
        });
        
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(1, 1, 1);
    }
    
    // Copy result
    let result_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Result Staging"),
        size: bodies_buffer.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    encoder.copy_buffer_to_buffer(&bodies_buffer, 0, &result_staging, 0, bodies_buffer.size());
    
    gpu.queue.submit(Some(encoder.finish()));
    gpu.device.poll(wgpu::Maintain::Wait);
    
    // Read initial state
    let initial_slice = initial_staging.slice(..);
    let (tx, rx) = futures::channel::oneshot::channel();
    initial_slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    gpu.device.poll(wgpu::Maintain::Wait);
    block_on(rx).unwrap().unwrap();
    
    let initial_data = initial_slice.get_mapped_range();
    let initial_bodies: Vec<Body> = bytemuck::cast_slice(&initial_data).to_vec();
    drop(initial_data);
    initial_staging.unmap();
    
    // Read result
    let result_slice = result_staging.slice(..);
    let (tx, rx) = futures::channel::oneshot::channel();
    result_slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    gpu.device.poll(wgpu::Maintain::Wait);
    block_on(rx).unwrap().unwrap();
    
    let result_data = result_slice.get_mapped_range();
    let result_bodies: Vec<Body> = bytemuck::cast_slice(&result_data).to_vec();
    drop(result_data);
    result_staging.unmap();
    
    // Print results
    println!("Contact solver test results:");
    println!("\nInitial state:");
    println!("Body 0: pos=[{:.3}, {:.3}, {:.3}], vel=[{:.3}, {:.3}, {:.3}]",
        initial_bodies[0].position[0], initial_bodies[0].position[1], initial_bodies[0].position[2],
        initial_bodies[0].velocity[0], initial_bodies[0].velocity[1], initial_bodies[0].velocity[2]);
    
    println!("\nAfter contact resolution:");
    println!("Body 0: pos=[{:.3}, {:.3}, {:.3}], vel=[{:.3}, {:.3}, {:.3}]",
        result_bodies[0].position[0], result_bodies[0].position[1], result_bodies[0].position[2],
        result_bodies[0].velocity[0], result_bodies[0].velocity[1], result_bodies[0].velocity[2]);
    
    // Calculate changes
    let pos_change = result_bodies[0].position[1] - initial_bodies[0].position[1];
    let vel_change = result_bodies[0].velocity[1] - initial_bodies[0].velocity[1];
    
    println!("\nChanges:");
    println!("Position Y change: {:.6}", pos_change);
    println!("Velocity Y change: {:.6}", vel_change);
    
    // Verify physics
    // Debug physics
    let penetration = 0.1;
    let force = 1000.0 * penetration; // 100 N
    let impulse = force * 0.016; // 1.6 N·s
    let expected_vel_change = impulse * 1.0; // inv_mass = 1.0
    
    println!("\nPhysics calculation:");
    println!("Penetration: {:.3} m", penetration);
    println!("Penalty force: {:.1} N", force);
    println!("Impulse: {:.3} N·s", impulse);
    println!("Expected velocity change: +{:.3} m/s", expected_vel_change);
    
    println!("\nExpected behavior:");
    println!("- Velocity should increase (become less negative) due to penalty force");
    println!("- Position should increase slightly due to position correction");
    
    if vel_change > 0.0 && pos_change > 0.0 {
        println!("\n✓ Contact solver test PASSED!");
    } else {
        println!("\n✗ Contact solver test FAILED!");
        println!("Debug: Check shader implementation");
    }
    
    Ok(())
}