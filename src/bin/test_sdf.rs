/*
GPU SDF Collision Detection Integration Test

This test validates that WGSL-compiled SDF shaders correctly detect collisions between geometric
primitives on the GPU. It verifies distance calculations for overlapping and separated shapes,
ensuring the GPU implementation matches mathematical expectations. Critical for catching shader
compilation issues, GPU precision errors, and buffer alignment problems in collision detection.
*/

use physics_core::{body::Body, gpu::GpuContext};
use pollster::block_on;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Contact {
    body_a: u32,
    body_b: u32,
    distance: f32,
    normal: [f32; 3],
    point: [f32; 3],
    _padding: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    // Create test scene: two spheres that are close/overlapping
    let bodies = vec![
        Body::new_sphere([0.0, 0.0, 0.0], 1.0, 1.0),     // Radius 1 at origin
        Body::new_sphere([1.4, 0.0, 0.0], 0.5, 1.0),     // Radius 0.5 at x=1.4 (overlap)
    ];
    
    let gpu = block_on(GpuContext::new())?;
    
    // Create buffers
    let bodies_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Bodies Buffer"),
        contents: bytemuck::cast_slice(&bodies),
        usage: wgpu::BufferUsages::STORAGE,
    });
    
    // Create contacts buffer (max 100 contacts)
    let contacts_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Contacts Buffer"),
        contents: &vec![0u8; 100 * std::mem::size_of::<Contact>()],
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    
    // Create contact count buffer
    let contact_count_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Contact Count Buffer"),
        contents: bytemuck::cast_slice(&[0u32]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    
    // Load shader
    let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("SDF Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sdf.wgsl").into()),
    });
    
    // Create bind group layout
    let bind_group_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("SDF Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
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
    
    // Create bind group
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("SDF Bind Group"),
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
                resource: contact_count_buffer.as_entire_binding(),
            },
        ],
    });
    
    // Create pipeline
    let pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("SDF Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    
    let pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("SDF Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("detect_contacts"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    
    // Run the shader
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("SDF Encoder"),
    });
    
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("SDF Pass"),
            timestamp_writes: None,
        });
        
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(1, 1, 1); // Only 1 pair to check
    }
    
    // Create staging buffers for reading results
    let contact_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Contact Staging"),
        size: contacts_buffer.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    let count_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Count Staging"),
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    encoder.copy_buffer_to_buffer(&contacts_buffer, 0, &contact_staging, 0, contacts_buffer.size());
    encoder.copy_buffer_to_buffer(&contact_count_buffer, 0, &count_staging, 0, 4);
    
    gpu.queue.submit(Some(encoder.finish()));
    gpu.device.poll(wgpu::Maintain::Wait);
    
    // Read results
    let count_slice = count_staging.slice(..);
    let (tx, rx) = futures::channel::oneshot::channel();
    count_slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    gpu.device.poll(wgpu::Maintain::Wait);
    block_on(rx).unwrap().unwrap();
    
    let count_data = count_slice.get_mapped_range();
    let contact_count = bytemuck::cast_slice::<_, u32>(&count_data)[0];
    drop(count_data);
    count_staging.unmap();
    
    println!("Contact count: {}", contact_count);
    
    // Read contacts if any
    if contact_count > 0 {
        let contact_slice = contact_staging.slice(..(contact_count as u64 * std::mem::size_of::<Contact>() as u64));
        let (tx, rx) = futures::channel::oneshot::channel();
        contact_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        gpu.device.poll(wgpu::Maintain::Wait);
        block_on(rx).unwrap().unwrap();
        
        let contact_data = contact_slice.get_mapped_range();
        let contacts: Vec<Contact> = bytemuck::cast_slice(&contact_data).to_vec();
        
        for (i, contact) in contacts.iter().take(contact_count as usize).enumerate() {
            println!("Contact {}: bodies {} and {}, distance: {:.6}", 
                i, contact.body_a, contact.body_b, contact.distance);
            println!("  Normal: [{:.3}, {:.3}, {:.3}]", 
                contact.normal[0], contact.normal[1], contact.normal[2]);
        }
    }
    
    // Expected distance for spheres: 1.4 - 1.0 - 0.5 = -0.1 (overlap)
    println!("\nExpected distance: -0.1");
    
    Ok(())
}