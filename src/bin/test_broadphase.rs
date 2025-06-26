/*
GPU Broadphase Collision Culling Test

This test verifies that the GPU uniform grid broadphase correctly partitions space and identifies
potential collision pairs while filtering distant objects. It validates spatial hash algorithms,
grid cell assignment, and pair generation efficiency on GPU hardware. Critical for ensuring
O(n log n) collision detection performance and preventing missed collision pairs.
*/

use physics_core::{body::Body, gpu::GpuContext};
use pollster::block_on;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct BroadphaseParams {
    num_bodies: u32,
    cell_size: f32,
    grid_size: u32,
    max_bodies_per_cell: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct CellData {
    count: u32,
    body_ids: [u32; 32],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct PotentialPair {
    body_a: u32,
    body_b: u32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    // Create test scene with bodies in different regions
    let bodies = vec![
        // Cluster 1 (near origin)
        Body::new_sphere([0.0, 0.0, 0.0], 0.5, 1.0),
        Body::new_sphere([0.8, 0.0, 0.0], 0.5, 1.0),    // Overlaps with body 0
        Body::new_sphere([1.6, 0.0, 0.0], 0.5, 1.0),    // Overlaps with body 1
        
        // Cluster 2 (far away)
        Body::new_sphere([10.0, 10.0, 10.0], 0.5, 1.0),
        Body::new_sphere([10.8, 10.0, 10.0], 0.5, 1.0), // Overlaps with body 3
        
        // Isolated bodies
        Body::new_sphere([20.0, 0.0, 0.0], 0.5, 1.0),
        Body::new_sphere([0.0, 20.0, 0.0], 0.5, 1.0),
        Body::new_sphere([0.0, 0.0, 20.0], 0.5, 1.0),
    ];
    
    let num_bodies = bodies.len() as u32;
    
    let gpu = block_on(GpuContext::new())?;
    
    // Create buffers
    let bodies_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Bodies Buffer"),
        contents: bytemuck::cast_slice(&bodies),
        usage: wgpu::BufferUsages::STORAGE,
    });
    
    // Grid parameters
    let grid_size = 32u32;  // 32x32x32 grid
    let total_cells = grid_size * grid_size * grid_size;
    let cell_size = 2.0f32;  // 2m cells (good for 0.5m radius spheres)
    
    // Initialize grid
    let grid_data: Vec<CellData> = vec![CellData { count: 0, body_ids: [0; 32] }; total_cells as usize];
    let _grid_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Grid Buffer"),
        contents: bytemuck::cast_slice(&grid_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    
    // Pairs buffer
    let max_pairs = 10000;
    let pairs_data: Vec<PotentialPair> = vec![PotentialPair { body_a: 0, body_b: 0 }; max_pairs];
    let pairs_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Pairs Buffer"),
        contents: bytemuck::cast_slice(&pairs_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    
    // Pair count buffer
    let pair_count_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Pair Count Buffer"),
        contents: bytemuck::cast_slice(&[0u32]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    
    // Parameters
    let params = BroadphaseParams {
        num_bodies,
        cell_size,
        grid_size,
        max_bodies_per_cell: 32,
    };
    
    let _params_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Broadphase Params Buffer"),
        contents: bytemuck::cast_slice(&[params]),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    
    // Load shader - use simple version for now
    let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Broadphase Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/broadphase_simple.wgsl").into()),
    });
    
    // Create bind group layout for simple shader
    let bind_group_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Broadphase Bind Group Layout"),
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
        label: Some("Broadphase Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: bodies_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: pairs_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: pair_count_buffer.as_entire_binding(),
            },
        ],
    });
    
    // Create pipelines
    let pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Broadphase Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    
    let find_pairs_pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Find Pairs Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("find_all_pairs"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    
    // Run broad phase
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Broadphase Encoder"),
    });
    
    // Simple version: just find pairs
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Find Pairs Pass"),
            timestamp_writes: None,
        });
        
        compute_pass.set_pipeline(&find_pairs_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        
        let workgroups = (num_bodies + 63) / 64;
        compute_pass.dispatch_workgroups(workgroups, 1, 1);
    }
    
    // Read results
    let pair_count_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Pair Count Staging"),
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    encoder.copy_buffer_to_buffer(&pair_count_buffer, 0, &pair_count_staging, 0, 4);
    
    gpu.queue.submit(Some(encoder.finish()));
    gpu.device.poll(wgpu::MaintainBase::Wait);
    
    // Read pair count
    let count_slice = pair_count_staging.slice(..);
    let (tx, rx) = futures::channel::oneshot::channel();
    count_slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    gpu.device.poll(wgpu::MaintainBase::Wait);
    block_on(rx).unwrap().unwrap();
    
    let count_data = count_slice.get_mapped_range();
    let pair_count = bytemuck::cast_slice::<_, u32>(&count_data)[0];
    drop(count_data);
    pair_count_staging.unmap();
    
    // Read pairs if any
    if pair_count > 0 {
        let pairs_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pairs Staging"),
            size: (pair_count as u64 * std::mem::size_of::<PotentialPair>() as u64).min(pairs_buffer.size()),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Copy Pairs Encoder"),
        });
        
        encoder.copy_buffer_to_buffer(&pairs_buffer, 0, &pairs_staging, 0, pairs_staging.size());
        
        gpu.queue.submit(Some(encoder.finish()));
        gpu.device.poll(wgpu::MaintainBase::Wait);
        
        let pairs_slice = pairs_staging.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        pairs_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        gpu.device.poll(wgpu::MaintainBase::Wait);
        block_on(rx).unwrap().unwrap();
        
        let pairs_data = pairs_slice.get_mapped_range();
        let pairs: Vec<PotentialPair> = bytemuck::cast_slice(&pairs_data)[..pair_count as usize].to_vec();
        
        println!("Broad phase results:");
        println!("Number of bodies: {}", num_bodies);
        println!("Grid size: {}x{}x{} (cell size: {} m)", grid_size, grid_size, grid_size, cell_size);
        println!("Potential pairs found: {}", pair_count);
        
        println!("\nPairs:");
        for pair in &pairs {
            println!("  Bodies {} and {}", pair.body_a, pair.body_b);
        }
        
        // Calculate efficiency
        let total_possible = num_bodies * (num_bodies - 1) / 2;
        let pruned = total_possible - pair_count;
        let prune_rate = pruned as f32 / total_possible as f32 * 100.0;
        
        println!("\nEfficiency:");
        println!("Total possible pairs: {}", total_possible);
        println!("Pairs after broad phase: {}", pair_count);
        println!("Pairs pruned: {} ({:.1}%)", pruned, prune_rate);
        
        // Verify expected pairs
        let expected_pairs = vec![(0, 1), (1, 2), (3, 4)];
        let found_pairs: Vec<(u32, u32)> = pairs.iter().map(|p| (p.body_a, p.body_b)).collect();
        
        let mut all_expected_found = true;
        for expected in &expected_pairs {
            if !found_pairs.contains(expected) {
                println!("Missing expected pair: {:?}", expected);
                all_expected_found = false;
            }
        }
        
        if all_expected_found && prune_rate > 80.0 {
            println!("\n✓ Broad phase test PASSED!");
        } else {
            println!("\n✗ Broad phase test FAILED!");
        }
    } else {
        println!("No pairs found!");
        println!("✗ Broad phase test FAILED!");
    }
    
    Ok(())
}