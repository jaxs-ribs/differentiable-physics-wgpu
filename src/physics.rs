use crate::{body::Body, gpu::GpuContext};
use bytemuck::{Pod, Zeroable};
use wgpu::{util::DeviceExt, BindGroup, Buffer, ComputePipeline};

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

// Verify size
const _: () = assert!(std::mem::size_of::<SimParams>() == 32);

pub struct PhysicsEngine {
    gpu: GpuContext,
    bodies_buffer: Buffer,
    params_buffer: Buffer,
    bind_group: BindGroup,
    integrator_pipeline: ComputePipeline,
    num_bodies: u32,
}

impl PhysicsEngine {
    pub async fn new(bodies: Vec<Body>) -> Result<Self, Box<dyn std::error::Error>> {
        let gpu = GpuContext::new().await?;
        let num_bodies = bodies.len() as u32;
        
        // Create bodies buffer
        let bodies_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bodies Buffer"),
            contents: bytemuck::cast_slice(&bodies),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        
        // Create params buffer
        let params = SimParams {
            dt: 0.016,
            gravity_x: 0.0,
            gravity_y: -9.81,
            gravity_z: 0.0,
            num_bodies,
            _padding: [0.0; 3],
        };
        
        let params_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        // Load shader
        let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Physics Step Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/physics_step.wgsl").into()),
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
        
        // Create bind group
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
        
        // Create compute pipeline
        let pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Physics Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let integrator_pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Integrator Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("physics_step"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        
        Ok(Self {
            gpu,
            bodies_buffer,
            params_buffer,
            bind_group,
            integrator_pipeline,
            num_bodies,
        })
    }
    
    pub fn step(&self) {
        let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Physics Step Encoder"),
        });
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Integrator Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.integrator_pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);
            
            let workgroups = (self.num_bodies + 63) / 64;
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }
        
        self.gpu.queue.submit(Some(encoder.finish()));
        self.gpu.device.poll(wgpu::Maintain::Wait);
    }
    
    pub async fn read_bodies(&self) -> Vec<Body> {
        // Create staging buffer
        let staging_buffer = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (self.num_bodies as u64) * std::mem::size_of::<Body>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Copy from GPU to staging
        let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Copy Encoder"),
        });
        
        encoder.copy_buffer_to_buffer(
            &self.bodies_buffer,
            0,
            &staging_buffer,
            0,
            staging_buffer.size(),
        );
        
        self.gpu.queue.submit(Some(encoder.finish()));
        
        // Map and read
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        
        self.gpu.device.poll(wgpu::Maintain::Wait);
        rx.await.unwrap().unwrap();
        
        let data = buffer_slice.get_mapped_range();
        let bodies: Vec<Body> = bytemuck::cast_slice(&data).to_vec();
        
        drop(data);
        staging_buffer.unmap();
        
        bodies
    }
}