use crate::{body::Body, gpu::GpuContext};
use super::simulation_parameters::SimulationParameters;
use wgpu::{util::DeviceExt, Buffer};

pub struct BufferManager {
    bodies_buffer: Buffer,
    params_buffer: Buffer,
}

impl BufferManager {
    pub fn new(
        gpu: &GpuContext,
        bodies: &[Body],
        parameters: &SimulationParameters,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let bodies_buffer = Self::create_bodies_buffer(gpu, bodies);
        let params_buffer = Self::create_params_buffer(gpu, parameters);
        
        Ok(Self {
            bodies_buffer,
            params_buffer,
        })
    }
    
    pub fn bodies_buffer(&self) -> &Buffer {
        &self.bodies_buffer
    }
    
    pub fn params_buffer(&self) -> &Buffer {
        &self.params_buffer
    }
    
    pub async fn read_bodies_from_gpu(&self, gpu: &GpuContext, num_bodies: u32) -> Vec<Body> {
        let staging_buffer = self.create_staging_buffer(gpu, num_bodies);
        self.copy_to_staging_buffer(gpu, &staging_buffer);
        self.read_from_staging_buffer(gpu, &staging_buffer).await
    }
    
    fn create_bodies_buffer(gpu: &GpuContext, bodies: &[Body]) -> Buffer {
        gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bodies Buffer"),
            contents: bytemuck::cast_slice(bodies),
            usage: wgpu::BufferUsages::STORAGE 
                | wgpu::BufferUsages::COPY_SRC 
                | wgpu::BufferUsages::COPY_DST,
        })
    }
    
    fn create_params_buffer(gpu: &GpuContext, parameters: &SimulationParameters) -> Buffer {
        gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params Buffer"),
            contents: bytemuck::cast_slice(&[*parameters]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }
    
    fn create_staging_buffer(&self, gpu: &GpuContext, num_bodies: u32) -> Buffer {
        let buffer_size = (num_bodies as u64) * std::mem::size_of::<Body>() as u64;
        
        gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }
    
    fn copy_to_staging_buffer(&self, gpu: &GpuContext, staging_buffer: &Buffer) {
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Copy Encoder"),
        });
        
        encoder.copy_buffer_to_buffer(
            &self.bodies_buffer,
            0,
            staging_buffer,
            0,
            staging_buffer.size(),
        );
        
        gpu.queue.submit(Some(encoder.finish()));
    }
    
    async fn read_from_staging_buffer(&self, gpu: &GpuContext, staging_buffer: &Buffer) -> Vec<Body> {
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        
        gpu.device.poll(wgpu::Maintain::Wait);
        rx.await.unwrap().unwrap();
        
        let data = buffer_slice.get_mapped_range();
        let bodies: Vec<Body> = bytemuck::cast_slice(&data).to_vec();
        
        drop(data);
        staging_buffer.unmap();
        
        bodies
    }
}