use crate::gpu::GpuContext;
use super::{gpu_pipeline::GpuPipeline, buffer_manager::BufferManager};

const WORKGROUP_SIZE: u32 = 64;

pub struct ComputeExecutor;

impl ComputeExecutor {
    pub fn new() -> Self {
        Self
    }
    
    pub fn execute_physics_step(
        &self,
        gpu: &GpuContext,
        pipeline: &GpuPipeline,
        _buffer_manager: &BufferManager,
        num_bodies: u32,
    ) {
        let mut encoder = self.create_command_encoder(gpu);
        self.dispatch_compute_pass(&mut encoder, pipeline, num_bodies);
        self.submit_and_wait(gpu, encoder);
    }
    
    fn create_command_encoder(&self, gpu: &GpuContext) -> wgpu::CommandEncoder {
        gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Physics Step Encoder"),
        })
    }
    
    fn dispatch_compute_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &GpuPipeline,
        num_bodies: u32,
    ) {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Integrator Pass"),
            timestamp_writes: None,
        });
        
        compute_pass.set_pipeline(pipeline.compute_pipeline());
        compute_pass.set_bind_group(0, pipeline.bind_group(), &[]);
        
        let workgroups = self.calculate_workgroups(num_bodies);
        compute_pass.dispatch_workgroups(workgroups, 1, 1);
    }
    
    fn calculate_workgroups(&self, num_bodies: u32) -> u32 {
        (num_bodies + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
    }
    
    fn submit_and_wait(&self, gpu: &GpuContext, encoder: wgpu::CommandEncoder) {
        gpu.queue.submit(Some(encoder.finish()));
        gpu.device.poll(wgpu::MaintainBase::Wait);
    }
}