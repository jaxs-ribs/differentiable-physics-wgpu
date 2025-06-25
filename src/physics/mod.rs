mod simulation_parameters;
mod gpu_pipeline;
mod buffer_manager;
mod compute_executor;

pub use simulation_parameters::SimulationParameters;

use crate::{body::Body, gpu::GpuContext};
use self::{
    gpu_pipeline::GpuPipeline,
    buffer_manager::BufferManager,
    compute_executor::ComputeExecutor,
};

pub struct PhysicsEngine {
    gpu: GpuContext,
    pipeline: GpuPipeline,
    buffer_manager: BufferManager,
    compute_executor: ComputeExecutor,
    num_bodies: u32,
}

impl PhysicsEngine {
    pub async fn new(bodies: Vec<Body>) -> Result<Self, Box<dyn std::error::Error>> {
        let gpu = GpuContext::new().await?;
        let num_bodies = bodies.len() as u32;
        
        let parameters = SimulationParameters::default(num_bodies);
        let buffer_manager = BufferManager::new(&gpu, &bodies, &parameters)?;
        let pipeline = GpuPipeline::new(&gpu, &buffer_manager)?;
        let compute_executor = ComputeExecutor::new();
        
        Ok(Self {
            gpu,
            pipeline,
            buffer_manager,
            compute_executor,
            num_bodies,
        })
    }
    
    pub fn step(&self) {
        self.compute_executor.execute_physics_step(
            &self.gpu,
            &self.pipeline,
            &self.buffer_manager,
            self.num_bodies,
        );
    }
    
    pub async fn read_bodies(&self) -> Vec<Body> {
        self.buffer_manager.read_bodies_from_gpu(&self.gpu, self.num_bodies).await
    }
}