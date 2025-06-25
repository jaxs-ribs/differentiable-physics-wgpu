use crate::gpu::GpuContext;
use super::buffer_manager::BufferManager;
use wgpu::{BindGroup, BindGroupLayout, ComputePipeline};

const PHYSICS_SHADER_PATH: &str = "shaders/physics_step.wgsl";

pub struct GpuPipeline {
    bind_group: BindGroup,
    compute_pipeline: ComputePipeline,
}

impl GpuPipeline {
    pub fn new(gpu: &GpuContext, buffer_manager: &BufferManager) -> Result<Self, Box<dyn std::error::Error>> {
        let shader = Self::load_shader(gpu);
        let bind_group_layout = Self::create_bind_group_layout(gpu);
        let bind_group = Self::create_bind_group(gpu, &bind_group_layout, buffer_manager);
        let compute_pipeline = Self::create_compute_pipeline(gpu, &shader, &bind_group_layout);
        
        Ok(Self {
            bind_group,
            compute_pipeline,
        })
    }
    
    pub fn bind_group(&self) -> &BindGroup {
        &self.bind_group
    }
    
    pub fn compute_pipeline(&self) -> &ComputePipeline {
        &self.compute_pipeline
    }
    
    fn load_shader(gpu: &GpuContext) -> wgpu::ShaderModule {
        crate::shaders::load_shader(&gpu.device, "physics_step")
    }
    
    fn create_bind_group_layout(gpu: &GpuContext) -> BindGroupLayout {
        gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        })
    }
    
    fn create_bind_group(
        gpu: &GpuContext,
        layout: &BindGroupLayout,
        buffer_manager: &BufferManager,
    ) -> BindGroup {
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Physics Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_manager.bodies_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_manager.params_buffer().as_entire_binding(),
                },
            ],
        })
    }
    
    fn create_compute_pipeline(
        gpu: &GpuContext,
        shader: &wgpu::ShaderModule,
        bind_group_layout: &BindGroupLayout,
    ) -> ComputePipeline {
        let pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Physics Pipeline Layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });
        
        gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Integrator Pipeline"),
            layout: Some(&pipeline_layout),
            module: shader,
            entry_point: Some("physics_step"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    }
}