use crate::{gpu::GpuContext, body::Body};
use pollster::block_on;

#[cfg(feature = "viz")]
use crate::viz::Visualizer;

#[cfg(feature = "viz")]
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
};

pub struct TestHarness {
    pub gpu: GpuContext,
}

impl TestHarness {
    pub async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let gpu = GpuContext::new().await?;
        Ok(Self { gpu })
    }
    
    pub fn new_blocking() -> Result<Self, Box<dyn std::error::Error>> {
        block_on(Self::new())
    }
}

#[cfg(feature = "viz")]
pub struct VisualizationTestHarness {
    pub gpu: GpuContext,
    pub event_loop: EventLoop<()>,
    pub visualizer: Visualizer,
}

#[cfg(feature = "viz")]
impl VisualizationTestHarness {
    pub async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let event_loop = EventLoop::new()?;
        let gpu = GpuContext::new().await?;
        let visualizer = Visualizer::new(&event_loop, &gpu).await?;
        
        Ok(Self {
            gpu,
            event_loop,
            visualizer,
        })
    }
    
    pub fn new_blocking() -> Result<Self, Box<dyn std::error::Error>> {
        block_on(Self::new())
    }
    
    pub fn run_visualization<F>(mut self, bodies: Vec<Body>, mut update_fn: F) -> Result<(), Box<dyn std::error::Error>>
    where
        F: FnMut(&mut Vec<Body>) + 'static,
    {
        let mut bodies = bodies;
        let gpu = self.gpu;
        let mut viz = self.visualizer;
        
        self.event_loop.run(move |event, control_flow| {
            match event {
                Event::WindowEvent {
                    ref event,
                    window_id,
                } if window_id == viz.window().id() => match event {
                    WindowEvent::CloseRequested => {
                        control_flow.exit();
                    },
                    WindowEvent::Resized(physical_size) => {
                        viz.resize(&gpu, *physical_size);
                    },
                    WindowEvent::RedrawRequested => {
                        update_fn(&mut bodies);
                        viz.update_bodies(&gpu, &bodies);
                        let vertex_count = bodies.len() as u32 * 24;
                        
                        match viz.render(&gpu, vertex_count) {
                            Ok(_) => {},
                            Err(wgpu::SurfaceError::Lost) => viz.resize(&gpu, viz.window().inner_size()),
                            Err(wgpu::SurfaceError::OutOfMemory) => control_flow.exit(),
                            Err(e) => eprintln!("Render error: {:?}", e),
                        }
                        
                        viz.window().request_redraw();
                    },
                    _ => {}
                },
                _ => {}
            }
        })?;
        
        Ok(())
    }
}

// Common GPU pipeline setup helpers
pub mod pipeline_helpers {
    use wgpu::util::DeviceExt;
    use crate::test_utils::simulation_params::TestSimulationParams;
    
    pub struct ComputePipelineSetup {
        pub bind_group: wgpu::BindGroup,
        pub pipeline: wgpu::ComputePipeline,
        pub bodies_buffer: wgpu::Buffer,
        pub params_buffer: wgpu::Buffer,
    }
    
    pub fn create_physics_compute_pipeline(
        device: &wgpu::Device,
        shader_source: &str,
        bodies: &[crate::body::Body],
        params: &TestSimulationParams,
    ) -> Result<ComputePipelineSetup, Box<dyn std::error::Error>> {
        // Create buffers
        let bodies_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bodies Buffer"),
            contents: bytemuck::cast_slice(bodies),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params Buffer"),
            contents: bytemuck::cast_slice(&[*params]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        
        // Create shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Test Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        
        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Test Bind Group Layout"),
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
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Test Bind Group"),
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
        
        // Create pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Test Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Test Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        
        Ok(ComputePipelineSetup {
            bind_group,
            pipeline,
            bodies_buffer,
            params_buffer,
        })
    }
}