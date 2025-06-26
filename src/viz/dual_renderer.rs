use crate::{body::Body, gpu::GpuContext};
use wgpu::util::DeviceExt;
use super::{
    camera::Camera,
    uniforms::ViewProjectionUniform,
    window::WindowManager,
    wireframe_geometry::WireframeGeometry,
};

const MAX_AABB_COUNT: usize = 1000;
const VERTEX_BUFFER_SIZE: u64 = 12 * 2 * 6 * 4 * MAX_AABB_COUNT as u64;
const CLEAR_COLOR: wgpu::Color = wgpu::Color {
    r: 0.1,
    g: 0.1,
    b: 0.1,
    a: 1.0,
};

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ColorUniform {
    color: [f32; 4],
}

pub struct DualRenderer {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,
    
    // Oracle (CPU) state buffers and resources
    oracle_line_buffer: wgpu::Buffer,
    oracle_color_buffer: wgpu::Buffer,
    oracle_bind_group: wgpu::BindGroup,
    oracle_vertex_count: u32,
    
    // GPU state buffers and resources
    gpu_line_buffer: wgpu::Buffer,
    gpu_color_buffer: wgpu::Buffer,
    gpu_bind_group: wgpu::BindGroup,
    gpu_vertex_count: u32,
    
    // Shared resources
    view_projection_buffer: wgpu::Buffer,
    camera: Camera,
    
    // Debug flags
    show_aabbs: bool,
    show_contacts: bool,
}

impl DualRenderer {
    pub async fn new(window_manager: &WindowManager, gpu: &GpuContext) -> Result<Self, Box<dyn std::error::Error>> {
        let surface = Self::create_surface(window_manager, gpu)?;
        let config = Self::create_surface_config(&surface, gpu, window_manager)?;
        surface.configure(&gpu.device, &config);
        
        let camera = Camera::new(config.width as f32 / config.height as f32);
        let view_projection_buffer = Self::create_uniform_buffer(gpu, &camera);
        
        // Create line buffers for both scenes
        let oracle_line_buffer = Self::create_line_buffer(gpu, "Oracle Line Buffer");
        let gpu_line_buffer = Self::create_line_buffer(gpu, "GPU Line Buffer");
        
        // Create color uniforms - green/transparent for oracle, red/opaque for GPU
        let oracle_color = ColorUniform { color: [0.0, 1.0, 0.0, 0.5] }; // Green, semi-transparent
        let gpu_color = ColorUniform { color: [1.0, 0.0, 0.0, 1.0] }; // Red, opaque
        
        let oracle_color_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Oracle Color Buffer"),
            contents: bytemuck::cast_slice(&[oracle_color]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        let gpu_color_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GPU Color Buffer"),
            contents: bytemuck::cast_slice(&[gpu_color]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        let bind_group_layout = Self::create_bind_group_layout(gpu);
        
        // Create bind groups for both scenes
        let oracle_bind_group = Self::create_bind_group(
            gpu,
            &bind_group_layout,
            &view_projection_buffer,
            &oracle_color_buffer,
            "Oracle"
        );
        let gpu_bind_group = Self::create_bind_group(
            gpu,
            &bind_group_layout,
            &view_projection_buffer,
            &gpu_color_buffer,
            "GPU"
        );
        
        let shader = Self::create_dual_shader(&gpu.device);
        let render_pipeline = Self::create_render_pipeline(gpu, &shader, &bind_group_layout, config.format);
        
        Ok(Self {
            surface,
            config,
            render_pipeline,
            oracle_line_buffer,
            oracle_color_buffer,
            oracle_bind_group,
            oracle_vertex_count: 0,
            gpu_line_buffer,
            gpu_color_buffer,
            gpu_bind_group,
            gpu_vertex_count: 0,
            view_projection_buffer,
            camera,
            show_aabbs: false,
            show_contacts: false,
        })
    }
    
    pub fn update_scenes(&mut self, gpu: &GpuContext, oracle_bodies: Option<&[Body]>, gpu_bodies: Option<&[Body]>) {
        // Update oracle scene
        if let Some(bodies) = oracle_bodies {
            let vertices = WireframeGeometry::generate_vertices_from_bodies(bodies);
            self.oracle_vertex_count = vertices.len() as u32;
            if !vertices.is_empty() {
                gpu.queue.write_buffer(&self.oracle_line_buffer, 0, bytemuck::cast_slice(&vertices));
            }
        }
        
        // Update GPU scene
        if let Some(bodies) = gpu_bodies {
            let vertices = WireframeGeometry::generate_vertices_from_bodies(bodies);
            self.gpu_vertex_count = vertices.len() as u32;
            if !vertices.is_empty() {
                gpu.queue.write_buffer(&self.gpu_line_buffer, 0, bytemuck::cast_slice(&vertices));
            }
        }
    }
    
    pub fn render(&self, gpu: &GpuContext) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Dual Render Encoder"),
        });
        
        self.encode_dual_render_pass(&mut encoder, &view);
        
        gpu.queue.submit(Some(encoder.finish()));
        output.present();
        
        Ok(())
    }
    
    pub fn resize(&mut self, gpu: &GpuContext, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&gpu.device, &self.config);
            
            self.camera.update_aspect_ratio(self.config.width as f32 / self.config.height as f32);
            self.update_uniform_buffer(gpu);
        }
    }
    
    pub fn set_debug_aabbs(&mut self, enabled: bool) {
        self.show_aabbs = enabled;
    }
    
    pub fn set_debug_contacts(&mut self, enabled: bool) {
        self.show_contacts = enabled;
    }
    
    fn create_surface(window_manager: &WindowManager, gpu: &GpuContext) -> Result<wgpu::Surface<'static>, Box<dyn std::error::Error>> {
        unsafe {
            let surface = gpu.instance.create_surface_unsafe(
                wgpu::SurfaceTargetUnsafe::from_window(window_manager.window())?
            )?;
            Ok(std::mem::transmute::<wgpu::Surface<'_>, wgpu::Surface<'static>>(surface))
        }
    }
    
    fn create_surface_config(
        surface: &wgpu::Surface,
        gpu: &GpuContext,
        window_manager: &WindowManager,
    ) -> Result<wgpu::SurfaceConfiguration, Box<dyn std::error::Error>> {
        let surface_caps = surface.get_capabilities(&gpu.adapter);
        let surface_format = surface_caps.formats.iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);
        
        let size = window_manager.inner_size();
        
        Ok(wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        })
    }
    
    fn create_uniform_buffer(gpu: &GpuContext, camera: &Camera) -> wgpu::Buffer {
        let uniform_data = ViewProjectionUniform::new(camera.view_projection_matrix_transposed());
        
        gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("View Projection Buffer"),
            contents: bytemuck::cast_slice(&[uniform_data]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }
    
    fn create_line_buffer(gpu: &GpuContext, label: &str) -> wgpu::Buffer {
        gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: VERTEX_BUFFER_SIZE,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }
    
    fn create_bind_group_layout(gpu: &GpuContext) -> wgpu::BindGroupLayout {
        gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Dual Renderer Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
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
        layout: &wgpu::BindGroupLayout,
        view_projection_buffer: &wgpu::Buffer,
        color_buffer: &wgpu::Buffer,
        label: &str,
    ) -> wgpu::BindGroup {
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{} Bind Group", label)),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: view_projection_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: color_buffer.as_entire_binding(),
                },
            ],
        })
    }
    
    fn create_dual_shader(device: &wgpu::Device) -> wgpu::ShaderModule {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Dual Renderer Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/dual_debug.wgsl").into()),
        })
    }
    
    fn create_render_pipeline(
        gpu: &GpuContext,
        shader: &wgpu::ShaderModule,
        bind_group_layout: &wgpu::BindGroupLayout,
        surface_format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        let pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Dual Renderer Pipeline Layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });
        
        gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Dual Renderer Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                buffers: &[Self::vertex_buffer_layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        })
    }
    
    fn vertex_buffer_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: 6 * 4, // 3 position + 3 color floats
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 3 * 4,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
    
    fn encode_dual_render_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Dual Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(CLEAR_COLOR),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        
        render_pass.set_pipeline(&self.render_pipeline);
        
        // Draw oracle scene (green/transparent)
        if self.oracle_vertex_count > 0 {
            render_pass.set_bind_group(0, &self.oracle_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.oracle_line_buffer.slice(..));
            render_pass.draw(0..self.oracle_vertex_count, 0..1);
        }
        
        // Draw GPU scene (red/opaque)
        if self.gpu_vertex_count > 0 {
            render_pass.set_bind_group(0, &self.gpu_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.gpu_line_buffer.slice(..));
            render_pass.draw(0..self.gpu_vertex_count, 0..1);
        }
    }
    
    fn update_uniform_buffer(&self, gpu: &GpuContext) {
        let uniform_data = ViewProjectionUniform::new(self.camera.view_projection_matrix_transposed());
        gpu.queue.write_buffer(&self.view_projection_buffer, 0, bytemuck::cast_slice(&[uniform_data]));
    }
}