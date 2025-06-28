pub mod body;
pub mod camera;
pub mod gpu;
pub mod mesh;
pub mod video;
pub mod loader;

use wgpu::util::DeviceExt;
use camera::Camera;
use gpu::GpuContext;
use mesh::WireframeGeometry;
use body::Body;

const MAX_AABB_COUNT: usize = 1000;
const VERTEX_BUFFER_SIZE: u64 = 12 * 2 * 6 * 4 * MAX_AABB_COUNT as u64;
const CLEAR_COLOR: wgpu::Color = wgpu::Color {
    r: 0.1,  
    g: 0.2,  
    b: 0.3,  
    a: 1.0,
};

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ViewProjectionUniform {
    matrix: [[f32; 4]; 4],
}

impl ViewProjectionUniform {
    fn new(matrix: [[f32; 4]; 4]) -> Self {
        Self { matrix }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ColorUniform {
    color: [f32; 4],
}

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,
    capture_pipeline: Option<wgpu::RenderPipeline>,
    
    // Primary scene buffers (for oracle/primary trace)
    primary_vertex_buffer: wgpu::Buffer,
    primary_color_buffer: wgpu::Buffer,
    primary_bind_group: wgpu::BindGroup,
    primary_vertex_count: u32,
    
    // Secondary scene buffers (for GPU/comparison trace)
    secondary_vertex_buffer: wgpu::Buffer,
    secondary_color_buffer: wgpu::Buffer,
    secondary_bind_group: wgpu::BindGroup,
    secondary_vertex_count: u32,
    
    // Shared resources
    view_projection_buffer: wgpu::Buffer,
    camera: Camera,
    
    // Frame capture resources (for video recording)
    capture_texture: Option<wgpu::Texture>,
    capture_view: Option<wgpu::TextureView>,
    staging_buffer: Option<wgpu::Buffer>,
    bytes_per_row: u32,
}

impl Renderer {
    pub async fn new(
        window: &winit::window::Window,
        gpu: &GpuContext,
        enable_capture: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let surface = unsafe {
            let surface = gpu.instance.create_surface_unsafe(
                wgpu::SurfaceTargetUnsafe::from_window(window)?
            )?;
            std::mem::transmute::<wgpu::Surface<'_>, wgpu::Surface<'static>>(surface)
        };
        
        let surface_caps = surface.get_capabilities(&gpu.adapter);
        let surface_format = surface_caps.formats.iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);
        
        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&gpu.device, &config);
        
        let camera = Camera::new(config.width as f32 / config.height as f32);
        let view_projection_buffer = Self::create_uniform_buffer(gpu, &camera);
        
        // Create vertex buffers for both scenes
        let primary_vertex_buffer = Self::create_vertex_buffer(gpu, "Primary Vertex Buffer");
        let secondary_vertex_buffer = Self::create_vertex_buffer(gpu, "Secondary Vertex Buffer");
        
        // Create color uniforms
        let primary_color = ColorUniform { color: [1.0, 1.0, 1.0, 1.0] };
        let secondary_color = ColorUniform { color: [1.0, 1.0, 1.0, 1.0] };
        
        let primary_color_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Primary Color Buffer"),
            contents: bytemuck::cast_slice(&[primary_color]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        let secondary_color_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Secondary Color Buffer"),
            contents: bytemuck::cast_slice(&[secondary_color]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        let bind_group_layout = Self::create_bind_group_layout(gpu);
        
        let primary_bind_group = Self::create_bind_group(
            gpu,
            &bind_group_layout,
            &view_projection_buffer,
            &primary_color_buffer,
            "Primary"
        );
        let secondary_bind_group = Self::create_bind_group(
            gpu,
            &bind_group_layout,
            &view_projection_buffer,
            &secondary_color_buffer,
            "Secondary"
        );
        
        let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Wireframe Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/wireframe.wgsl").into()),
        });
        
        let render_pipeline = Self::create_render_pipeline(gpu, &shader, &bind_group_layout, config.format);
        
        // Create capture resources if requested
        let (capture_texture, capture_view, staging_buffer, bytes_per_row, capture_pipeline) = if enable_capture {
            let unpadded_bytes_per_row = config.width * 4;
            let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
            let padded_bytes_per_row = ((unpadded_bytes_per_row + align - 1) / align) * align;
            
            let capture_format = config.format;
            
            let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Capture Texture"),
                size: wgpu::Extent3d {
                    width: config.width,
                    height: config.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: capture_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            
            let buffer_size = padded_bytes_per_row as u64 * config.height as u64;
            let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Capture Staging Buffer"),
                size: buffer_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            
            let capture_pipeline = if capture_format != config.format {
                Some(Self::create_render_pipeline(gpu, &shader, &bind_group_layout, capture_format))
            } else {
                None
            };
            
            (Some(texture), Some(view), Some(buffer), padded_bytes_per_row, capture_pipeline)
        } else {
            (None, None, None, 0, None)
        };
        
        Ok(Self {
            surface,
            config,
            render_pipeline,
            capture_pipeline,
            primary_vertex_buffer,
            primary_color_buffer,
            primary_bind_group,
            primary_vertex_count: 0,
            secondary_vertex_buffer,
            secondary_color_buffer,
            secondary_bind_group,
            secondary_vertex_count: 0,
            view_projection_buffer,
            camera,
            capture_texture,
            capture_view,
            staging_buffer,
            bytes_per_row,
        })
    }
    
    pub fn update_scenes(&mut self, gpu: &GpuContext, primary_bodies: Option<&[Body]>, secondary_bodies: Option<&[Body]>) {
        // Update primary scene
        if let Some(bodies) = primary_bodies {
            let vertices = WireframeGeometry::generate_vertices_from_bodies(bodies);
            self.primary_vertex_count = vertices.len() as u32 / 6; // 6 floats per vertex
            if !vertices.is_empty() {
                gpu.queue.write_buffer(&self.primary_vertex_buffer, 0, bytemuck::cast_slice(&vertices));
            }
        } else {
            self.primary_vertex_count = 0;
        }
        
        // Update secondary scene
        if let Some(bodies) = secondary_bodies {
            let vertices = WireframeGeometry::generate_vertices_from_bodies(bodies);
            self.secondary_vertex_count = vertices.len() as u32 / 6;
            if !vertices.is_empty() {
                gpu.queue.write_buffer(&self.secondary_vertex_buffer, 0, bytemuck::cast_slice(&vertices));
            }
        } else {
            self.secondary_vertex_count = 0;
        }
    }
    
    pub fn render(&self, gpu: &GpuContext) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let surface_view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        self.update_uniform_buffer(gpu);

        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        
        if let Some(capture_view) = &self.capture_view {
            self.encode_render_pass(&mut encoder, capture_view, true);
            self.encode_render_pass(&mut encoder, &surface_view, false);
        } else {
            self.encode_render_pass(&mut encoder, &surface_view, false);
        }
        
        let submission_index = gpu.queue.submit(Some(encoder.finish()));
        
        if self.capture_texture.is_some() {
            gpu.device.poll(wgpu::MaintainBase::WaitForSubmissionIndex(submission_index));
        }
        
        output.present();
        
        Ok(())
    }
    
    pub fn capture_frame(&self, gpu: &GpuContext) -> Option<Vec<u8>> {
        let capture_texture = self.capture_texture.as_ref()?;
        let staging_buffer = self.staging_buffer.as_ref()?;
        
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Capture Encoder"),
        });
        
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: capture_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: staging_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(self.bytes_per_row),
                    rows_per_image: Some(self.config.height),
                },
            },
            wgpu::Extent3d {
                width: self.config.width,
                height: self.config.height,
                depth_or_array_layers: 1,
            },
        );
        
        let copy_submission = gpu.queue.submit(Some(encoder.finish()));
        gpu.device.poll(wgpu::MaintainBase::WaitForSubmissionIndex(copy_submission));
        
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        
        gpu.device.poll(wgpu::MaintainBase::Wait);
        
        pollster::block_on(rx).unwrap().ok()?;
        
        let data = buffer_slice.get_mapped_range();
        
        let mut frame_data = Vec::with_capacity((self.config.width * self.config.height * 4) as usize);
        let unpadded_bytes_per_row = self.config.width * 4;
        
        for y in 0..self.config.height {
            let row_start = (y * self.bytes_per_row) as usize;
            let row_end = row_start + unpadded_bytes_per_row as usize;
            let row_data = &data[row_start..row_end];
            frame_data.extend_from_slice(row_data);
        }
        
        drop(data);
        staging_buffer.unmap();
        
        Some(frame_data)
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
    
    pub fn camera_mut(&mut self) -> &mut Camera {
        &mut self.camera
    }
    
    fn create_uniform_buffer(gpu: &GpuContext, camera: &Camera) -> wgpu::Buffer {
        let uniform_data = ViewProjectionUniform::new(camera.view_projection_matrix_transposed());
        
        gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("View Projection Buffer"),
            contents: bytemuck::cast_slice(&[uniform_data]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }
    
    fn create_vertex_buffer(gpu: &GpuContext, label: &str) -> wgpu::Buffer {
        gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: VERTEX_BUFFER_SIZE,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }
    
    fn create_bind_group_layout(gpu: &GpuContext) -> wgpu::BindGroupLayout {
        gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Renderer Bind Group Layout"),
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
    
    fn create_render_pipeline(
        gpu: &GpuContext,
        shader: &wgpu::ShaderModule,
        bind_group_layout: &wgpu::BindGroupLayout,
        surface_format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        let pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Renderer Pipeline Layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });
        
        gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Renderer Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: "vs_main",
                buffers: &[Self::vertex_buffer_layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
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
    
    fn encode_render_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        is_capture: bool,
    ) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Wireframe Render Pass"),
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
        
        let pipeline = if is_capture && self.capture_pipeline.is_some() {
            self.capture_pipeline.as_ref().unwrap()
        } else {
            &self.render_pipeline
        };
        render_pass.set_pipeline(pipeline);
        
        // Draw primary scene
        if self.primary_vertex_count > 0 {
            render_pass.set_bind_group(0, &self.primary_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.primary_vertex_buffer.slice(..));
            render_pass.draw(0..self.primary_vertex_count * 6, 0..1);
        }
        
        // Draw secondary scene
        if self.secondary_vertex_count > 0 {
            render_pass.set_bind_group(0, &self.secondary_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.secondary_vertex_buffer.slice(..));
            render_pass.draw(0..self.secondary_vertex_count * 6, 0..1);
        }
    }
    
    fn update_uniform_buffer(&self, gpu: &GpuContext) {
        let matrix = self.camera.view_projection_matrix_transposed();
        let uniform_data = ViewProjectionUniform::new(matrix);
        gpu.queue.write_buffer(&self.view_projection_buffer, 0, bytemuck::cast_slice(&[uniform_data]));
    }
}