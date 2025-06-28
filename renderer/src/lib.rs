//! Minimal SDF raymarching renderer for physics simulation.
//!
//! This renderer uses brute-force SDF raymarching to precisely visualize
//! spheres, boxes, and capsules. It's designed for verifiability with a
//! headless render-to-file mode.
//!
//! # Architecture
//! - `body`: Physics body data structures
//! - `camera`: 3D camera controls
//! - `gpu`: GPU context management
//! - `video`: Video recording functionality
//! - `loader`: NPY file loading

pub mod body;
pub mod camera;
pub mod gpu;
pub mod video;
pub mod loader;

use wgpu::util::DeviceExt;
use camera::Camera;
use gpu::GpuContext;
use body::Body;

const MAX_BODIES: usize = 1000;
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

pub struct Renderer {
    surface: Option<wgpu::Surface<'static>>,
    config: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,
    
    // Body data buffer
    bodies_buffer: wgpu::Buffer,
    
    // Camera resources
    view_projection_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    camera: Camera,
    
    // Frame capture resources (for video recording)
    capture_texture: Option<wgpu::Texture>,
    capture_view: Option<wgpu::TextureView>,
    staging_buffer: Option<wgpu::Buffer>,
    bytes_per_row: u32,
}

impl Renderer {
    pub async fn new(
        window: Option<&winit::window::Window>,
        gpu: &GpuContext,
        enable_capture: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Create surface if window is provided
        let surface = if let Some(window) = window {
            let surface = unsafe {
                let surface = gpu.instance.create_surface_unsafe(
                    wgpu::SurfaceTargetUnsafe::from_window(window)?
                )?;
                std::mem::transmute::<wgpu::Surface<'_>, wgpu::Surface<'static>>(surface)
            };
            Some(surface)
        } else {
            None
        };
        
        // Configure surface or use default config
        let (config, surface_format) = if let Some(ref surface) = surface {
            let surface_caps = surface.get_capabilities(&gpu.adapter);
            let surface_format = surface_caps.formats.iter()
                .find(|f| f.is_srgb())
                .copied()
                .unwrap_or(surface_caps.formats[0]);
            
            let size = window.unwrap().inner_size();
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
            (config, surface_format)
        } else {
            // Headless mode - use default config
            let config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: wgpu::TextureFormat::Bgra8UnormSrgb,
                width: 800,
                height: 600,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: wgpu::CompositeAlphaMode::Opaque,
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            };
            (config, wgpu::TextureFormat::Bgra8UnormSrgb)
        };
        
        let camera = Camera::new(config.width as f32 / config.height as f32);
        let view_projection_buffer = Self::create_uniform_buffer(gpu, &camera);
        
        // Create bodies buffer
        let bodies_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bodies Buffer"),
            size: (std::mem::size_of::<Body>() * MAX_BODIES) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let bind_group_layout = Self::create_bind_group_layout(gpu);
        let bind_group = Self::create_bind_group(
            gpu,
            &bind_group_layout,
            &view_projection_buffer,
            &bodies_buffer,
        );
        
        let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SDF Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/sdf.wgsl").into()),
        });
        
        let render_pipeline = Self::create_render_pipeline(gpu, &shader, &bind_group_layout, surface_format);
        
        // Create capture resources if requested
        let (capture_texture, capture_view, staging_buffer, bytes_per_row) = if enable_capture || surface.is_none() {
            let unpadded_bytes_per_row = config.width * 4;
            let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
            let padded_bytes_per_row = ((unpadded_bytes_per_row + align - 1) / align) * align;
            
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
                format: surface_format,
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
            
            (Some(texture), Some(view), Some(buffer), padded_bytes_per_row)
        } else {
            (None, None, None, 0)
        };
        
        Ok(Self {
            surface,
            config,
            render_pipeline,
            bodies_buffer,
            view_projection_buffer,
            bind_group,
            camera,
            capture_texture,
            capture_view,
            staging_buffer,
            bytes_per_row,
        })
    }
    
    pub fn update(&mut self, gpu: &GpuContext, bodies: &[Body]) {
        // Write entire body slice to GPU
        if !bodies.is_empty() {
            let bytes = bytemuck::cast_slice(bodies);
            gpu.queue.write_buffer(&self.bodies_buffer, 0, bytes);
        }
    }
    
    pub fn render(&self, gpu: &GpuContext) -> Result<(), wgpu::SurfaceError> {
        let output = if let Some(ref surface) = self.surface {
            Some(surface.get_current_texture()?)
        } else {
            None
        };
        
        let surface_view = output.as_ref().map(|o| 
            o.texture.create_view(&wgpu::TextureViewDescriptor::default())
        );
        
        self.update_uniform_buffer(gpu);

        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        
        // Render to capture texture if available, otherwise to surface
        let render_view = self.capture_view.as_ref()
            .or(surface_view.as_ref())
            .expect("No render target available");
        
        self.encode_render_pass(&mut encoder, render_view);
        
        let submission_index = gpu.queue.submit(Some(encoder.finish()));
        
        if self.capture_texture.is_some() {
            gpu.device.poll(wgpu::MaintainBase::WaitForSubmissionIndex(submission_index));
        }
        
        if let Some(output) = output {
            output.present();
        }
        
        Ok(())
    }
    
    pub fn render_to_texture(&self, gpu: &GpuContext) {
        self.update_uniform_buffer(gpu);

        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        
        if let Some(ref capture_view) = self.capture_view {
            self.encode_render_pass(&mut encoder, capture_view);
        }
        
        let submission_index = gpu.queue.submit(Some(encoder.finish()));
        gpu.device.poll(wgpu::MaintainBase::WaitForSubmissionIndex(submission_index));
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
            if let Some(ref surface) = self.surface {
                surface.configure(&gpu.device, &self.config);
            }
            
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
    
    fn create_bind_group_layout(gpu: &GpuContext) -> wgpu::BindGroupLayout {
        gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SDF Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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
        bodies_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SDF Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: view_projection_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bodies_buffer.as_entire_binding(),
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
            label: Some("SDF Pipeline Layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });
        
        gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SDF Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
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
    
    fn encode_render_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SDF Render Pass"),
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
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.draw(0..3, 0..1); // Single full-screen triangle
    }
    
    fn update_uniform_buffer(&self, gpu: &GpuContext) {
        let matrix = self.camera.view_projection_matrix_transposed();
        let uniform_data = ViewProjectionUniform::new(matrix);
        gpu.queue.write_buffer(&self.view_projection_buffer, 0, bytemuck::cast_slice(&[uniform_data]));
    }
}