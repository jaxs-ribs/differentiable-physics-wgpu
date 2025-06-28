//! Physics simulation renderer using SDF raymarching.
//! 
//! Core renderer for visualizing physics bodies (spheres, boxes, capsules)
//! with precise GPU-accelerated raymarching. Supports both interactive
//! viewing and headless frame capture.

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
const RENDER_TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;

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

/// Main rendering context for physics visualization.
pub struct Renderer {
    // Display resources
    surface: Option<wgpu::Surface<'static>>,
    config: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,
    
    // Scene data
    bodies_buffer: wgpu::Buffer,
    camera: Camera,
    
    // GPU uniforms
    view_projection_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    
    // Frame capture
    capture_resources: Option<CaptureResources>,
}

struct CaptureResources {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    staging_buffer: wgpu::Buffer,
    bytes_per_row: u32,
}

struct GpuResources {
    pipeline: wgpu::RenderPipeline,
    bodies_buffer: wgpu::Buffer,
    view_projection_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl Renderer {
    pub async fn new(
        window: Option<&winit::window::Window>,
        gpu: &GpuContext,
        enable_capture: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let surface = create_surface(window, gpu)?;
        let (config, surface_format) = configure_surface(&surface, window, gpu)?;
        
        let camera = Camera::new(config.width as f32 / config.height as f32);
        let resources = create_gpu_resources(gpu, &camera, surface_format)?;
        let capture_resources = if enable_capture || surface.is_none() {
            Some(create_capture_resources(gpu, &config, surface_format)?)
        } else {
            None
        };
        
        Ok(Self {
            surface,
            config,
            render_pipeline: resources.pipeline,
            bodies_buffer: resources.bodies_buffer,
            view_projection_buffer: resources.view_projection_buffer,
            bind_group: resources.bind_group,
            camera,
            capture_resources,
        })
    }
    
    /// Updates scene bodies on the GPU.
    pub fn update(&mut self, gpu: &GpuContext, bodies: &[Body]) {
        if !bodies.is_empty() {
            let bytes = bytemuck::cast_slice(bodies);
            gpu.queue.write_buffer(&self.bodies_buffer, 0, bytes);
        }
    }
    
    /// Renders the current frame to the display surface.
    pub fn render(&self, gpu: &GpuContext) -> Result<(), wgpu::SurfaceError> {
        let output = self.acquire_surface_texture()?;
        
        self.update_uniform_buffer(gpu);
        let commands = self.encode_render_commands(gpu, &output);
        let _submission = self.submit_and_wait(gpu, commands);
        
        self.present_frame(output);
        Ok(())
    }
    
    fn acquire_surface_texture(&self) -> Result<Option<wgpu::SurfaceTexture>, wgpu::SurfaceError> {
        match &self.surface {
            Some(surface) => Ok(Some(surface.get_current_texture()?)),
            None => Ok(None),
        }
    }
    
    fn encode_render_commands(
        &self,
        gpu: &GpuContext,
        output: &Option<wgpu::SurfaceTexture>,
    ) -> wgpu::CommandBuffer {
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        
        let view = self.get_render_target_view(output);
        self.encode_render_pass(&mut encoder, &view);
        
        encoder.finish()
    }
    
    fn get_render_target_view(&self, output: &Option<wgpu::SurfaceTexture>) -> wgpu::TextureView {
        if let Some(ref output) = output {
            output.texture.create_view(&wgpu::TextureViewDescriptor::default())
        } else if let Some(ref capture) = self.capture_resources {
            capture.texture.create_view(&wgpu::TextureViewDescriptor::default())
        } else {
            panic!("No render target available");
        }
    }
    
    fn submit_and_wait(&self, gpu: &GpuContext, commands: wgpu::CommandBuffer) -> wgpu::SubmissionIndex {
        let submission = gpu.queue.submit(Some(commands));
        
        if self.capture_resources.is_some() {
            gpu.device.poll(wgpu::MaintainBase::Wait);
        }
        
        submission
    }
    
    fn present_frame(&self, output: Option<wgpu::SurfaceTexture>) {
        if let Some(output) = output {
            output.present();
        }
    }
    
    /// Renders to texture for headless capture.
    pub fn render_to_texture(&self, gpu: &GpuContext) {
        if let Some(ref capture) = self.capture_resources {
            self.update_uniform_buffer(gpu);
            let commands = self.encode_frame(gpu, &capture.view);
            let submission = gpu.queue.submit(Some(commands));
            gpu.device.poll(wgpu::MaintainBase::WaitForSubmissionIndex(submission));
        }
    }
    
    /// Captures the current frame to a byte buffer.
    pub fn capture_frame(&self, gpu: &GpuContext) -> Option<Vec<u8>> {
        let capture = self.capture_resources.as_ref()?;
        
        self.copy_rendered_frame_to_buffer(gpu, capture);
        let raw_data = self.read_frame_from_buffer(gpu, &capture.staging_buffer)?;
        
        Some(self.extract_frame_pixels(&raw_data, capture.bytes_per_row))
    }
    
    fn copy_rendered_frame_to_buffer(&self, gpu: &GpuContext, capture: &CaptureResources) {
        copy_texture_to_buffer(gpu, capture, &self.config);
    }
    
    fn read_frame_from_buffer(
        &self,
        gpu: &GpuContext,
        buffer: &wgpu::Buffer,
    ) -> Option<Vec<u8>> {
        let view = read_buffer_data(gpu, buffer)?;
        Some(view.to_vec())
    }
    
    fn extract_frame_pixels(&self, raw_data: &[u8], bytes_per_row: u32) -> Vec<u8> {
        extract_frame_data(
            raw_data,
            self.config.width,
            self.config.height,
            bytes_per_row,
        )
    }
    
    /// Handles window resize events.
    pub fn resize(&mut self, gpu: &GpuContext, new_size: winit::dpi::PhysicalSize<u32>) {
        if !self.is_valid_size(new_size) {
            return;
        }
        
        self.update_config_size(new_size);
        self.reconfigure_surface(gpu);
        self.update_camera_aspect_ratio();
        self.update_uniform_buffer(gpu);
    }
    
    fn is_valid_size(&self, size: winit::dpi::PhysicalSize<u32>) -> bool {
        size.width > 0 && size.height > 0
    }
    
    fn update_config_size(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.config.width = new_size.width;
        self.config.height = new_size.height;
    }
    
    fn reconfigure_surface(&self, gpu: &GpuContext) {
        if let Some(ref surface) = self.surface {
            surface.configure(&gpu.device, &self.config);
        }
    }
    
    fn update_camera_aspect_ratio(&mut self) {
        let aspect_ratio = self.config.width as f32 / self.config.height as f32;
        self.camera.update_aspect_ratio(aspect_ratio);
    }
    
    /// Returns mutable camera for user interaction.
    pub fn camera_mut(&mut self) -> &mut Camera {
        &mut self.camera
    }
    
    fn encode_frame(&self, gpu: &GpuContext, view: &wgpu::TextureView) -> wgpu::CommandBuffer {
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        self.encode_render_pass(&mut encoder, view);
        encoder.finish()
    }
    
    fn encode_render_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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
        
        pass.set_pipeline(&self.render_pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
    
    fn update_uniform_buffer(&self, gpu: &GpuContext) {
        let matrix = self.camera.view_projection_matrix_transposed();
        let uniform_data = ViewProjectionUniform::new(matrix);
        gpu.queue.write_buffer(&self.view_projection_buffer, 0, bytemuck::cast_slice(&[uniform_data]));
    }
}

// Helper functions

fn create_surface(
    window: Option<&winit::window::Window>,
    gpu: &GpuContext,
) -> Result<Option<wgpu::Surface<'static>>, Box<dyn std::error::Error>> {
    match window {
        Some(window) => {
            let surface = unsafe {
                let surface = gpu.instance.create_surface_unsafe(
                    wgpu::SurfaceTargetUnsafe::from_window(window)?
                )?;
                std::mem::transmute::<wgpu::Surface<'_>, wgpu::Surface<'static>>(surface)
            };
            Ok(Some(surface))
        }
        None => Ok(None),
    }
}

fn configure_surface(
    surface: &Option<wgpu::Surface<'static>>,
    window: Option<&winit::window::Window>,
    gpu: &GpuContext,
) -> Result<(wgpu::SurfaceConfiguration, wgpu::TextureFormat), Box<dyn std::error::Error>> {
    match surface {
        Some(surface) => {
            let caps = surface.get_capabilities(&gpu.adapter);
            let format = caps.formats.iter()
                .find(|f| f.is_srgb())
                .copied()
                .unwrap_or(caps.formats[0]);
            
            let size = window.unwrap().inner_size();
            let config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width: size.width,
                height: size.height,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: caps.alpha_modes[0],
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            };
            surface.configure(&gpu.device, &config);
            Ok((config, format))
        }
        None => {
            let config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: RENDER_TARGET_FORMAT,
                width: 800,
                height: 600,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: wgpu::CompositeAlphaMode::Opaque,
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            };
            Ok((config, RENDER_TARGET_FORMAT))
        }
    }
}

fn create_gpu_resources(
    gpu: &GpuContext,
    camera: &Camera,
    surface_format: wgpu::TextureFormat,
) -> Result<GpuResources, Box<dyn std::error::Error>> {
    let buffers = create_gpu_buffers(gpu, camera);
    let shader = load_sdf_shader(gpu);
    let bind_group = create_resource_bindings(gpu, &buffers);
    let pipeline = create_render_pipeline(gpu, &shader, &bind_group.layout, surface_format);
    
    Ok(GpuResources {
        pipeline,
        bodies_buffer: buffers.bodies,
        view_projection_buffer: buffers.view_projection,
        bind_group: bind_group.group,
    })
}

struct GpuBuffers {
    view_projection: wgpu::Buffer,
    bodies: wgpu::Buffer,
}

struct ResourceBindings {
    layout: wgpu::BindGroupLayout,
    group: wgpu::BindGroup,
}

fn create_gpu_buffers(gpu: &GpuContext, camera: &Camera) -> GpuBuffers {
    GpuBuffers {
        view_projection: create_uniform_buffer(gpu, camera),
        bodies: create_bodies_buffer(gpu),
    }
}

fn load_sdf_shader(gpu: &GpuContext) -> wgpu::ShaderModule {
    gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("SDF Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/sdf.wgsl").into()),
    })
}

fn create_resource_bindings(gpu: &GpuContext, buffers: &GpuBuffers) -> ResourceBindings {
    let layout = create_bind_group_layout(gpu);
    let group = create_bind_group(
        gpu,
        &layout,
        &buffers.view_projection,
        &buffers.bodies,
    );
    
    ResourceBindings { layout, group }
}

fn create_uniform_buffer(gpu: &GpuContext, camera: &Camera) -> wgpu::Buffer {
    let uniform_data = ViewProjectionUniform::new(camera.view_projection_matrix_transposed());
    
    gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("View Projection Buffer"),
        contents: bytemuck::cast_slice(&[uniform_data]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

fn create_bodies_buffer(gpu: &GpuContext) -> wgpu::Buffer {
    gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Bodies Buffer"),
        size: (std::mem::size_of::<Body>() * MAX_BODIES) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
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
    let layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("SDF Pipeline Layout"),
        bind_group_layouts: &[bind_group_layout],
        push_constant_ranges: &[],
    });
    
    gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("SDF Pipeline"),
        layout: Some(&layout),
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

fn create_capture_resources(
    gpu: &GpuContext,
    config: &wgpu::SurfaceConfiguration,
    format: wgpu::TextureFormat,
) -> Result<CaptureResources, Box<dyn std::error::Error>> {
    let bytes_per_row = calculate_aligned_bytes_per_row(config.width);
    
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
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    
    let buffer_size = bytes_per_row as u64 * config.height as u64;
    let staging_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Capture Staging Buffer"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    
    Ok(CaptureResources {
        texture,
        view,
        staging_buffer,
        bytes_per_row,
    })
}

fn calculate_aligned_bytes_per_row(width: u32) -> u32 {
    let unpadded = width * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    ((unpadded + align - 1) / align) * align
}

fn copy_texture_to_buffer(
    gpu: &GpuContext,
    capture: &CaptureResources,
    config: &wgpu::SurfaceConfiguration,
) {
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Capture Encoder"),
    });
    
    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: &capture.texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &capture.staging_buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(capture.bytes_per_row),
                rows_per_image: Some(config.height),
            },
        },
        wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
    );
    
    let submission = gpu.queue.submit(Some(encoder.finish()));
    gpu.device.poll(wgpu::MaintainBase::WaitForSubmissionIndex(submission));
}

fn read_buffer_data<'a>(
    gpu: &'a GpuContext,
    buffer: &'a wgpu::Buffer,
) -> Option<wgpu::BufferView<'a>> {
    let slice = buffer.slice(..);
    let (tx, rx) = futures::channel::oneshot::channel();
    
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    
    gpu.device.poll(wgpu::MaintainBase::Wait);
    pollster::block_on(rx).unwrap().ok()?;
    
    Some(slice.get_mapped_range())
}

fn extract_frame_data(
    data: &[u8],
    width: u32,
    height: u32,
    bytes_per_row: u32,
) -> Vec<u8> {
    let mut frame_data = Vec::with_capacity(calculate_frame_size(width, height));
    
    for y in 0..height {
        append_row_data(&mut frame_data, data, y, width, bytes_per_row);
    }
    
    frame_data
}

fn calculate_frame_size(width: u32, height: u32) -> usize {
    (width * height * 4) as usize
}

fn append_row_data(
    frame_data: &mut Vec<u8>,
    source_data: &[u8],
    row: u32,
    width: u32,
    padded_bytes_per_row: u32,
) {
    let row_start = (row * padded_bytes_per_row) as usize;
    let row_bytes = (width * 4) as usize;
    let row_end = row_start + row_bytes;
    
    frame_data.extend_from_slice(&source_data[row_start..row_end]);
}