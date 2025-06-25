use crate::{body::Body, gpu::GpuContext};
use wgpu::util::DeviceExt;
use winit::{
    event_loop::EventLoop,
    window::Window,
};

pub struct Visualizer {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,
    line_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    window: Window,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ViewProjection {
    view_proj: [[f32; 4]; 4],
}

impl Visualizer {
    pub async fn new(event_loop: &EventLoop<()>, gpu: &GpuContext) -> Result<Self, Box<dyn std::error::Error>> {
        let window_attributes = Window::default_attributes()
            .with_title("Physics Engine Visualization")
            .with_inner_size(winit::dpi::LogicalSize::new(1024, 768));
        let window = event_loop.create_window(window_attributes)?;
        
        // Create surface - need to use unsafe to get 'static lifetime
        let surface = unsafe {
            let surface = gpu.instance.create_surface_unsafe(
                wgpu::SurfaceTargetUnsafe::from_window(&window)?
            )?;
            std::mem::transmute::<wgpu::Surface<'_>, wgpu::Surface<'static>>(surface)
        };
        
        let surface_caps = surface.get_capabilities(&gpu.adapter);
        let surface_format = surface_caps.formats.iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);
        
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: window.inner_size().width,
            height: window.inner_size().height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        
        surface.configure(&gpu.device, &config);
        
        // Create shader for wireframe rendering
        let shader_source = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
}

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(position, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
"#;
        
        let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Wireframe Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        
        // Create uniform buffer for view-projection matrix
        let view_proj = create_view_proj_matrix(config.width as f32 / config.height as f32);
        let uniform_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[ViewProjection { view_proj }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        // Create line buffer for AABB wireframes (12 lines * 2 vertices * 6 floats)
        let line_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Line Buffer"),
            size: 12 * 2 * 6 * 4 * 1000, // Space for 1000 AABBs
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create bind group layout and bind group
        let bind_group_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Wireframe Bind Group Layout"),
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
            ],
        });
        
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Wireframe Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });
        
        let pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Wireframe Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let render_pipeline = gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Wireframe Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 6 * 4, // 3 position + 3 color
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
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        
        Ok(Self {
            surface,
            config,
            render_pipeline,
            line_buffer,
            uniform_buffer,
            bind_group,
            window,
        })
    }
    
    pub fn update_bodies(&self, gpu: &GpuContext, bodies: &[Body]) {
        let mut vertices = Vec::new();
        
        for body in bodies {
            // Get AABB from body
            let pos = [body.position[0], body.position[1], body.position[2]];
            let (min, max) = match body.shape_data[0] {
                0 => { // Sphere
                    let r = body.shape_params[0];
                    ([pos[0] - r, pos[1] - r, pos[2] - r],
                     [pos[0] + r, pos[1] + r, pos[2] + r])
                },
                2 => { // Box
                    let hx = body.shape_params[0];
                    let hy = body.shape_params[1];
                    let hz = body.shape_params[2];
                    ([pos[0] - hx, pos[1] - hy, pos[2] - hz],
                     [pos[0] + hx, pos[1] + hy, pos[2] + hz])
                },
                _ => continue, // Skip unknown shapes
            };
            
            let color = if body.shape_data[1] == 1 {
                [0.5, 0.5, 0.5] // Gray for static
            } else {
                [0.0, 1.0, 0.0] // Green for dynamic
            };
            
            // Generate 12 lines for AABB edges
            let corners = [
                [min[0], min[1], min[2]],
                [max[0], min[1], min[2]],
                [max[0], max[1], min[2]],
                [min[0], max[1], min[2]],
                [min[0], min[1], max[2]],
                [max[0], min[1], max[2]],
                [max[0], max[1], max[2]],
                [min[0], max[1], max[2]],
            ];
            
            // Bottom face
            vertices.extend_from_slice(&[corners[0][0], corners[0][1], corners[0][2], color[0], color[1], color[2]]);
            vertices.extend_from_slice(&[corners[1][0], corners[1][1], corners[1][2], color[0], color[1], color[2]]);
            
            vertices.extend_from_slice(&[corners[1][0], corners[1][1], corners[1][2], color[0], color[1], color[2]]);
            vertices.extend_from_slice(&[corners[2][0], corners[2][1], corners[2][2], color[0], color[1], color[2]]);
            
            vertices.extend_from_slice(&[corners[2][0], corners[2][1], corners[2][2], color[0], color[1], color[2]]);
            vertices.extend_from_slice(&[corners[3][0], corners[3][1], corners[3][2], color[0], color[1], color[2]]);
            
            vertices.extend_from_slice(&[corners[3][0], corners[3][1], corners[3][2], color[0], color[1], color[2]]);
            vertices.extend_from_slice(&[corners[0][0], corners[0][1], corners[0][2], color[0], color[1], color[2]]);
            
            // Top face
            vertices.extend_from_slice(&[corners[4][0], corners[4][1], corners[4][2], color[0], color[1], color[2]]);
            vertices.extend_from_slice(&[corners[5][0], corners[5][1], corners[5][2], color[0], color[1], color[2]]);
            
            vertices.extend_from_slice(&[corners[5][0], corners[5][1], corners[5][2], color[0], color[1], color[2]]);
            vertices.extend_from_slice(&[corners[6][0], corners[6][1], corners[6][2], color[0], color[1], color[2]]);
            
            vertices.extend_from_slice(&[corners[6][0], corners[6][1], corners[6][2], color[0], color[1], color[2]]);
            vertices.extend_from_slice(&[corners[7][0], corners[7][1], corners[7][2], color[0], color[1], color[2]]);
            
            vertices.extend_from_slice(&[corners[7][0], corners[7][1], corners[7][2], color[0], color[1], color[2]]);
            vertices.extend_from_slice(&[corners[4][0], corners[4][1], corners[4][2], color[0], color[1], color[2]]);
            
            // Vertical edges
            vertices.extend_from_slice(&[corners[0][0], corners[0][1], corners[0][2], color[0], color[1], color[2]]);
            vertices.extend_from_slice(&[corners[4][0], corners[4][1], corners[4][2], color[0], color[1], color[2]]);
            
            vertices.extend_from_slice(&[corners[1][0], corners[1][1], corners[1][2], color[0], color[1], color[2]]);
            vertices.extend_from_slice(&[corners[5][0], corners[5][1], corners[5][2], color[0], color[1], color[2]]);
            
            vertices.extend_from_slice(&[corners[2][0], corners[2][1], corners[2][2], color[0], color[1], color[2]]);
            vertices.extend_from_slice(&[corners[6][0], corners[6][1], corners[6][2], color[0], color[1], color[2]]);
            
            vertices.extend_from_slice(&[corners[3][0], corners[3][1], corners[3][2], color[0], color[1], color[2]]);
            vertices.extend_from_slice(&[corners[7][0], corners[7][1], corners[7][2], color[0], color[1], color[2]]);
        }
        
        if !vertices.is_empty() {
            gpu.queue.write_buffer(&self.line_buffer, 0, bytemuck::cast_slice(&vertices));
        }
    }
    
    pub fn render(&self, gpu: &GpuContext, vertex_count: u32) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.1,
                            b: 0.1,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.line_buffer.slice(..));
            render_pass.draw(0..vertex_count, 0..1);
        }
        
        gpu.queue.submit(Some(encoder.finish()));
        output.present();
        
        Ok(())
    }
    
    pub fn resize(&mut self, gpu: &GpuContext, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&gpu.device, &self.config);
            
            // Update projection matrix
            let view_proj = create_view_proj_matrix(self.config.width as f32 / self.config.height as f32);
            gpu.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[ViewProjection { view_proj }]));
        }
    }
    
    pub fn window(&self) -> &Window {
        &self.window
    }
}

fn create_view_proj_matrix(aspect_ratio: f32) -> [[f32; 4]; 4] {
    let proj = perspective_matrix(45.0_f32.to_radians(), aspect_ratio, 0.1, 100.0);
    let view = look_at_matrix(
        [0.0, 10.0, 20.0], // eye
        [0.0, 5.0, 0.0],   // center
        [0.0, 1.0, 0.0],   // up
    );
    matrix_multiply(&proj, &view)
}

fn perspective_matrix(fov: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (fov / 2.0).tan();
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far + near) / (near - far), -1.0],
        [0.0, 0.0, (2.0 * far * near) / (near - far), 0.0],
    ]
}

fn look_at_matrix(eye: [f32; 3], center: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
    let f = normalize([
        center[0] - eye[0],
        center[1] - eye[1],
        center[2] - eye[2],
    ]);
    let s = normalize(cross(f, up));
    let u = cross(s, f);
    
    [
        [s[0], u[0], -f[0], 0.0],
        [s[1], u[1], -f[1], 0.0],
        [s[2], u[2], -f[2], 0.0],
        [-dot(s, eye), -dot(u, eye), dot(f, eye), 1.0],
    ]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / len, v[1] / len, v[2] / len]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn matrix_multiply(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut result = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}