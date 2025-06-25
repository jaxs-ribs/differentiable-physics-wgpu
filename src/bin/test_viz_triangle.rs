#[cfg(not(feature = "viz"))]
compile_error!("This test requires the 'viz' feature. Run with: cargo run --features viz --bin test_viz_triangle");

use physics_core::gpu::GpuContext;
use pollster::block_on;
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::Window,
};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    let event_loop = EventLoop::new()?;
    let gpu = block_on(GpuContext::new())?;
    
    let window = event_loop.create_window(Window::default_attributes()
        .with_title("Triangle Test")
        .with_inner_size(winit::dpi::LogicalSize::new(800, 600)))?;
    
    let surface = unsafe {
        let surface = gpu.instance.create_surface_unsafe(
            wgpu::SurfaceTargetUnsafe::from_window(&window)?
        )?;
        std::mem::transmute::<wgpu::Surface<'_>, wgpu::Surface<'static>>(surface)
    };
    
    let surface_caps = surface.get_capabilities(&gpu.adapter);
    let surface_format = surface_caps.formats[0];
    
    let mut config = wgpu::SurfaceConfiguration {
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
    
    // Simple shader that doesn't use any uniforms
    let shader_source = r#"
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
    // Direct NDC coordinates - no transformation
    out.clip_position = vec4<f32>(position.x, position.y, 0.0, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
"#;
    
    let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Triangle Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });
    
    // Create a triangle in NDC space
    let vertices = [
        Vertex { position: [0.0, 0.5, 0.0], color: [1.0, 0.0, 0.0] },    // Top - Red
        Vertex { position: [-0.5, -0.5, 0.0], color: [0.0, 1.0, 0.0] },  // Bottom left - Green
        Vertex { position: [0.5, -0.5, 0.0], color: [0.0, 0.0, 1.0] },   // Bottom right - Blue
    ];
    
    println!("Creating triangle with vertices:");
    for (i, v) in vertices.iter().enumerate() {
        println!("  Vertex {}: pos=({:.1}, {:.1}, {:.1}), color=({:.1}, {:.1}, {:.1})",
            i, v.position[0], v.position[1], v.position[2],
            v.color[0], v.color[1], v.color[2]);
    }
    
    let vertex_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Vertex Buffer"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    
    let pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Triangle Pipeline Layout"),
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });
    
    let render_pipeline = gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Triangle Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    wgpu::VertexAttribute {
                        offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
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
            topology: wgpu::PrimitiveTopology::TriangleList,
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
    
    event_loop.run(move |event, control_flow| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => match event {
                WindowEvent::CloseRequested => {
                    control_flow.exit();
                },
                WindowEvent::Resized(physical_size) => {
                    if physical_size.width > 0 && physical_size.height > 0 {
                        config.width = physical_size.width;
                        config.height = physical_size.height;
                        surface.configure(&gpu.device, &config);
                    }
                },
                WindowEvent::RedrawRequested => {
                    let output = surface.get_current_texture().unwrap();
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
                        
                        render_pass.set_pipeline(&render_pipeline);
                        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                        render_pass.draw(0..3, 0..1);
                    }
                    
                    gpu.queue.submit(Some(encoder.finish()));
                    output.present();
                    
                    println!("Frame rendered");
                    window.request_redraw();
                },
                _ => {}
            },
            _ => {}
        }
    })?;
    
    Ok(())
}