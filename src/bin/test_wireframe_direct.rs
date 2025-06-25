#[cfg(not(feature = "viz"))]
compile_error!("This demo requires the 'viz' feature");

use physics_core::{gpu::GpuContext};
use pollster::block_on;
use wgpu::util::DeviceExt;
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::Window,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    let event_loop = EventLoop::new()?;
    let gpu = block_on(GpuContext::new())?;
    
    // Create window
    let window = event_loop.create_window(
        Window::default_attributes()
            .with_title("Direct Wireframe Test")
            .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
    )?;
    
    let surface = unsafe {
        gpu.instance.create_surface_unsafe(
            wgpu::SurfaceTargetUnsafe::from_window(&window)?
        )?
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
    
    // Create a very simple shader that draws lines in NDC space
    let shader_source = r#"
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_idx: u32,
) -> VertexOutput {
    var out: VertexOutput;
    
    // Draw a simple triangle in NDC space
    if (vertex_idx == 0u) {
        out.clip_position = vec4<f32>(-0.5, -0.5, 0.5, 1.0);
        out.color = vec3<f32>(1.0, 0.0, 0.0);
    } else if (vertex_idx == 1u) {
        out.clip_position = vec4<f32>(0.5, -0.5, 0.5, 1.0);
        out.color = vec3<f32>(0.0, 1.0, 0.0);
    } else {
        out.clip_position = vec4<f32>(0.0, 0.5, 0.5, 1.0);
        out.color = vec3<f32>(0.0, 0.0, 1.0);
    }
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
"#;
    
    let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Test Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });
    
    let pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Test Pipeline Layout"),
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });
    
    let render_pipeline = gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Test Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[],
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
    
    println!("Direct Wireframe Test");
    println!("====================");
    println!("You should see a colorful triangle if rendering is working");
    
    event_loop.run(move |event, control_flow| {
        match event {
            Event::WindowEvent { event, window_id } if window_id == window.id() => {
                match event {
                    WindowEvent::CloseRequested => control_flow.exit(),
                    WindowEvent::Resized(physical_size) => {
                        config.width = physical_size.width;
                        config.height = physical_size.height;
                        surface.configure(&gpu.device, &config);
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
                                            g: 0.2,
                                            b: 0.3,
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
                            render_pass.draw(0..3, 0..1);
                        }
                        
                        gpu.queue.submit(Some(encoder.finish()));
                        output.present();
                        
                        window.request_redraw();
                    },
                    _ => {}
                }
            },
            _ => {}
        }
    })?;
    
    Ok(())
}