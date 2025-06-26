use clap::Parser;
use physics_core::{
    body::Body,
    gpu::GpuContext,
    viz::{WindowManager, DualRenderer},
};
use std::{path::PathBuf, time::{Instant, Duration}};
use winit::{
    dpi::PhysicalPosition,
    event::{ElementState, Event, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

#[derive(Parser, Debug)]
#[command(name = "debug_viz")]
#[command(about = "Visual debugger for comparing CPU and GPU physics simulations")]
struct Args {
    /// Path to the oracle (CPU) state file
    #[arg(long)]
    oracle: Option<PathBuf>,
    
    /// Path to the GPU state file
    #[arg(long)]
    gpu: Option<PathBuf>,
}

fn create_default_demo_scene() -> Vec<Body> {
    // Create a simple stack of 3 spheres for demo
    vec![
        // Ground sphere (large, static)
        Body {
            position: [0.0, 0.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0, 0.0],
            orientation: [1.0, 0.0, 0.0, 0.0], // Identity quaternion
            angular_vel: [0.0, 0.0, 0.0, 0.0],
            mass_data: [1e8, 0.0, 0.0, 0.0], // Very large mass = static
            shape_data: [0, 0, 0, 0], // Shape type 0 = sphere
            shape_params: [5.0, 0.0, 0.0, 0.0], // Radius 5
        },
        // Middle sphere
        Body {
            position: [0.0, 7.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0, 0.0],
            orientation: [1.0, 0.0, 0.0, 0.0],
            angular_vel: [0.0, 0.0, 0.0, 0.0],
            mass_data: [1.0, 1.0, 0.0, 0.0],
            shape_data: [0, 0, 0, 0], // Sphere
            shape_params: [1.0, 0.0, 0.0, 0.0], // Radius 1
        },
        // Top sphere
        Body {
            position: [2.0, 9.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0, 0.0],
            orientation: [1.0, 0.0, 0.0, 0.0],
            angular_vel: [0.0, 0.0, 0.0, 0.0],
            mass_data: [1.0, 1.0, 0.0, 0.0],
            shape_data: [0, 0, 0, 0], // Sphere
            shape_params: [1.0, 0.0, 0.0, 0.0], // Radius 1
        },
    ]
}

fn load_body_trace_from_npy(
    path: &PathBuf,
) -> Result<Vec<Vec<Body>>, Box<dyn std::error::Error>> {
    use npyz::NpyFile;
    use std::io::Read;
    
    let mut file = std::fs::File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    let npy = NpyFile::new(&buffer[..])?;
    
    let shape = npy.shape();
    if shape.len() != 2 {
        return Err(format!("Expected a 2D array (frames, data), but got shape: {:?}", shape).into());
    }
    
    let num_frames = shape[0] as usize;
    let data_per_frame = shape[1] as usize;
    let floats_per_body = 18;
    if data_per_frame % floats_per_body != 0 {
        return Err(format!("Frame data size ({}) is not divisible by floats per body ({})", data_per_frame, floats_per_body).into());
    }
    let num_bodies = data_per_frame / floats_per_body;
    
    let data: Vec<f32> = npy.into_vec()?;
    
    let mut frames = Vec::with_capacity(num_frames);
    for frame_idx in 0..num_frames {
        let mut bodies = Vec::with_capacity(num_bodies);
        for body_idx in 0..num_bodies {
            let frame_offset = frame_idx * data_per_frame;
            let body_offset = body_idx * floats_per_body;
            let offset = frame_offset + body_offset;
            
            let position = [data[offset], data[offset + 1], data[offset + 2], 0.0];
            let velocity = [data[offset + 3], data[offset + 4], data[offset + 5], 0.0];
            let orientation = [
                data[offset + 6],
                data[offset + 7],
                data[offset + 8],
                data[offset + 9],
            ];
            let angular_vel = [data[offset + 10], data[offset + 11], data[offset + 12], 0.0];
            let mass = data[offset + 13];
            let shape_type = data[offset + 14] as u32;
            let shape_params = [data[offset + 15], data[offset + 16], data[offset + 17], 0.0];
            
            bodies.push(Body {
                position,
                velocity,
                orientation,
                angular_vel,
                mass_data: [mass, if mass > 0.0 { 1.0 / mass } else { 0.0 }, 0.0, 0.0],
                shape_data: [shape_type, 0, 0, 0],
                shape_params,
            });
        }
        frames.push(bodies);
    }
    
    Ok(frames)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args = Args::parse();
    
    let mut oracle_trace: Option<Vec<Vec<Body>>> = None;
    let mut gpu_trace: Option<Vec<Vec<Body>>> = None;
    
    if let Some(path) = args.oracle {
        println!("👁️  Loading oracle trace...");
        oracle_trace = Some(load_body_trace_from_npy(&path)?);
    }
    if let Some(path) = args.gpu {
        println!("👁️  Loading GPU trace...");
        gpu_trace = Some(load_body_trace_from_npy(&path)?);
    }
    
    let num_frames = oracle_trace.as_ref().map_or(0, |t| t.len()).max(gpu_trace.as_ref().map_or(0, |t| t.len()));
    if num_frames == 0 {
        println!("No trace file loaded, nothing to display. Exiting.");
        println!("Tip: Use --oracle <path_to_trace.npy>");
        return Ok(());
    }
    
    let event_loop = EventLoop::new()?;
    let window_manager = WindowManager::new(&event_loop)?;
    let gpu_context = pollster::block_on(GpuContext::new())?;
    let mut renderer = pollster::block_on(DualRenderer::new(&window_manager, &gpu_context))?;
    
    let mut current_frame = 0;
    let frame_duration = Duration::from_millis(16); // ~60 FPS
    let mut last_update = Instant::now();

    let mut mouse_pressed = false;
    let mut last_mouse_pos: PhysicalPosition<f64> = PhysicalPosition::new(0.0, 0.0);
    const MOUSE_SENSITIVITY: f32 = 0.01;
    
    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::AboutToWait => {
                if last_update.elapsed() >= frame_duration {
                    current_frame = (current_frame + 1) % num_frames;
                    
                    let oracle_bodies = oracle_trace.as_ref().and_then(|t| t.get(current_frame)).map(|v| v.as_slice());
                    let gpu_bodies = gpu_trace.as_ref().and_then(|t| t.get(current_frame)).map(|v| v.as_slice());
                    
                    renderer.update_scenes(&gpu_context, oracle_bodies, gpu_bodies);
                    
                    last_update = Instant::now();
                    window_manager.window().request_redraw();
                }
            }
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => elwt.exit(),
                WindowEvent::Resized(physical_size) => {
                    renderer.resize(&gpu_context, physical_size);
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    if button == MouseButton::Left {
                        mouse_pressed = state == ElementState::Pressed;
                    }
                }
                WindowEvent::CursorMoved { position, .. } => {
                    if mouse_pressed {
                        let delta_x = (position.x - last_mouse_pos.x) as f32 * MOUSE_SENSITIVITY;
                        let delta_y = (position.y - last_mouse_pos.y) as f32 * MOUSE_SENSITIVITY;
                        renderer.camera_mut().rotate(-delta_x, -delta_y);
                    }
                    last_mouse_pos = position;
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    let scroll_amount = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(pos) => pos.y as f32,
                    };
                    renderer.camera_mut().zoom(scroll_amount);
                }
                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            physical_key,
                            state: ElementState::Pressed,
                            ..
                        },
                    ..
                } => match physical_key {
                    winit::keyboard::PhysicalKey::Code(code) => match code {
                        winit::keyboard::KeyCode::KeyQ | winit::keyboard::KeyCode::Escape => {
                            elwt.exit();
                        }
                        _ => {}
                    },
                    _ => {}
                },
                WindowEvent::RedrawRequested => {
                    match renderer.render(&gpu_context) {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost) => {
                            renderer.resize(&gpu_context, window_manager.inner_size())
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => elwt.exit(),
                        Err(e) => eprintln!("Render error: {:?}", e),
                    }
                }
                _ => {}
            },
            _ => {}
        }
    })?;

    Ok(())
}