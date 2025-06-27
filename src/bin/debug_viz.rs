use clap::Parser;
use physics_core::{
    body::Body,
    gpu::GpuContext,
    viz::{WindowManager, DualRenderer},
};
use std::{path::PathBuf, time::{Instant, Duration}, process::Command, sync::{Arc, Mutex}};
use winit::{
    dpi::PhysicalPosition,
    event::{ElementState, Event, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};
use image::{ImageBuffer, Rgba};

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
    
    /// Path to save video recording (e.g., preview.mp4)
    #[arg(long)]
    record: Option<PathBuf>,
    
    /// Duration of recording in seconds (default: 5)
    #[arg(long, default_value = "5")]
    duration: f32,
    
    /// Frames per second for recording (default: 30)
    #[arg(long, default_value = "30")]
    fps: u32,
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
        println!("👁️  Loading oracle trace from: {}", path.display());
        let trace = load_body_trace_from_npy(&path)?;
        println!("   Loaded {} frames with {} bodies each", trace.len(), trace.get(0).map_or(0, |f| f.len()));
        oracle_trace = Some(trace);
    }
    if let Some(path) = args.gpu {
        println!("👁️  Loading GPU trace from: {}", path.display());
        let trace = load_body_trace_from_npy(&path)?;
        println!("   Loaded {} frames with {} bodies each", trace.len(), trace.get(0).map_or(0, |f| f.len()));
        gpu_trace = Some(trace);
    }
    
    let num_frames = oracle_trace.as_ref().map_or(0, |t| t.len()).max(gpu_trace.as_ref().map_or(0, |t| t.len()));
    if num_frames == 0 {
        println!("No trace file loaded, nothing to display. Exiting.");
        println!("Tip: Use --oracle <path_to_trace.npy>");
        return Ok(());
    }
    
    // Setup video recording if requested
    let recording = args.record.is_some();
    
    let event_loop = EventLoop::new()?;
    let window_manager = WindowManager::new(&event_loop)?;
    let gpu_context = pollster::block_on(GpuContext::new())?;
    let mut renderer = pollster::block_on(DualRenderer::new_with_capture(&window_manager, &gpu_context, recording))?;
    
    let output_path = args.record.clone();
    // For recording, create a fixed-size window
    if recording {
        // Resize window to 800x600 for consistent video size
        let _ = window_manager.window().request_inner_size(winit::dpi::LogicalSize::new(800, 600));
    }
    
    let captured_frames = Arc::new(Mutex::new(Vec::<Vec<u8>>::new()));
    
    let mut current_frame = 0;
    let recording_fps = args.fps;
    let frame_duration = if recording {
        Duration::from_millis(1000 / recording_fps as u64)
    } else {
        Duration::from_millis(16) // ~60 FPS for interactive mode
    };
    let mut last_update = Instant::now();
    let max_frames = (recording_fps as f32 * args.duration) as usize;

    let mut mouse_pressed = false;
    let mut last_mouse_pos: PhysicalPosition<f64> = PhysicalPosition::new(0.0, 0.0);
    const MOUSE_SENSITIVITY: f32 = 0.01;
    
    let captured_frames_clone = captured_frames.clone();
    
    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::AboutToWait => {
                // Check if recording should end
                if recording && captured_frames_clone.lock().unwrap().len() >= max_frames {
                    elwt.exit();
                }
                
                if last_update.elapsed() >= frame_duration {
                    current_frame = (current_frame + 1) % num_frames;
                    
                    let oracle_bodies = oracle_trace.as_ref().and_then(|t| t.get(current_frame)).map(|v| v.as_slice());
                    let gpu_bodies = gpu_trace.as_ref().and_then(|t| t.get(current_frame)).map(|v| v.as_slice());
                    
                    renderer.update_scenes(&gpu_context, oracle_bodies, gpu_bodies);
                    
                    // During recording, we need to render and capture immediately
                    if recording {
                        match renderer.render(&gpu_context) {
                            Ok(_) => {
                                // Force GPU to complete all pending operations
                                gpu_context.device.poll(wgpu::MaintainBase::Wait);
                                gpu_context.queue.submit(std::iter::empty());
                                gpu_context.device.poll(wgpu::MaintainBase::Wait);
                                
                                let current_count = captured_frames_clone.lock().unwrap().len();
                                if current_count < max_frames {
                                    if let Some(frame_data) = renderer.capture_frame(&gpu_context) {
                                        captured_frames_clone.lock().unwrap().push(frame_data);
                                        if current_count % 30 == 0 {
                                            println!("Captured frame {} of {}", current_count + 1, max_frames);
                                        }
                                    }
                                }
                            }
                            Err(e) => eprintln!("Render error during recording: {:?}", e),
                        }
                    }
                    
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
                    // During recording, rendering is handled in AboutToWait
                    if !recording {
                        match renderer.render(&gpu_context) {
                            Ok(_) => {}
                            Err(wgpu::SurfaceError::Lost) => {
                                renderer.resize(&gpu_context, window_manager.inner_size())
                            }
                            Err(wgpu::SurfaceError::OutOfMemory) => elwt.exit(),
                            Err(e) => eprintln!("Render error: {:?}", e),
                        }
                    }
                }
                _ => {}
            },
            _ => {}
        }
    })?;
    
    // Save video if recording
    if recording {
        if let Some(output_path) = output_path {
            let frames = captured_frames.lock().unwrap();
            save_frames_as_video(&*frames, output_path, recording_fps, 800, 600)?;
        }
    }

    Ok(())
}

fn save_frames_as_video(
    frames: &[Vec<u8>],
    output_path: PathBuf,
    fps: u32,
    width: u32,
    height: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Saving {} frames to video...", frames.len());
    
    // Create temp directory
    let temp_dir = std::env::temp_dir().join(format!("physics_recording_{}", std::process::id()));
    std::fs::create_dir_all(&temp_dir)?;
    
    // Save frames as PNG files
    for (i, frame_data) in frames.iter().enumerate() {
        let filename = temp_dir.join(format!("frame_{:04}.png", i));
        
        // Convert BGRA to RGBA
        let mut rgba_data = vec![0u8; (width * height * 4) as usize];
        for j in 0..((width * height) as usize) {
            rgba_data[j * 4] = frame_data[j * 4 + 2];     // R
            rgba_data[j * 4 + 1] = frame_data[j * 4 + 1]; // G
            rgba_data[j * 4 + 2] = frame_data[j * 4];     // B
            rgba_data[j * 4 + 3] = frame_data[j * 4 + 3]; // A
        }
        
        let img = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(width, height, rgba_data)
            .ok_or("Failed to create image buffer")?;
        img.save(&filename)?;
    }
    
    // Run ffmpeg to create video
    let status = Command::new("ffmpeg")
        .args(&[
            "-y", // Overwrite output
            "-r", &fps.to_string(),
            "-i", &temp_dir.join("frame_%04d.png").to_string_lossy(),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "28",
            "-pix_fmt", "yuv420p",
            "-vf", "scale=800:600", // Ensure output size
            &output_path.to_string_lossy(),
        ])
        .status()?;
    
    if !status.success() {
        return Err("FFmpeg encoding failed".into());
    }
    
    // Clean up
    std::fs::remove_dir_all(&temp_dir)?;
    println!("Video saved to: {}", output_path.display());
    
    Ok(())
}