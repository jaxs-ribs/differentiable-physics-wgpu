//! Main entry point for the physics renderer application.
//!
//! This module handles:
//! - Command-line argument parsing
//! - Window creation and event loop
//! - User input handling (mouse, keyboard)
//! - Frame timing and video recording
//! - Coordination between loader, renderer, and video modules
//!
//! # Architecture
//! The main function follows a clear initialization-loop-cleanup pattern:
//! 1. Parse arguments and load trajectory data
//! 2. Initialize GPU and rendering context
//! 3. Run event loop with frame updates
//! 4. Save video if recording was enabled
//!
//! # Event Handling
//! - Mouse drag: Camera rotation
//! - Mouse wheel: Camera zoom
//! - Q/Escape: Exit application
//! - Window resize: Update renderer viewport

use clap::Parser;
use physics_renderer::{
    body::Body,
    gpu::GpuContext,
    loader::TrajectoryLoader,
    video::save_frames_as_video,
    Renderer,
};
use std::{path::PathBuf, time::{Instant, Duration}, sync::{Arc, Mutex}};
use winit::{
    dpi::PhysicalPosition,
    event::{ElementState, Event, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

#[derive(Parser, Debug)]
#[command(name = "physics_renderer")]
#[command(about = "SDF raymarching renderer for physics simulations")]
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
    
    /// Save a single frame to the specified file path (headless mode)
    #[arg(long)]
    save_frame: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args = Args::parse();
    
    // Check if we're in headless mode
    if let Some(output_path) = args.save_frame {
        // Headless rendering path
        println!("Running in headless mode, saving frame to: {}", output_path.display());
        
        // Initialize GPU without window
        let gpu_context = pollster::block_on(GpuContext::new())?;
        let mut renderer = pollster::block_on(Renderer::new(None, &gpu_context, true))?;
        
        // Create a hardcoded test scene
        let test_bodies = vec![
            Body::new_sphere([0.0, 0.0, 0.0], 1.0, 1.0),         // Sphere at origin
            Body::new_box([3.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1.0), // Box to the right
            Body::new_capsule([-3.0, 0.0, 0.0], 1.0, 0.5, 1.0),  // Capsule to the left
        ];
        
        // Update renderer with test bodies
        renderer.update(&gpu_context, &test_bodies);
        
        // Render to texture
        renderer.render_to_texture(&gpu_context);
        
        // Capture frame
        if let Some(frame_data) = renderer.capture_frame(&gpu_context) {
            // Convert BGRA to RGBA
            let width = 800;
            let height = 600;
            let mut rgba_data = vec![0u8; frame_data.len()];
            for i in 0..(width * height) as usize {
                rgba_data[i * 4] = frame_data[i * 4 + 2];     // R
                rgba_data[i * 4 + 1] = frame_data[i * 4 + 1]; // G
                rgba_data[i * 4 + 2] = frame_data[i * 4];     // B
                rgba_data[i * 4 + 3] = frame_data[i * 4 + 3]; // A
            }
            
            // Save as PNG
            image::save_buffer(
                &output_path,
                &rgba_data,
                width,
                height,
                image::ColorType::Rgba8,
            )?;
            
            println!("✅ SDF frame saved to {}", output_path.display());
        } else {
            eprintln!("Failed to capture frame");
            return Err("Frame capture failed".into());
        }
        
        return Ok(());
    }
    
    // Normal windowed mode path
    let mut oracle_trace: Option<Vec<Vec<Body>>> = None;
    let mut gpu_trace: Option<Vec<Vec<Body>>> = None;
    
    if let Some(path) = args.oracle {
        println!("👁️  Loading oracle trace from: {}", path.display());
        let run = TrajectoryLoader::load_trajectory(&path)?;
        let metadata = TrajectoryLoader::get_metadata(&run);
        println!("   Loaded {} frames with {} bodies each", metadata.num_frames, metadata.num_bodies);
        
        // Convert run to Vec<Vec<Body>>
        let mut frames = Vec::new();
        for frame_idx in 0..metadata.num_frames {
            frames.push(TrajectoryLoader::get_bodies_at_frame(&run, frame_idx)?);
        }
        oracle_trace = Some(frames);
    }
    
    if let Some(path) = args.gpu {
        println!("👁️  Loading GPU trace from: {}", path.display());
        let run = TrajectoryLoader::load_trajectory(&path)?;
        let metadata = TrajectoryLoader::get_metadata(&run);
        println!("   Loaded {} frames with {} bodies each", metadata.num_frames, metadata.num_bodies);
        
        // Convert run to Vec<Vec<Body>>
        let mut frames = Vec::new();
        for frame_idx in 0..metadata.num_frames {
            frames.push(TrajectoryLoader::get_bodies_at_frame(&run, frame_idx)?);
        }
        gpu_trace = Some(frames);
    }
    
    let num_frames = oracle_trace.as_ref().map_or(0, |t| t.len()).max(gpu_trace.as_ref().map_or(0, |t| t.len()));
    if num_frames == 0 {
        println!("No trace file loaded, using test scene.");
        // Create a simple test scene
        let test_scene = vec![vec![
            Body::new_sphere([0.0, 0.0, 0.0], 1.0, 1.0),
            Body::new_box([3.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1.0),
            Body::new_capsule([-3.0, 0.0, 0.0], 1.0, 0.5, 1.0),
        ]];
        oracle_trace = Some(test_scene);
    }
    
    // Setup video recording if requested
    let recording = args.record.is_some();
    
    let event_loop = EventLoop::new()?;
    
    // Create window
    let window = WindowBuilder::new()
        .with_title("SDF Physics Renderer")
        .with_inner_size(winit::dpi::PhysicalSize::new(800, 600))
        .build(&event_loop)?;
    
    // For recording, ensure consistent window size
    if recording {
        println!("Initial window size: {:?}", window.inner_size());
        let _ = window.request_inner_size(winit::dpi::PhysicalSize::new(800, 600));
        std::thread::sleep(Duration::from_millis(500));
        
        let current_size = window.inner_size();
        println!("Window size after resize request: {:?}", current_size);
        
        if current_size.width != 800 || current_size.height != 600 {
            println!("Warning: Window resize to 800x600 failed, using {}x{}", current_size.width, current_size.height);
        }
    }
    
    let gpu_context = pollster::block_on(GpuContext::new())?;
    let mut renderer = pollster::block_on(Renderer::new(Some(&window), &gpu_context, recording))?;
    
    let output_path = args.record.clone();
    
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
                    let num_frames = oracle_trace.as_ref().map_or(0, |t| t.len()).max(gpu_trace.as_ref().map_or(0, |t| t.len()));
                    if num_frames > 0 {
                        current_frame = (current_frame + 1) % num_frames;
                    }
                    
                    // Get bodies from current frame
                    let bodies = oracle_trace.as_ref()
                        .and_then(|t| t.get(current_frame))
                        .or_else(|| gpu_trace.as_ref().and_then(|t| t.get(current_frame)))
                        .map(|v| v.as_slice())
                        .unwrap_or(&[]);
                    
                    renderer.update(&gpu_context, bodies);
                    
                    // During recording, we need to render and capture immediately
                    if recording {
                        match renderer.render(&gpu_context) {
                            Ok(_) => {
                                let current_count = captured_frames_clone.lock().unwrap().len();
                                if current_count < max_frames {
                                    if let Some(frame_data) = renderer.capture_frame(&gpu_context) {
                                        if current_count % 30 == 0 {
                                            println!("Captured frame {} of {}", current_count + 1, max_frames);
                                        }
                                        captured_frames_clone.lock().unwrap().push(frame_data);
                                    }
                                }
                            }
                            Err(e) => eprintln!("Render error during recording: {:?}", e),
                        }
                    }
                    
                    last_update = Instant::now();
                    window.request_redraw();
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
                                renderer.resize(&gpu_context, window.inner_size())
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