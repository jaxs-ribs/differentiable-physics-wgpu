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
    loader::{TrajectoryLoader, TrajectoryRun},
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
    
    if let Some(output_path) = args.save_frame {
        return run_headless(output_path);
    }
    
    let (oracle_trace, gpu_trace) = load_traces(&args)?;
    let simulation_frames = get_frame_count(&oracle_trace, &gpu_trace);
    
    let recording = args.record.is_some();
    let event_loop = EventLoop::new()?;
    let window = create_window(&event_loop, recording)?;
    
    let gpu_context = pollster::block_on(GpuContext::new())?;
    let mut renderer = pollster::block_on(Renderer::new(Some(&window), &gpu_context, recording))?;
    
    let mut app_state = ApplicationState {
        oracle_trace,
        gpu_trace,
        current_frame: 0,
        recording,
        recording_fps: args.fps,
        frame_duration: calculate_frame_duration(recording, args.fps),
        last_update: Instant::now(),
        max_frames: calculate_max_frames(recording, simulation_frames, args.fps, args.duration),
        mouse_pressed: false,
        last_mouse_pos: PhysicalPosition::new(0.0, 0.0),
        captured_frames: Arc::new(Mutex::new(Vec::<Vec<u8>>::new())),
        output_path: args.record.clone(),
    };
    
    let captured_frames_clone = app_state.captured_frames.clone();
    let captured_frames_final = app_state.captured_frames.clone();
    let recording = app_state.recording;
    let output_path = app_state.output_path.clone();
    let recording_fps = app_state.recording_fps;
    
    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::AboutToWait => {
                handle_frame_update(
                    &mut app_state,
                    &mut renderer,
                    &gpu_context,
                    &window,
                    elwt,
                    &captured_frames_clone,
                );
            }
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => elwt.exit(),
                WindowEvent::Resized(physical_size) => {
                    renderer.resize(&gpu_context, physical_size);
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    handle_mouse_input(&mut app_state, state, button);
                }
                WindowEvent::CursorMoved { position, .. } => {
                    handle_cursor_moved(&mut app_state, &mut renderer, position);
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    handle_mouse_wheel(&mut renderer, delta);
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
                    handle_redraw(&app_state, &mut renderer, &gpu_context, &window, elwt);
                }
                _ => {}
            },
            _ => {}
        }
    })?;
    
    if recording {
        if let Some(output_path) = output_path {
            let frames = captured_frames_final.lock().unwrap();
            save_frames_as_video(
                &*frames,
                output_path,
                recording_fps,
                DEFAULT_WINDOW_WIDTH,
                DEFAULT_WINDOW_HEIGHT,
            )?;
        }
    }
    Ok(())
}

const MOUSE_SENSITIVITY: f32 = 0.01;
const DEFAULT_WINDOW_WIDTH: u32 = 800;
const DEFAULT_WINDOW_HEIGHT: u32 = 600;

struct ApplicationState {
    oracle_trace: Option<Vec<Vec<Body>>>,
    gpu_trace: Option<Vec<Vec<Body>>>,
    current_frame: usize,
    recording: bool,
    recording_fps: u32,
    frame_duration: Duration,
    last_update: Instant,
    max_frames: usize,
    mouse_pressed: bool,
    last_mouse_pos: PhysicalPosition<f64>,
    captured_frames: Arc<Mutex<Vec<Vec<u8>>>>,
    output_path: Option<PathBuf>,
}

fn run_headless(output_path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    println!("Running in headless mode, saving frame to: {}", output_path.display());
    
    let (gpu_context, mut renderer) = initialize_headless_renderer()?;
    render_test_scene(&gpu_context, &mut renderer);
    capture_and_save_frame(&gpu_context, &renderer, &output_path)?;
    
    Ok(())
}

fn initialize_headless_renderer() -> Result<(GpuContext, Renderer), Box<dyn std::error::Error>> {
    let gpu_context = pollster::block_on(GpuContext::new())?;
    let renderer = pollster::block_on(Renderer::new(None, &gpu_context, true))?;
    Ok((gpu_context, renderer))
}

fn render_test_scene(gpu_context: &GpuContext, renderer: &mut Renderer) {
    let test_bodies = create_test_scene();
    renderer.update(gpu_context, &test_bodies);
    renderer.render_to_texture(gpu_context);
}

fn capture_and_save_frame(
    gpu_context: &GpuContext,
    renderer: &Renderer,
    output_path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let frame_data = renderer.capture_frame(gpu_context)
        .ok_or("Frame capture failed")?;
    
    let rgba_data = convert_bgra_to_rgba(&frame_data, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT);
    save_frame_as_png(output_path, &rgba_data, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)?;
    
    println!("✅ SDF frame saved to {}", output_path.display());
    Ok(())
}

fn create_test_scene() -> Vec<Body> {
    vec![
        Body::new_sphere([0.0, 0.0, 0.0], 1.0, 1.0),
        Body::new_box([3.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1.0),
        Body::new_capsule([-3.0, 0.0, 0.0], 1.0, 0.5, 1.0),
    ]
}

fn convert_bgra_to_rgba(bgra_data: &[u8], width: u32, height: u32) -> Vec<u8> {
    let pixel_count = (width * height) as usize;
    let mut rgba_data = vec![0u8; bgra_data.len()];
    
    for i in 0..pixel_count {
        let offset = i * 4;
        copy_pixel_bgra_to_rgba(&bgra_data[offset..], &mut rgba_data[offset..]);
    }
    
    rgba_data
}

fn copy_pixel_bgra_to_rgba(bgra_pixel: &[u8], rgba_pixel: &mut [u8]) {
    rgba_pixel[0] = bgra_pixel[2]; // R <- B
    rgba_pixel[1] = bgra_pixel[1]; // G <- G
    rgba_pixel[2] = bgra_pixel[0]; // B <- R
    rgba_pixel[3] = bgra_pixel[3]; // A <- A
}

fn save_frame_as_png(
    path: &PathBuf,
    data: &[u8],
    width: u32,
    height: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    image::save_buffer(path, data, width, height, image::ColorType::Rgba8)?;
    Ok(())
}

fn load_traces(args: &Args) -> Result<(Option<Vec<Vec<Body>>>, Option<Vec<Vec<Body>>>), Box<dyn std::error::Error>> {
    let oracle_trace = load_trace_file(&args.oracle, "oracle")?;
    let gpu_trace = load_trace_file(&args.gpu, "GPU")?;
    
    let (oracle_trace, gpu_trace) = ensure_has_frames(oracle_trace, gpu_trace);
    Ok((oracle_trace, gpu_trace))
}

fn load_trace_file(
    path: &Option<PathBuf>,
    trace_type: &str,
) -> Result<Option<Vec<Vec<Body>>>, Box<dyn std::error::Error>> {
    let Some(path) = path else {
        return Ok(None);
    };
    
    print_loading_message(trace_type, path);
    let run = TrajectoryLoader::load_trajectory(path)?;
    let frames = extract_frames_from_trajectory(&run)?;
    
    Ok(Some(frames))
}

fn print_loading_message(trace_type: &str, path: &PathBuf) {
    println!("👁️  Loading {} trace from: {}", trace_type, path.display());
}

fn extract_frames_from_trajectory(
    run: &TrajectoryRun,
) -> Result<Vec<Vec<Body>>, Box<dyn std::error::Error>> {
    let metadata = TrajectoryLoader::get_metadata(run);
    println!("   Loaded {} frames with {} bodies each", metadata.num_frames, metadata.num_bodies);
    
    Ok((0..metadata.num_frames)
        .map(|idx| TrajectoryLoader::get_bodies_at_frame(run, idx))
        .collect::<Result<Vec<_>, _>>()?)
}

fn ensure_has_frames(
    oracle_trace: Option<Vec<Vec<Body>>>,
    gpu_trace: Option<Vec<Vec<Body>>>,
) -> (Option<Vec<Vec<Body>>>, Option<Vec<Vec<Body>>>) {
    let has_frames = oracle_trace.as_ref().map_or(false, |t| !t.is_empty())
        || gpu_trace.as_ref().map_or(false, |t| !t.is_empty());
    
    if !has_frames {
        println!("No trace file loaded, using test scene.");
        (Some(vec![create_test_scene()]), gpu_trace)
    } else {
        (oracle_trace, gpu_trace)
    }
}

fn get_frame_count(
    oracle_trace: &Option<Vec<Vec<Body>>>,
    gpu_trace: &Option<Vec<Vec<Body>>>,
) -> usize {
    oracle_trace.as_ref().map_or(0, |t| t.len())
        .max(gpu_trace.as_ref().map_or(0, |t| t.len()))
}

fn create_window(
    event_loop: &EventLoop<()>,
    recording: bool,
) -> Result<winit::window::Window, Box<dyn std::error::Error>> {
    let window = WindowBuilder::new()
        .with_title("SDF Physics Renderer")
        .with_inner_size(winit::dpi::PhysicalSize::new(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT))
        .build(event_loop)?;
    
    if recording {
        ensure_window_size(&window, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT);
    }
    
    Ok(window)
}

fn ensure_window_size(window: &winit::window::Window, width: u32, height: u32) {
    println!("Initial window size: {:?}", window.inner_size());
    let _ = window.request_inner_size(winit::dpi::PhysicalSize::new(width, height));
    std::thread::sleep(Duration::from_millis(500));
    
    let current_size = window.inner_size();
    println!("Window size after resize request: {:?}", current_size);
    
    if current_size.width != width || current_size.height != height {
        println!("Warning: Window resize to {}x{} failed, using {}x{}",
                 width, height, current_size.width, current_size.height);
    }
}

fn calculate_frame_duration(recording: bool, fps: u32) -> Duration {
    if recording {
        Duration::from_millis(1000 / fps as u64)
    } else {
        Duration::from_millis(16) // ~60 FPS for interactive mode
    }
}

fn calculate_max_frames(
    recording: bool,
    simulation_frames: usize,
    fps: u32,
    duration: f32,
) -> usize {
    if recording {
        simulation_frames
    } else {
        (fps as f32 * duration) as usize
    }
}

fn handle_frame_update(
    state: &mut ApplicationState,
    renderer: &mut Renderer,
    gpu_context: &GpuContext,
    window: &winit::window::Window,
    elwt: &winit::event_loop::EventLoopWindowTarget<()>,
    captured_frames: &Arc<Mutex<Vec<Vec<u8>>>>,
) {
    if should_exit_recording(state, captured_frames) {
        elwt.exit();
        return;
    }
    
    if !is_time_for_next_frame(state) {
        return;
    }
    
    if !advance_simulation_frame(state, elwt) {
        return;
    }
    
    update_and_render_frame(state, renderer, gpu_context, captured_frames);
    request_next_frame(state, window);
}

fn is_time_for_next_frame(state: &ApplicationState) -> bool {
    state.last_update.elapsed() >= state.frame_duration
}

fn advance_simulation_frame(
    state: &mut ApplicationState,
    elwt: &winit::event_loop::EventLoopWindowTarget<()>,
) -> bool {
    match advance_frame(state) {
        Some(new_frame) => {
            state.current_frame = new_frame;
            true
        }
        None => {
            elwt.exit();
            false
        }
    }
}

fn update_and_render_frame(
    state: &ApplicationState,
    renderer: &mut Renderer,
    gpu_context: &GpuContext,
    captured_frames: &Arc<Mutex<Vec<Vec<u8>>>>,
) {
    let bodies = get_current_frame_bodies(state);
    renderer.update(gpu_context, bodies);
    
    if state.recording {
        capture_frame_for_recording(renderer, gpu_context, captured_frames, state.max_frames);
    }
}

fn request_next_frame(state: &mut ApplicationState, window: &winit::window::Window) {
    state.last_update = Instant::now();
    window.request_redraw();
}

fn should_exit_recording(
    state: &ApplicationState,
    captured_frames: &Arc<Mutex<Vec<Vec<u8>>>>,
) -> bool {
    state.recording && captured_frames.lock().unwrap().len() >= state.max_frames
}

fn advance_frame(state: &ApplicationState) -> Option<usize> {
    let num_frames = get_frame_count(&state.oracle_trace, &state.gpu_trace);
    if num_frames == 0 {
        return Some(state.current_frame);
    }
    
    if state.recording {
        let next = state.current_frame + 1;
        if next >= num_frames {
            None
        } else {
            Some(next)
        }
    } else {
        Some((state.current_frame + 1) % num_frames)
    }
}

fn get_current_frame_bodies(state: &ApplicationState) -> &[Body] {
    state.oracle_trace.as_ref()
        .and_then(|t| t.get(state.current_frame))
        .or_else(|| state.gpu_trace.as_ref().and_then(|t| t.get(state.current_frame)))
        .map(|v| v.as_slice())
        .unwrap_or(&[])
}

fn capture_frame_for_recording(
    renderer: &mut Renderer,
    gpu_context: &GpuContext,
    captured_frames: &Arc<Mutex<Vec<Vec<u8>>>>,
    max_frames: usize,
) {
    if let Err(e) = renderer.render(gpu_context) {
        eprintln!("Render error during recording: {:?}", e);
        return;
    }
    
    store_captured_frame(renderer, gpu_context, captured_frames, max_frames);
}

fn store_captured_frame(
    renderer: &Renderer,
    gpu_context: &GpuContext,
    captured_frames: &Arc<Mutex<Vec<Vec<u8>>>>,
    max_frames: usize,
) {
    let mut frames = captured_frames.lock().unwrap();
    
    if frames.len() >= max_frames {
        return;
    }
    
    if let Some(frame_data) = renderer.capture_frame(gpu_context) {
        report_capture_progress(&frames, max_frames);
        frames.push(frame_data);
    }
}

fn report_capture_progress(frames: &[Vec<u8>], max_frames: usize) {
    const REPORT_INTERVAL: usize = 30;
    if frames.len() % REPORT_INTERVAL == 0 {
        println!("Captured frame {} of {}", frames.len() + 1, max_frames);
    }
}

fn handle_mouse_input(
    state: &mut ApplicationState,
    element_state: ElementState,
    button: MouseButton,
) {
    if button == MouseButton::Left {
        state.mouse_pressed = element_state == ElementState::Pressed;
    }
}

fn handle_cursor_moved(
    state: &mut ApplicationState,
    renderer: &mut Renderer,
    position: PhysicalPosition<f64>,
) {
    if state.mouse_pressed {
        let delta_x = (position.x - state.last_mouse_pos.x) as f32 * MOUSE_SENSITIVITY;
        let delta_y = (position.y - state.last_mouse_pos.y) as f32 * MOUSE_SENSITIVITY;
        renderer.camera_mut().rotate(-delta_x, -delta_y);
    }
    state.last_mouse_pos = position;
}

fn handle_mouse_wheel(renderer: &mut Renderer, delta: MouseScrollDelta) {
    let scroll_amount = match delta {
        MouseScrollDelta::LineDelta(_, y) => y,
        MouseScrollDelta::PixelDelta(pos) => pos.y as f32,
    };
    renderer.camera_mut().zoom(scroll_amount);
}

fn handle_redraw(
    state: &ApplicationState,
    renderer: &mut Renderer,
    gpu_context: &GpuContext,
    window: &winit::window::Window,
    elwt: &winit::event_loop::EventLoopWindowTarget<()>,
) {
    if !state.recording {
        match renderer.render(gpu_context) {
            Ok(_) => {}
            Err(wgpu::SurfaceError::Lost) => {
                renderer.resize(gpu_context, window.inner_size())
            }
            Err(wgpu::SurfaceError::OutOfMemory) => elwt.exit(),
            Err(e) => eprintln!("Render error: {:?}", e),
        }
    }
}

