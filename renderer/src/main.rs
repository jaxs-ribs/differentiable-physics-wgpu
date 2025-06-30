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
    /// Simulation files to render, with their properties.
    /// Format: <path> --color <r> <g> <b> --alpha <a>
    #[arg(num_args = 1.., value_parser = clap::value_parser!(String))]
    simulations: Vec<String>,

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

struct SimulationTrace {
    frames: Vec<Vec<Body>>,
    color: [f32; 3],
    alpha: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args = Args::parse();
    
    if let Some(output_path) = args.save_frame {
        return run_headless(output_path);
    }
    
    let traces = load_traces(&args.simulations)?;
    let simulation_frames = traces.iter().map(|t| t.frames.len()).max().unwrap_or(0);
    
    let recording = args.record.is_some();
    let event_loop = EventLoop::new()?;
    let window = create_window(&event_loop, recording)?;
    
    let gpu_context = pollster::block_on(GpuContext::new())?;
    let mut renderer = pollster::block_on(Renderer::new(Some(&window), &gpu_context, recording, traces.len()))?;
    
    let mut app_state = ApplicationState {
        traces,
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
    traces: Vec<SimulationTrace>,
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

fn load_traces(args: &[String]) -> Result<Vec<SimulationTrace>, Box<dyn std::error::Error>> {
    let mut traces = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let path = PathBuf::from(&args[i]);
        i += 1;

        let mut color = [1.0, 1.0, 1.0];
        if i + 4 <= args.len() && args[i] == "--color" {
            color = [
                args[i + 1].parse()?,
                args[i + 2].parse()?,
                args[i + 3].parse()?,
            ];
            i += 4;
        }

        let mut alpha = 1.0;
        if i + 2 <= args.len() && args[i] == "--alpha" {
            alpha = args[i + 1].parse()?;
            i += 2;
        }

        let run = TrajectoryLoader::load_trajectory(&path)?;
        let frames = extract_frames_from_trajectory(&run)?;
        traces.push(SimulationTrace { frames, color, alpha });
    }

    if traces.is_empty() {
        println!("No trace file loaded, using test scene.");
        traces.push(SimulationTrace {
            frames: vec![create_test_scene()],
            color: [1.0, 0.0, 0.0],
            alpha: 1.0,
        });
    }

    Ok(traces)
}

fn extract_frames_from_trajectory(
    run: &TrajectoryRun,
) -> Result<Vec<Vec<Body>>, Box<dyn std::error::Error>> {
    let metadata = TrajectoryLoader::get_metadata(run);
    println!("   Loaded {} frames with {} bodies each", metadata.num_frames, metadata.num_bodies);
    
    (0..metadata.num_frames)
        .map(|idx| TrajectoryLoader::get_bodies_at_frame(run, idx))
        .collect::<Result<Vec<_>, _>>()
}

fn advance_frame(state: &mut ApplicationState) -> Option<usize> {
    let num_frames = state.traces.iter().map(|t| t.frames.len()).max().unwrap_or(0);
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

fn update_and_render_frame(
    state: &ApplicationState,
    renderer: &mut Renderer,
    gpu_context: &GpuContext,
    captured_frames: &Arc<Mutex<Vec<Vec<u8>>>>,
) {
    let bodies_per_trace: Vec<_> = state.traces.iter()
        .map(|trace| trace.frames.get(state.current_frame).map_or(&[][..], |f| f.as_slice()))
        .collect();
    
    let colors: Vec<_> = state.traces.iter().map(|t| t.color).collect();
    let alphas: Vec<_> = state.traces.iter().map(|t| t.alpha).collect();

    renderer.update(gpu_context, &bodies_per_trace, &colors, &alphas);
    
    if state.recording {
        capture_frame_for_recording(renderer, gpu_context, captured_frames, state.max_frames);
    }
}

