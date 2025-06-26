use clap::Parser;
use physics_core::{
    body::Body,
    gpu::GpuContext,
    viz::{WindowManager, DualRenderer},
};
use std::path::PathBuf;
use winit::{
    event::{ElementState, Event, KeyEvent, WindowEvent},
    event_loop::EventLoop,
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

fn load_bodies_from_npy(path: &PathBuf) -> Result<Vec<Body>, Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::io::Read;
    
    // Read the file
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    
    // Use npy crate to parse
    let reader = npy::NpyData::from_bytes(&buffer)?;
    let data: Vec<f32> = reader.to_vec();
    
    // Each body has: 3 pos + 3 vel + 4 quat + 3 angular_vel + 1 mass + 1 shape_type + 3 shape_params = 18 floats
    const FLOATS_PER_BODY: usize = 18;
    let num_bodies = data.len() / FLOATS_PER_BODY;
    
    let mut bodies = Vec::new();
    
    for i in 0..num_bodies {
        let offset = i * FLOATS_PER_BODY;
        
        let position = [data[offset], data[offset + 1], data[offset + 2], 0.0];
        let velocity = [data[offset + 3], data[offset + 4], data[offset + 5], 0.0];
        let orientation = [data[offset + 6], data[offset + 7], data[offset + 8], data[offset + 9]];
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
            shape_data: [shape_type, 0, 0, 0], // shape_type, flags=0
            shape_params,
        });
    }
    
    Ok(bodies)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args = Args::parse();
    
    // Load bodies from npy files if provided
    let oracle_bodies = args.oracle.as_ref()
        .and_then(|path| load_bodies_from_npy(path).ok());
    let gpu_bodies = args.gpu.as_ref()
        .and_then(|path| load_bodies_from_npy(path).ok());
    
    if oracle_bodies.is_none() && gpu_bodies.is_none() {
        eprintln!("Warning: No state files provided. Use --oracle and/or --gpu to load states.");
        eprintln!("Example: cargo run --features viz --bin debug_viz -- --oracle cpu_state.npy --gpu gpu_state.npy");
    }
    
    // Create window and event loop
    let event_loop = EventLoop::new()?;
    let window_manager = WindowManager::new(&event_loop)?;
    
    // Create GPU context and renderer
    let gpu_context = pollster::block_on(GpuContext::new())?;
    let mut renderer = pollster::block_on(DualRenderer::new(&window_manager, &gpu_context))?;
    
    // Update scenes with loaded bodies
    if oracle_bodies.is_some() || gpu_bodies.is_some() {
        renderer.update_scenes(&gpu_context, oracle_bodies.as_deref(), gpu_bodies.as_deref());
    }
    
    // State for debug toggles
    let mut show_aabbs = false;
    let mut show_contacts = false;
    
    let _ = event_loop.run(move |event, elwt| {
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    elwt.exit();
                }
                WindowEvent::Resized(physical_size) => {
                    renderer.resize(&gpu_context, physical_size);
                }
                WindowEvent::KeyboardInput {
                    event: KeyEvent {
                        physical_key,
                        state: ElementState::Pressed,
                        ..
                    },
                    ..
                } => {
                    match physical_key {
                        winit::keyboard::PhysicalKey::Code(code) => {
                            match code {
                                winit::keyboard::KeyCode::KeyB => {
                                    show_aabbs = !show_aabbs;
                                    println!("AABBs: {}", if show_aabbs { "ON" } else { "OFF" });
                                    renderer.set_debug_aabbs(show_aabbs);
                                }
                                winit::keyboard::KeyCode::KeyC => {
                                    show_contacts = !show_contacts;
                                    println!("Contacts: {}", if show_contacts { "ON" } else { "OFF" });
                                    renderer.set_debug_contacts(show_contacts);
                                }
                                winit::keyboard::KeyCode::Escape => {
                                    elwt.exit();
                                }
                                _ => {}
                            }
                        }
                        _ => {}
                    }
                }
                WindowEvent::RedrawRequested => {
                    match renderer.render(&gpu_context) {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost) => renderer.resize(&gpu_context, window_manager.inner_size()),
                        Err(wgpu::SurfaceError::OutOfMemory) => elwt.exit(),
                        Err(e) => eprintln!("Render error: {:?}", e),
                    }
                }
                _ => {}
            },
            Event::AboutToWait => {
                window_manager.window().request_redraw();
            }
            _ => {}
        }
    })?;
    
    Ok(())
}