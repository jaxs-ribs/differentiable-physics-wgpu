use clap::{Parser, ValueEnum};
use physics_renderer::loader::{TrajectoryLoader, RunMetadata};
use std::path::PathBuf;

#[derive(Clone, Debug, ValueEnum)]
enum RenderMode {
    Correctness,
    Benchmark,
}

#[derive(Parser, Debug)]
#[command(name = "physics_renderer")]
#[command(about = "A comparative replay renderer for physics simulations")]
#[command(long_about = "Renders physics simulation replays in dual-mode for comparison:\n- Correctness mode: Frame-synchronized comparison\n- Benchmark mode: Real-time performance comparison")]
struct Args {
    /// Primary simulation replay file (.npy format)
    #[arg(short = 'p', long = "primary", help = "Primary simulation .npy file")]
    primary: PathBuf,

    /// Secondary simulation replay file (.npy format) for ghost comparison
    #[arg(short = 's', long = "secondary", help = "Secondary simulation .npy file")]
    secondary: Option<PathBuf>,

    /// Rendering mode
    #[arg(short = 'm', long = "mode", default_value = "correctness", help = "Rendering mode")]
    mode: RenderMode,

    /// Enable verbose logging
    #[arg(short = 'v', long = "verbose", help = "Enable verbose logging")]
    verbose: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Initialize logging
    if args.verbose {
        env_logger::Builder::from_default_env()
            .filter_level(log::LevelFilter::Debug)
            .init();
    } else {
        env_logger::Builder::from_default_env()
            .filter_level(log::LevelFilter::Info)
            .init();
    }

    log::info!("Physics Renderer starting...");
    log::info!("Mode: {:?}", args.mode);
    log::info!("Primary file: {:?}", args.primary);
    if let Some(ref secondary) = args.secondary {
        log::info!("Secondary file: {:?}", secondary);
    }

    // Load primary simulation run
    let primary_run = TrajectoryLoader::load_trajectory(&args.primary)?;
    let primary_metadata = TrajectoryLoader::get_metadata(&primary_run);
    
    println!("✓ Loaded primary simulation:");
    print_run_info(&primary_metadata);
    
    // Load secondary simulation run if provided
    let secondary_metadata = if let Some(secondary_path) = &args.secondary {
        let secondary_run = TrajectoryLoader::load_trajectory(secondary_path)?;
        let metadata = TrajectoryLoader::get_metadata(&secondary_run);
        println!("✓ Loaded secondary simulation:");
        print_run_info(&metadata);
        Some(metadata)
    } else {
        println!("• No secondary simulation provided (single-run mode)");
        None
    };
    
    // Analyze mode implications
    match args.mode {
        RenderMode::Correctness => {
            println!("\n📐 Correctness Mode:");
            println!("  • All runs will be synchronized frame-by-frame");
            if let Some(ref secondary) = secondary_metadata {
                let max_frames = primary_metadata.num_frames.max(secondary.num_frames);
                println!("  • Total playback duration: {} frames", max_frames);
                if primary_metadata.num_frames != secondary.num_frames {
                    println!("  ⚠️  Frame count mismatch detected - runs may not be directly comparable");
                }
            }
        }
        RenderMode::Benchmark => {
            println!("\n⚡ Benchmark Mode:");
            if let Some(ref secondary) = secondary_metadata {
                let faster_duration = primary_metadata.duration_seconds.min(secondary.duration_seconds);
                println!("  • Playback capped at fastest run duration: {:.2}s", faster_duration);
                
                if primary_metadata.duration_seconds < secondary.duration_seconds {
                    println!("  • Primary run is faster - secondary will be cut off");
                } else if secondary.duration_seconds < primary_metadata.duration_seconds {
                    println!("  • Secondary run is faster - primary will be cut off");
                } else {
                    println!("  • Both runs have equal duration");
                }
            } else {
                println!("  • Single run will play at natural duration: {:.2}s", primary_metadata.duration_seconds);
            }
        }
    }
    
    // Test frame loading
    println!("\n🔍 Testing data integrity...");
    let test_frame = 0;
    let bodies = TrajectoryLoader::get_bodies_at_frame(&primary_run, test_frame)?;
    println!("  • Successfully loaded {} bodies from frame {}", bodies.len(), test_frame);
    
    if let Some(body) = bodies.first() {
        println!("  • Sample body position: [{:.2}, {:.2}, {:.2}]", 
                 body.position()[0], body.position()[1], body.position()[2]);
        println!("  • Sample body type: {} ({})", 
                 body.shape_type(), 
                 match body.shape_type() {
                     0 => "sphere",
                     2 => "box", 
                     _ => "unknown"
                 });
    }
    
    println!("\n🚀 Ready to render! (Rendering pipeline not yet implemented)");
    
    // TODO: Phase 3 - Implement actual rendering with visualization
    // For now, we've successfully validated the data loading pipeline
    
    Ok(())
}

fn print_run_info(metadata: &RunMetadata) {
    println!("    File: {}", metadata.file_path);
    println!("    Frames: {} ({:.2}s at 60 FPS)", metadata.num_frames, metadata.duration_seconds);
    println!("    Bodies: {}", metadata.num_bodies);
}