/*
Complete Physics Pipeline Integration Test

This test validates the entire physics simulation pipeline by running a complete step including
integration, broadphase, narrowphase collision detection, and contact resolution. It serves as
an end-to-end validation that all GPU kernels work together correctly and produces JSON output
for cross-validation with Python reference implementations.
*/

use physics_core::{body::Body, physics::PhysicsEngine};
use pollster::block_on;
use std::io::{self, Read};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    // Read test scene from stdin (JSON format)
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    
    // For now, create a simple test scene
    let bodies = vec![
        Body::new_sphere([0.0, 5.0, 0.0], 0.5, 1.0),
        Body::new_static_sphere([0.0, -100.0, 0.0], 100.0),
    ];
    
    // Create engine and run one step
    let engine = block_on(PhysicsEngine::new(bodies))?;
    engine.step();
    
    // Read back results
    let results = block_on(engine.read_bodies());
    
    // Output as JSON with more precision
    println!("{{");
    println!("  \"bodies\": [");
    for (i, body) in results.iter().enumerate() {
        println!("    {{");
        println!("      \"position\": [{:.6}, {:.6}, {:.6}],", body.position[0], body.position[1], body.position[2]);
        println!("      \"velocity\": [{:.6}, {:.6}, {:.6}],", body.velocity[0], body.velocity[1], body.velocity[2]);
        println!("      \"mass\": {:.6}", body.mass_data[0]);
        print!("    }}");
        if i < results.len() - 1 {
            println!(",");
        } else {
            println!();
        }
    }
    println!("  ]");
    println!("}}");
    
    Ok(())
}