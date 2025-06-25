use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        print_usage();
        return;
    }
    
    print_usage();
    println!("\nTo run specific commands, use the physics_core wrapper script or cargo directly:");
    println!("  ./physics_core benchmark 10000");
    println!("  cargo run --bin benchmark -- 10000");
}

fn print_usage() {
    println!("Physics Core - WebGPU-accelerated rigid body physics engine");
    println!();
    println!("This is the main binary. For a unified CLI, use the ./physics_core wrapper script.");
    println!();
    println!("Available commands via wrapper:");
    println!("  ./physics_core benchmark [bodies]     Run performance benchmarks");
    println!("  ./physics_core demo <type>           Run demos:");
    println!("    simple              Console output demo");
    println!("    ascii               ASCII visualization");
    println!("    viz                 Wireframe visualization");
    println!("  ./physics_core test <component>      Run component tests:");
    println!("    sdf                 Test collision detection");
    println!("    contact             Test contact solver");
    println!("    broadphase          Test broad phase");
    println!("    runner              Generic test runner");
    println!("  ./physics_core help                  Show help message");
    println!();
    println!("Or run binaries directly:");
    println!("  cargo run --bin benchmark");
    println!("  cargo run --bin demo_simple");
    println!("  cargo run --bin demo_ascii");
    println!("  cargo run --features viz --bin demo_viz");
    println!("  cargo run --bin test_sdf");
    println!("  etc.");
}
