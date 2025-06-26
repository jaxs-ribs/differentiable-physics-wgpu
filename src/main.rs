fn main() {
    print_usage();
}

fn print_usage() {
    println!("Physics Core - WebGPU-accelerated rigid body physics engine");
    println!();
    println!("Run binaries directly with cargo:");
    println!("");
    println!("Benchmarks:");
    println!("  cargo run --bin benchmark");
    println!("  cargo run --bin benchmark_full");
    println!("");
    println!("Demos:");
    println!("  cargo run --bin demo_simple");
    println!("  cargo run --bin demo_ascii");
    println!("  cargo run --features viz --bin demo_viz");
    println!("");
    println!("Tests:");
    println!("  cargo run --bin test_sdf");
    println!("  cargo run --bin test_contact_solver");
    println!("  cargo run --bin test_broadphase");
    println!("  cargo run --bin test_broadphase_grid");
    println!("  cargo run --bin test_runner");
    println!("");
    println!("Visualization tests (require --features viz):");
    println!("  cargo run --features viz --bin test_viz_simple");
    println!("  cargo run --features viz --bin test_viz_pipeline");
    println!("  cargo run --features viz --bin test_viz_triangle");
    println!("  cargo run --features viz --bin test_viz_debug");
    println!("  cargo run --features viz --bin test_viz_debug_vertices");
    println!("  cargo run --features viz --bin test_wireframe_direct");
    println!("  cargo run --features viz --bin test_matrix_debug");
    println!("");
    println!("Matrix tests:");
    println!("  cargo run --bin test_matrix_column");
}
