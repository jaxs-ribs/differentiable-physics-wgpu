#[cfg(not(feature = "viz"))]
compile_error!("This test requires the 'viz' feature");

fn main() {
    // Simple test of matrix math
    let eye = [0.0, 10.0, 20.0];
    let target = [0.0, 5.0, 0.0];
    
    // Forward vector (from eye to target)
    let forward = [
        target[0] - eye[0],  // 0 - 0 = 0
        target[1] - eye[1],  // 5 - 10 = -5
        target[2] - eye[2],  // 0 - 20 = -20
    ];
    
    println!("Camera Setup:");
    println!("  Eye: {:?}", eye);
    println!("  Target: {:?}", target);
    println!("  Forward (unnormalized): {:?}", forward);
    
    // Test a simple point transformation
    // Point at origin in world space
    let world_point = [0.0, 0.0, 0.0, 1.0];
    
    // In view space, this point should be at:
    // Relative to camera: (0 - 0, 0 - 10, 0 - 20) = (0, -10, -20)
    println!("\nExpected view space position: (0, -10, -20)");
    
    // Point at target in world space
    let target_point = [0.0, 5.0, 0.0, 1.0];
    // In view space: (0 - 0, 5 - 10, 0 - 20) = (0, -5, -20)
    println!("Target in view space should be: (0, -5, -20)");
    
    // With a 45 degree FOV and the point at z=-20:
    // The visible range at z=-20 is approximately ±20 units in x and y
    // So our points should be visible
    
    println!("\nSimple perspective check:");
    let z: f32 = -20.0;
    let fov = 45.0_f32.to_radians();
    let visible_range = z.abs() * (fov / 2.0).tan();
    println!("  At z={}, visible range is ±{:.1} units", z, visible_range);
}