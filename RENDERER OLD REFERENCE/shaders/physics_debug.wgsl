// Debug physics shader - just applies gravity without params

struct Body {
    position: vec4<f32>,
    velocity: vec4<f32>,
    orientation: vec4<f32>,
    angular_vel: vec4<f32>,
    mass_data: vec4<f32>,
    shape_data: vec4<u32>,
    shape_params: vec4<f32>,
}

@group(0) @binding(0) var<storage, read_write> bodies: array<Body>;

@compute @workgroup_size(64)
fn physics_step(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let num_bodies = arrayLength(&bodies);
    
    if (idx >= num_bodies) {
        return;
    }
    
    // Skip static bodies
    if (bodies[idx].shape_data.y == 1u) {
        return;
    }
    
    // Hardcoded values for debugging
    let dt = 0.016;
    let gravity = vec3<f32>(0.0, -9.81, 0.0);
    
    // Apply gravity
    bodies[idx].velocity.y = bodies[idx].velocity.y + gravity.y * dt;
    
    // Update position
    bodies[idx].position.y = bodies[idx].position.y + bodies[idx].velocity.y * dt;
}