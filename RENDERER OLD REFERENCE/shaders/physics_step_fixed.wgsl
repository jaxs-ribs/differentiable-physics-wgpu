// Fixed physics step shader with proper alignment

struct Body {
    position: vec4<f32>,
    velocity: vec4<f32>,
    orientation: vec4<f32>,
    angular_vel: vec4<f32>,
    mass_data: vec4<f32>,      // mass, inv_mass, padding, padding
    shape_data: vec4<u32>,     // shape_type, flags, padding, padding
    shape_params: vec4<f32>,
}

struct SimParams {
    dt: f32,
    gravity_x: f32,
    gravity_y: f32, 
    gravity_z: f32,
    num_bodies: u32,
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
}

@group(0) @binding(0) var<storage, read_write> bodies: array<Body>;
@group(0) @binding(1) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn physics_step(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let num_bodies = params.num_bodies;
    
    if (idx >= num_bodies) {
        return;
    }
    
    // Skip static bodies (flag = 1)
    if (bodies[idx].shape_data.y == 1u) {
        return;
    }
    
    // Use params from uniform buffer
    let dt = params.dt;
    let gravity = vec3<f32>(params.gravity_x, params.gravity_y, params.gravity_z);
    
    // Apply gravity (only if mass > 0)
    if (bodies[idx].mass_data.x > 0.0) {
        bodies[idx].velocity.x = bodies[idx].velocity.x + gravity.x * dt;
        bodies[idx].velocity.y = bodies[idx].velocity.y + gravity.y * dt;
        bodies[idx].velocity.z = bodies[idx].velocity.z + gravity.z * dt;
    }
    
    // Update position
    bodies[idx].position.x = bodies[idx].position.x + bodies[idx].velocity.x * dt;
    bodies[idx].position.y = bodies[idx].position.y + bodies[idx].velocity.y * dt;
    bodies[idx].position.z = bodies[idx].position.z + bodies[idx].velocity.z * dt;
    
    // Normalize quaternion to prevent drift
    bodies[idx].orientation = normalize(bodies[idx].orientation);
}