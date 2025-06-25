// Contact solver using soft penalty method

struct Body {
    position: vec4<f32>,
    velocity: vec4<f32>,
    orientation: vec4<f32>,
    angular_vel: vec4<f32>,
    mass_data: vec4<f32>,      // mass, inv_mass, padding, padding
    shape_data: vec4<u32>,     // shape_type, flags, padding, padding
    shape_params: vec4<f32>,
}

struct Contact {
    body_a: u32,
    body_b: u32,
    distance: f32,
    _padding1: f32,
    normal: vec4<f32>,
    point: vec4<f32>,
}

struct SolverParams {
    values: vec4<f32>,      // dt, stiffness, damping, restitution
    counts: vec4<u32>,      // num_contacts, 0, 0, 0
    _padding: vec4<f32>,
}

@group(0) @binding(0) var<storage, read_write> bodies: array<Body>;
@group(0) @binding(1) var<storage, read> contacts: array<Contact>;
@group(0) @binding(2) var<uniform> params: SolverParams;

@compute @workgroup_size(64)
fn solve_contacts(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.counts.x) {
        return;
    }
    
    let contact = contacts[idx];
    let body_a = bodies[contact.body_a];
    let body_b = bodies[contact.body_b];
    
    // Skip if both bodies are static
    if (body_a.shape_data.y == 1u && body_b.shape_data.y == 1u) {
        return;
    }
    
    // Only process penetrating contacts
    if (contact.distance >= 0.0) {
        return;
    }
    
    // Penalty force magnitude
    let penetration = -contact.distance;
    let force_magnitude = params.values.y * penetration; // stiffness
    
    // Relative velocity at contact point
    let vel_a = body_a.velocity.xyz;
    let vel_b = body_b.velocity.xyz;
    let relative_vel = vel_a - vel_b; // Velocity of A relative to B
    
    // Velocity along normal (positive = approaching)
    let vel_normal = dot(relative_vel, contact.normal.xyz);
    
    // Damping force should oppose penetration velocity
    // If bodies are approaching (vel_normal < 0), damping adds to penalty force
    let damping_force = -params.values.z * vel_normal; // damping
    
    // Total force along normal
    let total_force = force_magnitude + damping_force;
    
    // Apply forces (F = ma, so a = F/m)
    let dt = params.values.x; // dt
    if (body_a.mass_data.x > 0.0) {
        let impulse_a = -contact.normal.xyz * total_force * dt;
        bodies[contact.body_a].velocity.x += impulse_a.x * body_a.mass_data.y; // Use inv_mass
        bodies[contact.body_a].velocity.y += impulse_a.y * body_a.mass_data.y;
        bodies[contact.body_a].velocity.z += impulse_a.z * body_a.mass_data.y;
    }
    
    if (body_b.mass_data.x > 0.0) {
        let impulse_b = contact.normal.xyz * total_force * dt;
        bodies[contact.body_b].velocity.x += impulse_b.x * body_b.mass_data.y; // Use inv_mass
        bodies[contact.body_b].velocity.y += impulse_b.y * body_b.mass_data.y;
        bodies[contact.body_b].velocity.z += impulse_b.z * body_b.mass_data.y;
    }
    
    // Simple position correction to prevent sinking
    let correction_percent = 0.2; // Usually 20% to 80%
    let slop = 0.01; // Usually 0.01 to 0.1
    let correction = max(penetration - slop, 0.0) * correction_percent;
    
    if (body_a.mass_data.x > 0.0) {
        let pos_correction_a = -contact.normal.xyz * correction * 0.5;
        bodies[contact.body_a].position.x += pos_correction_a.x;
        bodies[contact.body_a].position.y += pos_correction_a.y;
        bodies[contact.body_a].position.z += pos_correction_a.z;
    }
    
    if (body_b.mass_data.x > 0.0) {
        let pos_correction_b = contact.normal.xyz * correction * 0.5;
        bodies[contact.body_b].position.x += pos_correction_b.x;
        bodies[contact.body_b].position.y += pos_correction_b.y;
        bodies[contact.body_b].position.z += pos_correction_b.z;
    }
}