// SDF (Signed Distance Function) calculations for collision detection

fn sdf_sphere(point: vec3<f32>, center: vec3<f32>, radius: f32) -> f32 {
    return length(point - center) - radius;
}

fn sdf_box(point: vec3<f32>, center: vec3<f32>, half_extents: vec3<f32>) -> f32 {
    let q = abs(point - center) - half_extents;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sdf_capsule(point: vec3<f32>, center: vec3<f32>, radius: f32, height: f32) -> f32 {
    var p = point - center;
    let h = height * 0.5;
    p.y = clamp(p.y, -h, h);
    return length(vec3<f32>(p.x, 0.0, p.z)) - radius;
}

// Transform point from world space to body space
fn world_to_body(world_point: vec3<f32>, body_pos: vec3<f32>, body_quat: vec4<f32>) -> vec3<f32> {
    // For now, just translate (no rotation)
    // TODO: Implement quaternion rotation
    return world_point - body_pos;
}

// Get SDF distance for a body at a world point
fn sdf_body(world_point: vec3<f32>, body_idx: u32) -> f32 {
    let body = bodies[body_idx];
    let local_point = world_to_body(world_point, body.position.xyz, body.orientation);
    
    let shape_type = body.shape_data.x;
    
    if (shape_type == 0u) { // Sphere
        return sdf_sphere(local_point, vec3<f32>(0.0), body.shape_params.x);
    } else if (shape_type == 1u) { // Capsule
        return sdf_capsule(local_point, vec3<f32>(0.0), body.shape_params.x, body.shape_params.y);
    } else if (shape_type == 2u) { // Box
        return sdf_box(local_point, vec3<f32>(0.0), body.shape_params.xyz);
    }
    
    return 1000000.0; // Large distance for unknown shapes
}

// Compute distance between two bodies
fn body_body_distance(idx_a: u32, idx_b: u32) -> f32 {
    let body_a = bodies[idx_a];
    let body_b = bodies[idx_b];
    
    // For spheres, we can compute exact distance
    if (body_a.shape_data.x == 0u && body_b.shape_data.x == 0u) {
        let center_dist = length(body_b.position.xyz - body_a.position.xyz);
        return center_dist - body_a.shape_params.x - body_b.shape_params.x;
    }
    
    // For other shapes, sample SDF
    // This is approximate but good enough for narrow phase
    let sample_point = body_a.position.xyz;
    let dist_in_b = sdf_body(sample_point, idx_b);
    
    return dist_in_b - body_a.shape_params.x; // Subtract radius of A
}

struct Body {
    position: vec4<f32>,
    velocity: vec4<f32>,
    orientation: vec4<f32>,
    angular_vel: vec4<f32>,
    mass_data: vec4<f32>,
    shape_data: vec4<u32>,
    shape_params: vec4<f32>,
}

struct Contact {
    body_a: u32,
    body_b: u32,
    distance: f32,
    normal: vec3<f32>,
    point: vec3<f32>,
    _padding: f32,
}

@group(0) @binding(0) var<storage, read> bodies: array<Body>;
@group(0) @binding(1) var<storage, read_write> contacts: array<Contact>;
@group(0) @binding(2) var<storage, read_write> contact_count: atomic<u32>;

@compute @workgroup_size(64)
fn detect_contacts(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let num_bodies = 2u; // TODO: pass as parameter
    
    // Each thread handles one potential pair
    let total_pairs = num_bodies * (num_bodies - 1u) / 2u;
    if (idx >= total_pairs) {
        return;
    }
    
    // Convert linear index to pair indices
    // Using triangular number formula
    var i = 0u;
    var j = 1u;
    var k = idx;
    
    while (k >= num_bodies - i - 1u) {
        k = k - (num_bodies - i - 1u);
        i = i + 1u;
    }
    j = i + k + 1u;
    
    // Skip if either body is static (both bodies static)
    if (bodies[i].shape_data.y == 1u && bodies[j].shape_data.y == 1u) {
        return;
    }
    
    // Compute distance
    let distance = body_body_distance(i, j);
    
    // If close enough, create contact
    if (distance < 0.1) { // 10cm threshold
        let contact_idx = atomicAdd(&contact_count, 1u);
        if (contact_idx < 100u) { // Max contacts limit
            var contact: Contact;
            contact.body_a = i;
            contact.body_b = j;
            contact.distance = distance;
            
            // Compute contact normal (from A to B)
            let delta = bodies[j].position.xyz - bodies[i].position.xyz;
            let len = length(delta);
            if (len > 0.0001) {
                contact.normal = delta / len;
            } else {
                contact.normal = vec3<f32>(0.0, 1.0, 0.0);
            }
            
            // Contact point (midpoint for now)
            contact.point = (bodies[i].position.xyz + bodies[j].position.xyz) * 0.5;
            
            contacts[contact_idx] = contact;
        }
    }
}