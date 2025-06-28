// SDF Raymarching Shader for Physics Renderer

struct ViewProjection {
    matrix: mat4x4<f32>,
}

struct Body {
    position: vec4<f32>,      // xyz position, w unused
    velocity: vec4<f32>,      // xyz velocity, w unused
    orientation: vec4<f32>,   // xyzw quaternion
    angular_vel: vec4<f32>,   // xyz angular velocity, w unused
    mass_data: vec4<f32>,     // mass, inv_mass, padding, padding
    shape_data: vec4<u32>,    // shape_type, flags, padding, padding
    shape_params: vec4<f32>,  // shape-specific parameters
}

@group(0) @binding(0) var<uniform> view_projection: ViewProjection;
@group(0) @binding(1) var<storage, read> bodies: array<Body>;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    // Full-screen triangle
    var pos: vec2<f32>;
    if (vertex_index == 0u) {
        pos = vec2<f32>(-1.0, -1.0);
    } else if (vertex_index == 1u) {
        pos = vec2<f32>(3.0, -1.0);
    } else {
        pos = vec2<f32>(-1.0, 3.0);
    }
    return vec4<f32>(pos.x, pos.y, 0.0, 1.0);
}

// SDF for sphere
fn sdSphere(p: vec3<f32>, center: vec3<f32>, radius: f32) -> f32 {
    return length(p - center) - radius;
}

// SDF for box
fn sdBox(p: vec3<f32>, center: vec3<f32>, half_extents: vec3<f32>) -> f32 {
    let d = abs(p - center) - half_extents;
    let outside_dist = length(max(d, vec3<f32>(0.0)));
    let inside_dist = min(max(d.x, max(d.y, d.z)), 0.0);
    return outside_dist + inside_dist;
}

// SDF for capsule (vertical capsule)
fn sdCapsule(p: vec3<f32>, center: vec3<f32>, half_height: f32, radius: f32) -> f32 {
    let pa = p - (center - vec3<f32>(0.0, half_height, 0.0));
    let ba = vec3<f32>(0.0, 2.0 * half_height, 0.0);
    let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - radius;
}

// Apply rotation quaternion to a vector
fn rotate_vector(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
    let qxyz = q.xyz;
    let qw = q.w;
    let t = 2.0 * cross(qxyz, v);
    return v + qw * t + cross(qxyz, t);
}

// Map the entire scene
fn map_scene(p: vec3<f32>) -> f32 {
    var min_dist = 1000000.0;
    
    // Loop through all bodies
    let num_bodies = arrayLength(&bodies);
    for (var i: u32 = 0u; i < num_bodies; i = i + 1u) {
        let body = bodies[i];
        let pos = body.position.xyz;
        let shape_type = body.shape_data.x;
        
        // Transform point to body's local space
        let local_p = rotate_vector(p - pos, vec4<f32>(-body.orientation.x, -body.orientation.y, -body.orientation.z, body.orientation.w));
        
        var dist = 1000000.0;
        
        // Calculate SDF based on shape type
        if (shape_type == 0u) { // Sphere
            // Sphere is rotation-invariant, so we can use world space directly
            dist = sdSphere(p, pos, body.shape_params.x);
        } else if (shape_type == 2u) { // Box
            let half_extents = body.shape_params.xyz;
            // Box SDF is calculated in local space
            dist = sdBox(local_p, vec3<f32>(0.0), half_extents);
        } else if (shape_type == 3u) { // Capsule
            let half_height = body.shape_params.x;
            let radius = body.shape_params.y;
            // Capsule SDF is calculated in local space
            dist = sdCapsule(local_p, vec3<f32>(0.0), half_height, radius);
        }
        
        min_dist = min(min_dist, dist);
    }
    
    return min_dist;
}

// Calculate normal using finite differences
fn calculate_normal(p: vec3<f32>) -> vec3<f32> {
    let eps = 0.001;
    let dx = vec3<f32>(eps, 0.0, 0.0);
    let dy = vec3<f32>(0.0, eps, 0.0);
    let dz = vec3<f32>(0.0, 0.0, eps);
    
    return normalize(vec3<f32>(
        map_scene(p + dx) - map_scene(p - dx),
        map_scene(p + dy) - map_scene(p - dy),
        map_scene(p + dz) - map_scene(p - dz)
    ));
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    // Convert fragment coordinates to UV coordinates
    let resolution = vec2<f32>(800.0, 600.0);
    // Flip Y coordinate to match world space (Y-up)
    let uv = vec2<f32>(
        (frag_coord.x - 0.5 * resolution.x) / resolution.y,
        -(frag_coord.y - 0.5 * resolution.y) / resolution.y
    );
    
    // Hardcoded camera for headless mode
    let cam_pos = vec3<f32>(10.0, 10.0, 10.0);
    let cam_target = vec3<f32>(0.0, 0.0, 0.0);
    
    // Calculate camera basis vectors
    let forward = normalize(cam_target - cam_pos);
    let world_up = vec3<f32>(0.0, 1.0, 0.0);
    let right = normalize(cross(world_up, forward));
    let up = normalize(cross(forward, right));
    
    // Generate ray
    let ray_origin = cam_pos;
    let ray_dir = normalize(forward + uv.x * right + uv.y * up);
    
    // Raymarch
    var t = 0.1;
    var hit = false;
    var hit_pos = vec3<f32>(0.0);
    
    for (var i = 0; i < 64; i = i + 1) {
        let p = ray_origin + ray_dir * t;
        let d = map_scene(p);
        
        if (d < 0.001) {
            hit = true;
            hit_pos = p;
            break;
        }
        
        t = t + d;
        if (t > 100.0) {
            break;
        }
    }
    
    // Color based on hit
    if (hit) {
        let normal = calculate_normal(hit_pos);
        let color = abs(normal); // Use absolute value of normal as color
        return vec4<f32>(color, 1.0);
    } else {
        return vec4<f32>(0.1, 0.2, 0.3, 1.0); // Background color
    }
}