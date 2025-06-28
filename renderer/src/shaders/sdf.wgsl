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
    let pos = get_fullscreen_triangle_position(vertex_index);
    return vec4<f32>(pos, 0.0, 1.0);
}

fn get_fullscreen_triangle_position(vertex_index: u32) -> vec2<f32> {
    switch vertex_index {
        case 0u: { return vec2<f32>(-1.0, -1.0); }
        case 1u: { return vec2<f32>(3.0, -1.0); }
        default: { return vec2<f32>(-1.0, 3.0); }
    }
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

const SHAPE_SPHERE: u32 = 0u;
const SHAPE_BOX: u32 = 2u;
const SHAPE_CAPSULE: u32 = 3u;
const MAX_DISTANCE: f32 = 1000000.0;

fn map_scene(p: vec3<f32>) -> f32 {
    var min_dist = MAX_DISTANCE;
    let num_bodies = arrayLength(&bodies);
    
    for (var i: u32 = 0u; i < num_bodies; i = i + 1u) {
        let dist = evaluate_body_sdf(p, bodies[i]);
        min_dist = min(min_dist, dist);
    }
    
    return min_dist;
}

fn evaluate_body_sdf(p: vec3<f32>, body: Body) -> f32 {
    let shape_type = body.shape_data.x;
    
    if (shape_type == SHAPE_SPHERE) {
        return evaluate_sphere(p, body);
    } else if (shape_type == SHAPE_BOX) {
        return evaluate_box(p, body);
    } else if (shape_type == SHAPE_CAPSULE) {
        return evaluate_capsule(p, body);
    }
    
    return MAX_DISTANCE;
}

fn evaluate_sphere(p: vec3<f32>, body: Body) -> f32 {
    return sdSphere(p, body.position.xyz, body.shape_params.x);
}

fn evaluate_box(p: vec3<f32>, body: Body) -> f32 {
    let local_p = world_to_local(p, body);
    return sdBox(local_p, vec3<f32>(0.0), body.shape_params.xyz);
}

fn evaluate_capsule(p: vec3<f32>, body: Body) -> f32 {
    let local_p = world_to_local(p, body);
    return sdCapsule(local_p, vec3<f32>(0.0), body.shape_params.x, body.shape_params.y);
}

fn world_to_local(p: vec3<f32>, body: Body) -> vec3<f32> {
    let translated = p - body.position.xyz;
    let inverse_rotation = conjugate_quaternion(body.orientation);
    return rotate_vector(translated, inverse_rotation);
}

fn conjugate_quaternion(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.x, -q.y, -q.z, q.w);
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

const RESOLUTION: vec2<f32> = vec2<f32>(800.0, 600.0);
const RAYMARCH_MAX_STEPS: i32 = 64;
const RAYMARCH_MIN_DIST: f32 = 0.001;
const RAYMARCH_MAX_DIST: f32 = 100.0;
const RAYMARCH_START_DIST: f32 = 0.1;
const BACKGROUND_COLOR: vec3<f32> = vec3<f32>(0.1, 0.2, 0.3);

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = screen_to_uv(frag_coord.xy);
    let ray = generate_camera_ray(uv);
    let hit_info = raymarch(ray.origin, ray.direction);
    
    return shade_pixel(hit_info);
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

struct HitInfo {
    hit: bool,
    position: vec3<f32>,
}

fn screen_to_uv(screen_pos: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        (screen_pos.x - 0.5 * RESOLUTION.x) / RESOLUTION.y,
        -(screen_pos.y - 0.5 * RESOLUTION.y) / RESOLUTION.y
    );
}

fn generate_camera_ray(uv: vec2<f32>) -> Ray {
    let cam_pos = vec3<f32>(10.0, 10.0, 10.0);
    let cam_target = vec3<f32>(0.0, 0.0, 0.0);
    let camera_matrix = build_camera_matrix(cam_pos, cam_target);
    
    return Ray(
        cam_pos,
        normalize(camera_matrix.forward + uv.x * camera_matrix.right + uv.y * camera_matrix.up)
    );
}

struct CameraMatrix {
    forward: vec3<f32>,
    right: vec3<f32>,
    up: vec3<f32>,
}

fn build_camera_matrix(position: vec3<f32>, target: vec3<f32>) -> CameraMatrix {
    let forward = normalize(target - position);
    let world_up = vec3<f32>(0.0, 1.0, 0.0);
    let right = normalize(cross(world_up, forward));
    let up = normalize(cross(forward, right));
    
    return CameraMatrix(forward, right, up);
}

fn raymarch(origin: vec3<f32>, direction: vec3<f32>) -> HitInfo {
    var t = RAYMARCH_START_DIST;
    
    for (var i = 0; i < RAYMARCH_MAX_STEPS; i = i + 1) {
        let p = origin + direction * t;
        let d = map_scene(p);
        
        if (d < RAYMARCH_MIN_DIST) {
            return HitInfo(true, p);
        }
        
        t = t + d;
        if (t > RAYMARCH_MAX_DIST) {
            break;
        }
    }
    
    return HitInfo(false, vec3<f32>(0.0));
}

fn shade_pixel(hit_info: HitInfo) -> vec4<f32> {
    if (hit_info.hit) {
        let normal = calculate_normal(hit_info.position);
        let color = abs(normal);
        return vec4<f32>(color, 1.0);
    } else {
        return vec4<f32>(BACKGROUND_COLOR, 1.0);
    }
}