// Simplified broad phase - each body checks its own cell only
// This avoids duplicates but may miss some edge cases

struct Body {
    position: vec4<f32>,
    velocity: vec4<f32>,
    orientation: vec4<f32>,
    angular_vel: vec4<f32>,
    mass_data: vec4<f32>,
    shape_data: vec4<u32>,
    shape_params: vec4<f32>,
}

struct PotentialPair {
    body_a: u32,
    body_b: u32,
}

@group(0) @binding(0) var<storage, read> bodies: array<Body>;
@group(0) @binding(1) var<storage, read_write> pairs: array<PotentialPair>;
@group(0) @binding(2) var<storage, read_write> pair_count: atomic<u32>;

// Simple O(n^2) broad phase for testing
@compute @workgroup_size(64)
fn find_all_pairs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx_a = global_id.x;
    let num_bodies = 8u; // Hardcoded for test
    
    if (idx_a >= num_bodies) {
        return;
    }
    
    let body_a = bodies[idx_a];
    
    // Skip static bodies as first body
    if (body_a.shape_data.y == 1u) {
        return;
    }
    
    // Simple sphere AABB
    let pos_a = body_a.position.xyz;
    let radius_a = body_a.shape_params.x;
    
    // Check against all subsequent bodies
    for (var idx_b = idx_a + 1u; idx_b < num_bodies; idx_b = idx_b + 1u) {
        let body_b = bodies[idx_b];
        
        // Skip if both static
        if (body_b.shape_data.y == 1u) {
            continue;
        }
        
        let pos_b = body_b.position.xyz;
        let radius_b = body_b.shape_params.x;
        
        // Simple distance check for spheres
        let dist = length(pos_b - pos_a);
        let threshold = (radius_a + radius_b) * 1.5; // Some margin
        
        if (dist < threshold) {
            let pair_idx = atomicAdd(&pair_count, 1u);
            if (pair_idx < 10000u) {
                pairs[pair_idx].body_a = idx_a;
                pairs[pair_idx].body_b = idx_b;
            }
        }
    }
}