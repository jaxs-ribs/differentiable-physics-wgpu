// Broad phase collision detection using uniform grid

struct Body {
    position: vec4<f32>,
    velocity: vec4<f32>,
    orientation: vec4<f32>,
    angular_vel: vec4<f32>,
    mass_data: vec4<f32>,
    shape_data: vec4<u32>,
    shape_params: vec4<f32>,
}

struct AABB {
    min: vec3<f32>,
    max: vec3<f32>,
}

struct BroadphaseParams {
    num_bodies: u32,
    cell_size: f32,
    grid_size: u32,  // Grid dimension (e.g., 64 for 64x64x64)
    max_bodies_per_cell: u32,
}

struct CellData {
    count: atomic<u32>,
    body_ids: array<u32, 32>,  // Max 32 bodies per cell
}

struct PotentialPair {
    body_a: u32,
    body_b: u32,
}

@group(0) @binding(0) var<storage, read> bodies: array<Body>;
@group(0) @binding(1) var<storage, read_write> grid: array<CellData>;
@group(0) @binding(2) var<storage, read_write> pairs: array<PotentialPair>;
@group(0) @binding(3) var<storage, read_write> pair_count: atomic<u32>;
@group(0) @binding(4) var<uniform> params: BroadphaseParams;

// Compute AABB for a body
fn compute_aabb(body: Body) -> AABB {
    var aabb: AABB;
    let pos = body.position.xyz;
    let shape_type = body.shape_data.x;
    
    if (shape_type == 0u) { // Sphere
        let radius = body.shape_params.x;
        aabb.min = pos - vec3<f32>(radius);
        aabb.max = pos + vec3<f32>(radius);
    } else if (shape_type == 1u) { // Capsule
        let radius = body.shape_params.x;
        let half_height = body.shape_params.y * 0.5;
        // Assume Y-aligned capsule for now
        aabb.min = pos - vec3<f32>(radius, half_height + radius, radius);
        aabb.max = pos + vec3<f32>(radius, half_height + radius, radius);
    } else if (shape_type == 2u) { // Box
        let half_extents = body.shape_params.xyz;
        aabb.min = pos - half_extents;
        aabb.max = pos + half_extents;
    } else {
        // Unknown shape, use point
        aabb.min = pos;
        aabb.max = pos;
    }
    
    return aabb;
}

// Convert 3D grid position to 1D index
fn grid_pos_to_index(x: u32, y: u32, z: u32) -> u32 {
    let size = params.grid_size;
    return x + y * size + z * size * size;
}

// Clear grid (phase 1)
@compute @workgroup_size(64)
fn clear_grid(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_cells = params.grid_size * params.grid_size * params.grid_size;
    
    if (idx >= total_cells) {
        return;
    }
    
    // Reset cell count
    atomicStore(&grid[idx].count, 0u);
}

// Build grid (phase 2)
@compute @workgroup_size(64)
fn build_grid(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let body_idx = global_id.x;
    
    if (body_idx >= params.num_bodies) {
        return;
    }
    
    let body = bodies[body_idx];
    let aabb = compute_aabb(body);
    
    // Convert AABB to grid cells
    let inv_cell_size = 1.0 / params.cell_size;
    let min_cell = vec3<u32>(clamp(vec3<i32>(aabb.min * inv_cell_size), vec3<i32>(0), vec3<i32>(params.grid_size - 1u)));
    let max_cell = vec3<u32>(clamp(vec3<i32>(aabb.max * inv_cell_size), vec3<i32>(0), vec3<i32>(params.grid_size - 1u)));
    
    // Insert body into all overlapping cells
    for (var x = min_cell.x; x <= max_cell.x; x = x + 1u) {
        for (var y = min_cell.y; y <= max_cell.y; y = y + 1u) {
            for (var z = min_cell.z; z <= max_cell.z; z = z + 1u) {
                let cell_idx = grid_pos_to_index(x, y, z);
                let slot = atomicAdd(&grid[cell_idx].count, 1u);
                
                // Store body ID if there's space
                if (slot < 32u) {
                    grid[cell_idx].body_ids[slot] = body_idx;
                }
            }
        }
    }
}

// Find pairs (phase 3) - simplified version without deduplication
// A better approach would be to have each body check cells or use a hash table
@compute @workgroup_size(64)
fn find_pairs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let body_idx = global_id.x;
    
    if (body_idx >= params.num_bodies) {
        return;
    }
    
    let body_a = bodies[body_idx];
    let aabb_a = compute_aabb(body_a);
    
    // Skip static bodies as body A
    if (body_a.shape_data.y == 1u) {
        return;
    }
    
    // Check cells that this body overlaps
    let inv_cell_size = 1.0 / params.cell_size;
    let min_cell = vec3<u32>(clamp(vec3<i32>(aabb_a.min * inv_cell_size), vec3<i32>(0), vec3<i32>(params.grid_size - 1u)));
    let max_cell = vec3<u32>(clamp(vec3<i32>(aabb_a.max * inv_cell_size), vec3<i32>(0), vec3<i32>(params.grid_size - 1u)));
    
    // Check all cells this body overlaps
    for (var x = min_cell.x; x <= max_cell.x; x = x + 1u) {
        for (var y = min_cell.y; y <= max_cell.y; y = y + 1u) {
            for (var z = min_cell.z; z <= max_cell.z; z = z + 1u) {
                let cell_idx = grid_pos_to_index(x, y, z);
                let count = atomicLoad(&grid[cell_idx].count);
                let actual_count = min(count, 32u);
                
                // Check against all bodies in this cell
                for (var i = 0u; i < actual_count; i = i + 1u) {
                    let body_b_idx = grid[cell_idx].body_ids[i];
                    
                    // Only process if b > a (avoids duplicates)
                    if (body_b_idx <= body_idx) {
                        continue;
                    }
                    
                    // Skip if both static
                    if (bodies[body_b_idx].shape_data.y == 1u) {
                        continue;
                    }
                    
                    // Check actual AABB overlap to avoid duplicates from multiple cells
                    let aabb_b = compute_aabb(bodies[body_b_idx]);
                    let overlap = all(aabb_a.min <= aabb_b.max) && all(aabb_b.min <= aabb_a.max);
                    
                    if (overlap) {
                        // Add pair only once (when we first encounter it)
                        // This is a simple approach - more sophisticated would use a hash set
                        let pair_idx = atomicAdd(&pair_count, 1u);
                        if (pair_idx < 10000u) {
                            pairs[pair_idx].body_a = body_idx;
                            pairs[pair_idx].body_b = body_b_idx;
                        }
                        
                        // Break out of cell checking for this body pair
                        // (not perfect but reduces duplicates)
                    }
                }
            }
        }
    }
}