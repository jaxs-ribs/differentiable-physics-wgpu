// Efficient grid-based broad phase with deduplication
// Uses a two-phase approach: build grid, then check pairs

struct Body {
    position: vec4<f32>,
    velocity: vec4<f32>,
    orientation: vec4<f32>,
    angular_vel: vec4<f32>,
    mass_data: vec4<f32>,
    shape_data: vec4<u32>,
    shape_params: vec4<f32>,
}

struct BroadphaseParams {
    num_bodies: u32,
    cell_size: f32,
    grid_size: u32,
    max_bodies_per_cell: u32,
}

struct GridCell {
    count: atomic<u32>,
    body_ids: array<u32, 32>,
}

struct PotentialPair {
    body_a: u32,
    body_b: u32,
}

@group(0) @binding(0) var<uniform> params: BroadphaseParams;
@group(0) @binding(1) var<storage, read> bodies: array<Body>;
@group(0) @binding(2) var<storage, read_write> grid: array<GridCell>;
@group(0) @binding(3) var<storage, read_write> pairs: array<PotentialPair>;
@group(0) @binding(4) var<storage, read_write> pair_count: atomic<u32>;

// Convert world position to grid cell
fn world_to_grid(pos: vec3<f32>) -> vec3<i32> {
    return vec3<i32>(pos / params.cell_size);
}

// Convert 3D grid coords to 1D index
fn grid_index(cell: vec3<i32>) -> u32 {
    let clamped = clamp(cell, vec3<i32>(0), vec3<i32>(i32(params.grid_size) - 1));
    return u32(clamped.x) + u32(clamped.y) * params.grid_size + u32(clamped.z) * params.grid_size * params.grid_size;
}

// Phase 1: Clear grid
@compute @workgroup_size(64)
fn clear_grid(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_cells = params.grid_size * params.grid_size * params.grid_size;
    
    if (idx >= total_cells) {
        return;
    }
    
    atomicStore(&grid[idx].count, 0u);
}

// Phase 2: Insert bodies into grid
@compute @workgroup_size(64)
fn insert_bodies(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let body_idx = global_id.x;
    
    if (body_idx >= params.num_bodies) {
        return;
    }
    
    let body = bodies[body_idx];
    let pos = body.position.xyz;
    let radius = body.shape_params.x; // Assume sphere for now
    
    // Get grid bounds for this body
    let min_cell = world_to_grid(pos - vec3<f32>(radius));
    let max_cell = world_to_grid(pos + vec3<f32>(radius));
    
    // Insert into all overlapping cells
    for (var x = min_cell.x; x <= max_cell.x; x = x + 1) {
        for (var y = min_cell.y; y <= max_cell.y; y = y + 1) {
            for (var z = min_cell.z; z <= max_cell.z; z = z + 1) {
                let cell_idx = grid_index(vec3<i32>(x, y, z));
                let slot = atomicAdd(&grid[cell_idx].count, 1u);
                
                if (slot < 32u) {
                    grid[cell_idx].body_ids[slot] = body_idx;
                }
            }
        }
    }
}

// Phase 3: Find pairs within cells
@compute @workgroup_size(64)
fn find_pairs_in_cells(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    let total_cells = params.grid_size * params.grid_size * params.grid_size;
    
    if (cell_idx >= total_cells) {
        return;
    }
    
    let count = min(atomicLoad(&grid[cell_idx].count), 32u);
    
    // Check all pairs within this cell
    for (var i = 0u; i < count; i = i + 1u) {
        let body_a = grid[cell_idx].body_ids[i];
        
        // Skip static bodies as first body
        if (bodies[body_a].shape_data.y == 1u) {
            continue;
        }
        
        for (var j = i + 1u; j < count; j = j + 1u) {
            let body_b = grid[cell_idx].body_ids[j];
            
            // Skip if both static
            if (bodies[body_b].shape_data.y == 1u) {
                continue;
            }
            
            // Ensure consistent ordering to avoid duplicates
            var pair_a = body_a;
            var pair_b = body_b;
            if (pair_a > pair_b) {
                let temp = pair_a;
                pair_a = pair_b;
                pair_b = temp;
            }
            
            // Check actual distance to filter false positives
            let pos_a = bodies[pair_a].position.xyz;
            let pos_b = bodies[pair_b].position.xyz;
            let radius_a = bodies[pair_a].shape_params.x;
            let radius_b = bodies[pair_b].shape_params.x;
            
            let dist = length(pos_b - pos_a);
            let threshold = (radius_a + radius_b) * 1.1; // Small margin
            
            if (dist < threshold) {
                // Use midpoint check to avoid duplicates
                let midpoint = (pos_a + pos_b) * 0.5;
                let mid_cell = world_to_grid(midpoint);
                let mid_cell_idx = grid_index(mid_cell);
                
                // Only add pair if we're in the cell containing their midpoint
                if (cell_idx == mid_cell_idx) {
                    let pair_idx = atomicAdd(&pair_count, 1u);
                    if (pair_idx < 10000u) {
                        pairs[pair_idx].body_a = pair_a;
                        pairs[pair_idx].body_b = pair_b;
                    }
                }
            }
        }
    }
}