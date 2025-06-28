use crate::body::Body;

const STATIC_BODY_COLOR: [f32; 3] = [0.5, 0.5, 0.5]; // Gray
const DYNAMIC_BODY_COLOR: [f32; 3] = [0.0, 1.0, 0.0]; // Green
const VERTICES_PER_LINE: usize = 2;
const FLOATS_PER_VERTEX: usize = 6; // 3 position + 3 color
const LINES_PER_AABB: usize = 12;
const LINE_THICKNESS: f32 = 0.05; // Thickness for triangle-based lines

pub struct WireframeGeometry;

// Debug mode to render triangles instead of lines
const DEBUG_TRIANGLES: bool = true;
const DEBUG_FULLSCREEN: bool = false; // Render actual bodies as triangles

impl WireframeGeometry {
    pub fn generate_vertices_from_bodies(bodies: &[Body]) -> Vec<f32> {
        let mut vertices = Vec::new();
        
        static mut FIRST_GENERATION: bool = true;
        
        if DEBUG_TRIANGLES {
            if DEBUG_FULLSCREEN {
                // Add a full-screen triangle in NDC space to bypass camera issues
                unsafe {
                    if FIRST_GENERATION {
                        println!("Adding fullscreen debug triangle");
                    }
                }
                
                // Full screen triangle in view space (will be transformed by camera)
                // Make it huge to ensure it covers the view
                vertices.extend_from_slice(&[
                    -100.0, -100.0, -50.0, 1.0, 0.0, 0.0,  // Bottom-left (red)
                    100.0, -100.0, -50.0, 0.0, 1.0, 0.0,   // Bottom-right (green)
                    0.0, 100.0, -50.0, 0.0, 0.0, 1.0,      // Top-center (blue)
                ]);
            } else {
                // First add a big visible triangle in the center to test rendering
                unsafe {
                    if FIRST_GENERATION {
                        println!("Adding debug triangle at origin");
                    }
                }
                
                // Big colorful triangle at origin
                vertices.extend_from_slice(&[
                    -10.0, -5.0, 0.0, 1.0, 0.0, 0.0,  // Left vertex (red)
                    10.0, -5.0, 0.0, 0.0, 1.0, 0.0,   // Right vertex (green)
                    0.0, 15.0, 0.0, 0.0, 0.0, 1.0,    // Top vertex (blue)
                ]);
            }
            
            // Add debug triangle for each body to ensure something is visible
            for (index, body) in bodies.iter().enumerate() {
                let pos = [body.position[0], body.position[1], body.position[2]];
                let color = Self::get_body_color(body);
                let size = 2.0;
                
                unsafe {
                    if FIRST_GENERATION && index == 0 {
                        println!("Debug triangle at position: {:?}, color: {:?}", pos, color);
                    }
                }
                
                // Add a simple triangle at body position
                vertices.extend_from_slice(&[
                    pos[0] - size, pos[1], pos[2], color[0], color[1], color[2],
                    pos[0] + size, pos[1], pos[2], color[0], color[1], color[2],
                    pos[0], pos[1] + size * 2.0, pos[2], color[0], color[1], color[2],
                ]);
            }
        } else {
            // Original AABB rendering
            for (index, body) in bodies.iter().enumerate() {
                let aabb = Self::calculate_aabb(body);
                if let Some((min, max)) = aabb {
                    let color = Self::get_body_color(body);
                    
                    unsafe {
                        if FIRST_GENERATION && index == 0 {
                            println!("First body AABB: min={:?}, max={:?}, color={:?}", min, max, color);
                            println!("Body position: {:?}", body.position);
                        }
                    }
                    
                    Self::add_aabb_lines(&mut vertices, &min, &max, &color);
                }
            }
        }
        
        unsafe {
            if FIRST_GENERATION && !vertices.is_empty() {
                FIRST_GENERATION = false;
                println!("Generated {} vertices, first vertex: {:?}", vertices.len(), &vertices[0..6]);
                // Check if vertices are reasonable
                let mut min_coords = [f32::MAX; 3];
                let mut max_coords = [f32::MIN; 3];
                for i in (0..vertices.len()).step_by(6) {
                    for j in 0..3 {
                        min_coords[j] = min_coords[j].min(vertices[i + j]);
                        max_coords[j] = max_coords[j].max(vertices[i + j]);
                    }
                }
                println!("Vertex bounds: min={:?}, max={:?}", min_coords, max_coords);
            }
        }
        
        vertices
    }
    
    fn calculate_aabb(body: &Body) -> Option<([f32; 3], [f32; 3])> {
        let position = [body.position[0], body.position[1], body.position[2]];
        
        match body.shape_data[0] {
            0 => Self::calculate_sphere_aabb(&position, body.shape_params[0]),
            2 => Self::calculate_box_aabb(&position, &body.shape_params),
            shape_type => {
                println!("Unknown shape type: {}", shape_type);
                None
            }
        }
    }
    
    fn calculate_sphere_aabb(position: &[f32; 3], radius: f32) -> Option<([f32; 3], [f32; 3])> {
        let min = [position[0] - radius, position[1] - radius, position[2] - radius];
        let max = [position[0] + radius, position[1] + radius, position[2] + radius];
        Some((min, max))
    }
    
    fn calculate_box_aabb(position: &[f32; 3], half_extents: &[f32]) -> Option<([f32; 3], [f32; 3])> {
        let min = [
            position[0] - half_extents[0],
            position[1] - half_extents[1],
            position[2] - half_extents[2],
        ];
        let max = [
            position[0] + half_extents[0],
            position[1] + half_extents[1],
            position[2] + half_extents[2],
        ];
        Some((min, max))
    }
    
    fn get_body_color(body: &Body) -> [f32; 3] {
        // Bodies with very large mass (>1000) are considered static
        if body.mass_data[0] > 1000.0 {
            STATIC_BODY_COLOR
        } else {
            DYNAMIC_BODY_COLOR
        }
    }
    
    fn add_aabb_lines(vertices: &mut Vec<f32>, min: &[f32; 3], max: &[f32; 3], color: &[f32; 3]) {
        let corners = Self::calculate_corners(min, max);
        
        // Bottom face
        Self::add_line(vertices, &corners[0], &corners[1], color);
        Self::add_line(vertices, &corners[1], &corners[2], color);
        Self::add_line(vertices, &corners[2], &corners[3], color);
        Self::add_line(vertices, &corners[3], &corners[0], color);
        
        // Top face
        Self::add_line(vertices, &corners[4], &corners[5], color);
        Self::add_line(vertices, &corners[5], &corners[6], color);
        Self::add_line(vertices, &corners[6], &corners[7], color);
        Self::add_line(vertices, &corners[7], &corners[4], color);
        
        // Vertical edges
        Self::add_line(vertices, &corners[0], &corners[4], color);
        Self::add_line(vertices, &corners[1], &corners[5], color);
        Self::add_line(vertices, &corners[2], &corners[6], color);
        Self::add_line(vertices, &corners[3], &corners[7], color);
    }
    
    fn calculate_corners(min: &[f32; 3], max: &[f32; 3]) -> [[f32; 3]; 8] {
        [
            [min[0], min[1], min[2]],
            [max[0], min[1], min[2]],
            [max[0], max[1], min[2]],
            [min[0], max[1], min[2]],
            [min[0], min[1], max[2]],
            [max[0], min[1], max[2]],
            [max[0], max[1], max[2]],
            [min[0], max[1], max[2]],
        ]
    }
    
    fn add_line(vertices: &mut Vec<f32>, start: &[f32; 3], end: &[f32; 3], color: &[f32; 3]) {
        // For now, just add simple lines. We'll convert to triangles if needed.
        vertices.extend_from_slice(&[start[0], start[1], start[2], color[0], color[1], color[2]]);
        vertices.extend_from_slice(&[end[0], end[1], end[2], color[0], color[1], color[2]]);
    }
    
    fn add_thick_line(vertices: &mut Vec<f32>, start: &[f32; 3], end: &[f32; 3], color: &[f32; 3], thickness: f32) {
        // Create a thick line using two triangles (6 vertices)
        let dir = [
            end[0] - start[0],
            end[1] - start[1],
            end[2] - start[2]
        ];
        
        // Find a perpendicular vector
        let perp = if dir[0].abs() < 0.9 {
            Self::normalize(&Self::cross(&dir, &[1.0, 0.0, 0.0]))
        } else {
            Self::normalize(&Self::cross(&dir, &[0.0, 1.0, 0.0]))
        };
        
        let half_thickness = thickness * 0.5;
        
        // Create 4 corners of the line quad
        let p1 = [start[0] - perp[0] * half_thickness, start[1] - perp[1] * half_thickness, start[2] - perp[2] * half_thickness];
        let p2 = [start[0] + perp[0] * half_thickness, start[1] + perp[1] * half_thickness, start[2] + perp[2] * half_thickness];
        let p3 = [end[0] - perp[0] * half_thickness, end[1] - perp[1] * half_thickness, end[2] - perp[2] * half_thickness];
        let p4 = [end[0] + perp[0] * half_thickness, end[1] + perp[1] * half_thickness, end[2] + perp[2] * half_thickness];
        
        // First triangle (p1, p2, p3)
        vertices.extend_from_slice(&[p1[0], p1[1], p1[2], color[0], color[1], color[2]]);
        vertices.extend_from_slice(&[p2[0], p2[1], p2[2], color[0], color[1], color[2]]);
        vertices.extend_from_slice(&[p3[0], p3[1], p3[2], color[0], color[1], color[2]]);
        
        // Second triangle (p2, p3, p4)
        vertices.extend_from_slice(&[p2[0], p2[1], p2[2], color[0], color[1], color[2]]);
        vertices.extend_from_slice(&[p3[0], p3[1], p3[2], color[0], color[1], color[2]]);
        vertices.extend_from_slice(&[p4[0], p4[1], p4[2], color[0], color[1], color[2]]);
    }
    
    fn normalize(v: &[f32; 3]) -> [f32; 3] {
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if len > 0.0 {
            [v[0] / len, v[1] / len, v[2] / len]
        } else {
            [0.0, 0.0, 0.0]
        }
    }
    
    fn cross(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
        ]
    }
    
}