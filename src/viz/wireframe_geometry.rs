use crate::body::Body;

const STATIC_BODY_COLOR: [f32; 3] = [0.5, 0.5, 0.5]; // Gray
const DYNAMIC_BODY_COLOR: [f32; 3] = [0.0, 1.0, 0.0]; // Green
const VERTICES_PER_LINE: usize = 2;
const FLOATS_PER_VERTEX: usize = 6; // 3 position + 3 color
const LINES_PER_AABB: usize = 12;

pub struct WireframeGeometry;

impl WireframeGeometry {
    pub fn generate_vertices_from_bodies(bodies: &[Body]) -> Vec<f32> {
        let mut vertices = Vec::new();
        
        for (index, body) in bodies.iter().enumerate() {
            let aabb = Self::calculate_aabb(body);
            if let Some((min, max)) = aabb {
                let color = Self::get_body_color(body);
                Self::add_aabb_lines(&mut vertices, &min, &max, &color);
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
        vertices.extend_from_slice(&[start[0], start[1], start[2], color[0], color[1], color[2]]);
        vertices.extend_from_slice(&[end[0], end[1], end[2], color[0], color[1], color[2]]);
    }
    
}