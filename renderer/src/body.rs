use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Body {
    // Exactly 27 floats = 108 bytes (no alignment constraints)
    pub data: [f32; 27],
}

impl Body {
    // Data layout: [pos(3), vel(3), orient(4), angvel(3), force(3), torque(3), mass_props(4), shape_props(4)]
    // Total: 3+3+4+3+3+3+4+4 = 27 floats
    
    pub fn new_sphere(position: [f32; 3], radius: f32, mass: f32) -> Self {
        let mut data = [0.0f32; 27];
        
        // Position (0-2)
        data[0..3].copy_from_slice(&position);
        // Velocity (3-5) - already zeroed
        // Orientation (6-9) - identity quaternion
        data[6] = 1.0;
        // Angular velocity (10-12) - already zeroed
        // Force (13-15) - already zeroed
        // Torque (16-18) - already zeroed
        // Mass props (19-22): mass, inv_mass, inertia, inv_inertia
        data[19] = mass;
        data[20] = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        data[21] = 1.0;
        data[22] = 1.0;
        // Shape props (23-26): type, flags, param1, param2
        data[23] = 0.0; // sphere type
        data[24] = 0.0; // dynamic
        data[25] = radius;
        data[26] = 0.0;
        
        Self { data }
    }
    
    pub fn new_static_sphere(position: [f32; 3], radius: f32) -> Self {
        let mut data = [0.0f32; 27];
        
        data[0..3].copy_from_slice(&position);
        data[6] = 1.0; // identity quaternion w
        data[23] = 0.0; // sphere type
        data[24] = 1.0; // static
        data[25] = radius;
        
        Self { data }
    }
    
    pub fn new_box(position: [f32; 3], half_extents: [f32; 3], mass: f32) -> Self {
        let mut data = [0.0f32; 27];
        
        data[0..3].copy_from_slice(&position);
        data[6] = 1.0; // identity quaternion w
        data[19] = mass;
        data[20] = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        data[21] = 1.0;
        data[22] = 1.0;
        data[23] = 2.0; // box type
        data[24] = 0.0; // dynamic
        data[25] = half_extents[0];
        data[26] = half_extents[1];
        
        Self { data }
    }
    
    pub fn new_static_box(position: [f32; 3], half_extents: [f32; 3]) -> Self {
        let mut data = [0.0f32; 27];
        
        data[0..3].copy_from_slice(&position);
        data[6] = 1.0; // identity quaternion w
        data[23] = 2.0; // box type
        data[24] = 1.0; // static
        data[25] = half_extents[0];
        data[26] = half_extents[1];
        
        Self { data }
    }
    
    // Accessor methods
    pub fn position(&self) -> &[f32] {
        &self.data[0..3]
    }
    
    pub fn velocity(&self) -> &[f32] {
        &self.data[3..6]
    }
    
    pub fn orientation(&self) -> &[f32] {
        &self.data[6..10]
    }
    
    pub fn angular_vel(&self) -> &[f32] {
        &self.data[10..13]
    }
    
    pub fn mass_props(&self) -> &[f32] {
        &self.data[19..23]
    }
    
    pub fn shape_props(&self) -> &[f32] {
        &self.data[23..27]
    }
    
    // Helper methods to access shape data as expected by legacy code
    pub fn shape_type(&self) -> u32 {
        self.data[23] as u32
    }
    
    pub fn is_static(&self) -> bool {
        self.data[24] != 0.0
    }
    
    pub fn radius(&self) -> f32 {
        if self.shape_type() == 0 { // sphere
            self.data[25]
        } else {
            0.0
        }
    }
    
    pub fn half_extents(&self) -> [f32; 3] {
        if self.shape_type() == 2 { // box
            [self.data[25], self.data[26], self.data[25]] // assuming cubic for simplicity
        } else {
            [0.0; 3]
        }
    }
}

// Size check is done at runtime for now

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_body_size() {
        assert_eq!(std::mem::size_of::<Body>(), 108);
        assert_eq!(std::mem::align_of::<Body>(), 4);
    }

    #[test]
    fn test_body_creation() {
        let body = Body::new_sphere([1.0, 2.0, 3.0], 0.5, 1.0);
        assert_eq!(body.position()[0], 1.0);
        assert_eq!(body.position()[1], 2.0);
        assert_eq!(body.position()[2], 3.0);
        assert_eq!(body.radius(), 0.5); // radius
        assert_eq!(body.mass_props()[0], 1.0); // mass
        assert_eq!(body.shape_type(), 0); // sphere type
        assert!(!body.is_static()); // dynamic
    }

    #[test]
    fn test_static_body_creation() {
        let body = Body::new_static_sphere([0.0, 0.0, 0.0], 1.0);
        assert!(body.is_static()); // static flag
        assert_eq!(body.mass_props()[0], 0.0); // no mass for static
    }

    #[test]
    fn test_box_creation() {
        let body = Body::new_static_box([0.0, -1.0, 0.0], [10.0, 1.0, 10.0]);
        assert_eq!(body.shape_type(), 2); // box type
        assert!(body.is_static()); // static flag
        let half_extents = body.half_extents();
        assert_eq!(half_extents[0], 10.0); // half extent x
        assert_eq!(half_extents[1], 1.0); // half extent y
    }
}