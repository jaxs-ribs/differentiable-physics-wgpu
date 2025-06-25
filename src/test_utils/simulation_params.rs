use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct TestSimulationParams {
    pub dt: f32,
    pub gravity_x: f32,
    pub gravity_y: f32,
    pub gravity_z: f32,
    pub num_bodies: u32,
    pub _padding: [f32; 3],
}

impl TestSimulationParams {
    pub fn new(num_bodies: u32) -> Self {
        Self {
            dt: 0.016,
            gravity_x: 0.0,
            gravity_y: -9.81,
            gravity_z: 0.0,
            num_bodies,
            _padding: [0.0; 3],
        }
    }
    
    pub fn with_gravity(mut self, gravity: [f32; 3]) -> Self {
        self.gravity_x = gravity[0];
        self.gravity_y = gravity[1];
        self.gravity_z = gravity[2];
        self
    }
    
    pub fn with_timestep(mut self, dt: f32) -> Self {
        self.dt = dt;
        self
    }
}

impl Default for TestSimulationParams {
    fn default() -> Self {
        Self::new(0)
    }
}

// Verify struct size for GPU alignment
const _: () = assert!(std::mem::size_of::<TestSimulationParams>() == 32);