use bytemuck::{Pod, Zeroable};

const DEFAULT_TIME_STEP: f32 = 0.016;
const DEFAULT_GRAVITY: [f32; 3] = [0.0, -9.81, 0.0];

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SimulationParameters {
    pub dt: f32,
    pub gravity_x: f32,
    pub gravity_y: f32,
    pub gravity_z: f32,
    pub num_bodies: u32,
    _padding: [f32; 3],
}

// Verify size for GPU alignment
const _: () = assert!(std::mem::size_of::<SimulationParameters>() == 32);

impl SimulationParameters {
    pub fn new(num_bodies: u32, time_step: f32, gravity: [f32; 3]) -> Self {
        Self {
            dt: time_step,
            gravity_x: gravity[0],
            gravity_y: gravity[1],
            gravity_z: gravity[2],
            num_bodies,
            _padding: [0.0; 3],
        }
    }
    
    pub fn default(num_bodies: u32) -> Self {
        Self::new(num_bodies, DEFAULT_TIME_STEP, DEFAULT_GRAVITY)
    }
    
    pub fn with_gravity(mut self, gravity: [f32; 3]) -> Self {
        self.gravity_x = gravity[0];
        self.gravity_y = gravity[1];
        self.gravity_z = gravity[2];
        self
    }
    
    pub fn with_time_step(mut self, dt: f32) -> Self {
        self.dt = dt;
        self
    }
}