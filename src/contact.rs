use bytemuck::{Pod, Zeroable};

#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Contact {
    pub body_a: u32,
    pub body_b: u32,
    pub distance: f32,
    pub _padding1: f32,
    pub normal: [f32; 4],  // vec4 for alignment
    pub point: [f32; 4],   // vec4 for alignment
}

impl Contact {
    pub const MAX_CONTACTS: usize = 1000;
}