#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ViewProjectionUniform {
    pub view_proj: [[f32; 4]; 4],
}

impl ViewProjectionUniform {
    pub fn new(view_proj: [[f32; 4]; 4]) -> Self {
        Self { view_proj }
    }
}