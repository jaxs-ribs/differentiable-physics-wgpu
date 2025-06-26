use glam::Mat4;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ViewProjectionUniform {
    view_projection: [[f32; 4]; 4],
}

impl ViewProjectionUniform {
    pub fn new(view_projection_matrix: [[f32; 4]; 4]) -> Self {
        Self { view_projection: view_projection_matrix }
    }
}