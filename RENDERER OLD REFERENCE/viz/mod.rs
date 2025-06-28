mod camera;
mod renderer;
mod dual_renderer;
mod window;
mod wireframe_geometry;
mod shader_manager;
mod uniforms;

pub use self::window::WindowManager;
pub use self::renderer::Renderer;
pub use self::dual_renderer::DualRenderer;

use crate::{body::Body, gpu::GpuContext};
use winit::event_loop::EventLoop;

pub struct Visualizer {
    window_manager: WindowManager,
    renderer: Renderer,
}

impl Visualizer {
    pub async fn new(event_loop: &EventLoop<()>, gpu: &GpuContext) -> Result<Self, Box<dyn std::error::Error>> {
        let window_manager = WindowManager::new(event_loop)?;
        let renderer = Renderer::new(&window_manager, gpu).await?;
        
        Ok(Self {
            window_manager,
            renderer,
        })
    }
    
    pub fn update_bodies(&self, gpu: &GpuContext, bodies: &[Body]) {
        self.renderer.update_bodies(gpu, bodies);
    }
    
    pub fn render(&self, gpu: &GpuContext, vertex_count: u32) -> Result<(), wgpu::SurfaceError> {
        self.renderer.render(gpu, vertex_count)
    }
    
    pub fn resize(&mut self, gpu: &GpuContext, new_size: winit::dpi::PhysicalSize<u32>) {
        self.renderer.resize(gpu, new_size);
    }
    
    pub fn window(&self) -> &winit::window::Window {
        self.window_manager.window()
    }
}