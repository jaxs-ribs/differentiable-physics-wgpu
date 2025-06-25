use winit::{event_loop::EventLoop, window::Window};

const DEFAULT_WINDOW_WIDTH: u32 = 1024;
const DEFAULT_WINDOW_HEIGHT: u32 = 768;
const WINDOW_TITLE: &str = "Physics Engine Visualization";

pub struct WindowManager {
    window: Window,
}

impl WindowManager {
    pub fn new(event_loop: &EventLoop<()>) -> Result<Self, Box<dyn std::error::Error>> {
        let window_attributes = Window::default_attributes()
            .with_title(WINDOW_TITLE)
            .with_inner_size(winit::dpi::LogicalSize::new(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT));
        
        let window = event_loop.create_window(window_attributes)?;
        
        Ok(Self { window })
    }
    
    pub fn window(&self) -> &Window {
        &self.window
    }
    
    pub fn inner_size(&self) -> winit::dpi::PhysicalSize<u32> {
        self.window.inner_size()
    }
}