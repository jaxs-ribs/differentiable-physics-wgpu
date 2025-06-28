use wgpu::{Buffer, BufferUsages, util::DeviceExt};
use anyhow::Result;

pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    pub async fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| anyhow::anyhow!("Failed to find suitable adapter"))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Primary Device"),
                    required_features: wgpu::Features::default(),
                    required_limits: wgpu::Limits::default(),
                },
                None, // Trace path
            )
            .await?;

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }
    
    pub fn create_buffer_init(&self, label: &str, data: &[u8], usage: BufferUsages) -> Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: data,
            usage,
        })
    }
}