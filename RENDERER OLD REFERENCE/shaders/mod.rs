use std::collections::HashMap;

pub struct ShaderBuilder {
    common_types: String,
    shader_cache: HashMap<String, String>,
}

impl ShaderBuilder {
    pub fn new() -> Self {
        let common_types = include_str!("common/types.wgsl").to_string();
        
        Self {
            common_types,
            shader_cache: HashMap::new(),
        }
    }
    
    pub fn get_shader(&mut self, name: &str) -> &str {
        if !self.shader_cache.contains_key(name) {
            let shader_source = match name {
                "physics_step" => self.build_physics_step_shader(),
                "physics_step_fixed" => include_str!("physics_step_fixed.wgsl").to_string(),
                "integrator" => include_str!("integrator.wgsl").to_string(),
                "broadphase" => include_str!("broadphase.wgsl").to_string(),
                "broadphase_simple" => include_str!("broadphase_simple.wgsl").to_string(),
                "broadphase_grid" => include_str!("broadphase_grid.wgsl").to_string(),
                "contact_solver" => include_str!("contact_solver.wgsl").to_string(),
                "sdf" => include_str!("sdf.wgsl").to_string(),
                "physics_debug" => include_str!("physics_debug.wgsl").to_string(),
                _ => panic!("Unknown shader: {}", name),
            };
            
            self.shader_cache.insert(name.to_string(), shader_source);
        }
        
        self.shader_cache.get(name).unwrap()
    }
    
    fn build_shader_with_common(&self, specific_source: &str) -> String {
        format!("{}\n\n{}", self.common_types, specific_source)
    }
    
    fn build_physics_step_shader(&self) -> String {
        let specific = include_str!("physics_step_impl.wgsl");
        self.build_shader_with_common(specific)
    }
}

impl Default for ShaderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// Shader loading helper for convenience
pub fn load_shader(device: &wgpu::Device, name: &str) -> wgpu::ShaderModule {
    let mut builder = ShaderBuilder::new();
    let source = builder.get_shader(name);
    
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(name),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}