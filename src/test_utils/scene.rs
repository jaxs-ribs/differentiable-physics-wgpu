use crate::body::Body;

pub struct SceneConfig {
    pub grid_size: usize,
    pub spacing: f32,
    pub initial_height: f32,
    pub sphere_radius: f32,
    pub mass: f32,
    pub ground_size: [f32; 3],
    pub add_small_velocity: bool,
}

impl Default for SceneConfig {
    fn default() -> Self {
        Self {
            grid_size: 3,
            spacing: 2.0,
            initial_height: 10.0,
            sphere_radius: 0.5,
            mass: 1.0,
            ground_size: [20.0, 1.0, 20.0],
            add_small_velocity: true,
        }
    }
}

pub struct SceneBuilder;

impl SceneBuilder {
    pub fn create_sphere_grid(config: SceneConfig) -> Vec<Body> {
        let mut bodies = Vec::new();
        
        let offset = (config.grid_size as f32 - 1.0) * config.spacing / 2.0;
        
        for i in 0..config.grid_size {
            for j in 0..config.grid_size {
                let x = i as f32 * config.spacing - offset;
                let z = j as f32 * config.spacing - offset;
                
                let mut sphere = Body::new_sphere(
                    [x, config.initial_height, z],
                    config.sphere_radius,
                    config.mass,
                );
                
                if config.add_small_velocity {
                    // Add small deterministic velocity based on position
                    sphere.velocity[0] = (i as f32 - config.grid_size as f32 / 2.0) * 0.1;
                    sphere.velocity[2] = (j as f32 - config.grid_size as f32 / 2.0) * 0.1;
                }
                
                bodies.push(sphere);
            }
        }
        
        bodies.push(Self::create_ground_plane(config.ground_size));
        bodies
    }
    
    pub fn create_falling_spheres(count: usize, height_range: (f32, f32)) -> Vec<Body> {
        let mut bodies = Vec::new();
        
        let grid_size = (count as f32).sqrt().ceil() as usize;
        let spacing = 10.0 / grid_size as f32;
        
        for i in 0..count {
            let row = i / grid_size;
            let col = i % grid_size;
            
            let x = (col as f32 - grid_size as f32 / 2.0) * spacing;
            let y = height_range.0 + (row as f32 / grid_size as f32) * (height_range.1 - height_range.0);
            let z = (row as f32 - grid_size as f32 / 2.0) * spacing;
            
            bodies.push(Body::new_sphere([x, y, z], 0.5, 1.0));
        }
        
        bodies.push(Self::create_ground_plane([20.0, 1.0, 20.0]));
        bodies
    }
    
    pub fn create_ground_plane(size: [f32; 3]) -> Body {
        Body::new_static_box([0.0, -size[1], 0.0], size)
    }
    
    pub fn create_test_scene() -> Vec<Body> {
        Self::create_sphere_grid(SceneConfig::default())
    }
    
    pub fn create_benchmark_scene(sphere_count: usize) -> Vec<Body> {
        let config = SceneConfig {
            grid_size: (sphere_count as f32).sqrt() as usize,
            ..Default::default()
        };
        Self::create_sphere_grid(config)
    }
}