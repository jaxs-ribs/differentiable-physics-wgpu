use crate::body::Body;

pub fn integrate_cpu(bodies: &mut [Body], dt: f32, gravity: [f32; 3]) {
    for body in bodies.iter_mut() {
        // Skip static bodies
        if body.shape_data[1] == 1 {
            continue;
        }
        
        // Apply gravity
        if body.mass_data[0] > 0.0 {
            body.velocity[0] += gravity[0] * dt;
            body.velocity[1] += gravity[1] * dt;
            body.velocity[2] += gravity[2] * dt;
        }
        
        // Update position
        body.position[0] += body.velocity[0] * dt;
        body.position[1] += body.velocity[1] * dt;
        body.position[2] += body.velocity[2] * dt;
    }
}