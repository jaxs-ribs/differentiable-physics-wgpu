use physics_renderer::body::Body;
use bytemuck::{Pod, Zeroable};

#[test]
fn test_body_size_and_alignment() {
    assert_eq!(std::mem::size_of::<Body>(), 112, "Body struct should be exactly 112 bytes");
    assert_eq!(std::mem::align_of::<Body>(), 16, "Body struct should be 16-byte aligned");
}

#[test]
fn test_body_is_pod() {
    // This test verifies that Body implements Pod and Zeroable
    // We can't use trait objects for Pod/Zeroable, but we can verify they work
    let body = Body::new_sphere([0.0, 0.0, 0.0], 1.0, 1.0);
    let _bytes: &[u8] = bytemuck::bytes_of(&body);
    let _zeroed: Body = bytemuck::Zeroable::zeroed();
}

#[test]
fn test_sphere_creation() {
    let pos = [1.0, 2.0, 3.0];
    let radius = 0.5;
    let mass = 10.0;
    
    let body = Body::new_sphere(pos, radius, mass);
    
    assert_eq!(body.position[0], 1.0);
    assert_eq!(body.position[1], 2.0);
    assert_eq!(body.position[2], 3.0);
    assert_eq!(body.position[3], 0.0); // padding
    
    assert_eq!(body.velocity, [0.0; 4]);
    assert_eq!(body.orientation, [1.0, 0.0, 0.0, 0.0]); // identity quaternion
    assert_eq!(body.angular_vel, [0.0; 4]);
    
    assert_eq!(body.mass_data[0], mass);
    assert_eq!(body.mass_data[1], 1.0 / mass); // inverse mass
    
    assert_eq!(body.shape_data[0], 0); // sphere type
    assert_eq!(body.shape_data[1], 0); // dynamic flag
    
    assert_eq!(body.shape_params[0], radius);
}

#[test]
fn test_static_sphere_creation() {
    let pos = [5.0, 6.0, 7.0];
    let radius = 2.0;
    
    let body = Body::new_static_sphere(pos, radius);
    
    assert_eq!(body.position[0], 5.0);
    assert_eq!(body.position[1], 6.0);
    assert_eq!(body.position[2], 7.0);
    
    assert_eq!(body.mass_data[0], 0.0); // static bodies have no mass
    assert_eq!(body.mass_data[1], 0.0); // no inverse mass
    
    assert_eq!(body.shape_data[0], 0); // sphere type
    assert_eq!(body.shape_data[1], 1); // static flag
    
    assert_eq!(body.shape_params[0], radius);
}

#[test]
fn test_box_creation() {
    let pos = [0.0, 1.0, 0.0];
    let half_extents = [2.0, 3.0, 4.0];
    let mass = 50.0;
    
    let body = Body::new_box(pos, half_extents, mass);
    
    assert_eq!(body.position[0], 0.0);
    assert_eq!(body.position[1], 1.0);
    assert_eq!(body.position[2], 0.0);
    
    assert_eq!(body.mass_data[0], mass);
    assert_eq!(body.mass_data[1], 1.0 / mass);
    
    assert_eq!(body.shape_data[0], 2); // box type
    assert_eq!(body.shape_data[1], 0); // dynamic flag
    
    assert_eq!(body.shape_params[0], 2.0);
    assert_eq!(body.shape_params[1], 3.0);
    assert_eq!(body.shape_params[2], 4.0);
}

#[test]
fn test_static_box_creation() {
    let pos = [-10.0, 0.0, 10.0];
    let half_extents = [100.0, 1.0, 100.0];
    
    let body = Body::new_static_box(pos, half_extents);
    
    assert_eq!(body.position[0], -10.0);
    assert_eq!(body.position[1], 0.0);
    assert_eq!(body.position[2], 10.0);
    
    assert_eq!(body.mass_data[0], 0.0); // static bodies have no mass
    assert_eq!(body.mass_data[1], 0.0); // no inverse mass
    
    assert_eq!(body.shape_data[0], 2); // box type
    assert_eq!(body.shape_data[1], 1); // static flag
    
    assert_eq!(body.shape_params[0], 100.0);
    assert_eq!(body.shape_params[1], 1.0);
    assert_eq!(body.shape_params[2], 100.0);
}

#[test]
fn test_zero_mass_handling() {
    // Test that zero mass is handled correctly
    let body = Body::new_sphere([0.0, 0.0, 0.0], 1.0, 0.0);
    
    assert_eq!(body.mass_data[0], 0.0);
    assert_eq!(body.mass_data[1], 0.0); // inverse mass should be 0, not infinity
}

#[test]
fn test_body_copy_trait() {
    let body1 = Body::new_sphere([1.0, 2.0, 3.0], 0.5, 1.0);
    let body2 = body1; // Copy
    
    // Both should have the same values
    assert_eq!(body1.position, body2.position);
    assert_eq!(body1.velocity, body2.velocity);
    assert_eq!(body1.orientation, body2.orientation);
    assert_eq!(body1.angular_vel, body2.angular_vel);
    assert_eq!(body1.mass_data, body2.mass_data);
    assert_eq!(body1.shape_data, body2.shape_data);
    assert_eq!(body1.shape_params, body2.shape_params);
}