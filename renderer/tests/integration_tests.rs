use physics_renderer::{body::Body, camera::Camera};

#[test]
fn test_camera_matrices() {
    // Create camera and get view-projection matrix
    let camera = Camera::new(16.0 / 9.0);
    let matrix = camera.view_projection_matrix_transposed();
    
    // Verify matrix is valid
    for row in &matrix {
        for &val in row {
            assert!(val.is_finite());
        }
    }
}

#[test]
fn test_body_creation() {
    // Create various body types
    let bodies = vec![
        Body::new_box([0.0, 5.0, 0.0], [1.0, 1.0, 1.0], 1.0),
        Body::new_box([5.0, 5.0, 0.0], [0.5, 2.0, 0.5], 0.5),
        Body::new_static_box([0.0, 0.0, 0.0], [10.0, 0.1, 10.0]),
    ];
    
    // Verify bodies are created correctly
    assert_eq!(bodies.len(), 3);
    assert_eq!(bodies[0].mass_data[0], 1.0);
    assert_eq!(bodies[1].mass_data[0], 0.5);
    assert_eq!(bodies[2].mass_data[0], 0.0); // Static body has no mass
}

#[test]
fn test_mixed_scene_bodies() {
    // Create a complex scene with various body types
    let mut bodies = Vec::new();
    
    // Add a ground plane
    bodies.push(Body::new_static_box([0.0, -10.0, 0.0], [50.0, 1.0, 50.0]));
    
    // Add some dynamic spheres
    for i in 0..10 {
        let x = (i as f32 - 5.0) * 2.0;
        bodies.push(Body::new_sphere([x, 5.0, 0.0], 0.5, 1.0));
    }
    
    // Add some dynamic boxes
    for i in 0..5 {
        let z = (i as f32 - 2.5) * 3.0;
        bodies.push(Body::new_box([0.0, 10.0, z], [1.0, 1.0, 1.0], 2.0));
    }
    
    // Add some walls
    bodies.push(Body::new_static_box([-20.0, 0.0, 0.0], [1.0, 10.0, 20.0]));
    bodies.push(Body::new_static_box([20.0, 0.0, 0.0], [1.0, 10.0, 20.0]));
    
    // Verify body count
    assert_eq!(bodies.len(), 18); // 1 ground + 10 spheres + 5 boxes + 2 walls
}

#[test]
fn test_camera_movement() {
    // Create camera at different positions
    let camera1 = Camera::new(16.0 / 9.0);
    let mut camera2 = Camera::new(16.0 / 9.0);
    
    // Move second camera
    camera2.rotate(std::f32::consts::PI / 4.0, 0.0);
    camera2.zoom(-10.0);
    
    let matrix1 = camera1.view_projection_matrix_transposed();
    let matrix2 = camera2.view_projection_matrix_transposed();
    
    // Different camera positions should give different matrices
    assert_ne!(matrix1, matrix2);
}

#[test]
fn test_large_body_count() {
    // Test with many bodies
    let mut bodies = Vec::new();
    
    // Create a grid of spheres
    for x in -10..=10 {
        for y in -10..=10 {
            for z in -10..=10 {
                bodies.push(Body::new_sphere(
                    [x as f32 * 2.0, y as f32 * 2.0, z as f32 * 2.0],
                    0.5,
                    1.0
                ));
            }
        }
    }
    
    // Should handle large body counts
    assert_eq!(bodies.len(), 21 * 21 * 21); // 9261 bodies
}

#[test] 
fn test_extreme_scales() {
    let bodies = vec![
        Body::new_sphere([0.0, 0.0, 0.0], 0.001, 0.001), // Very small
        Body::new_sphere([100.0, 0.0, 0.0], 1.0, 1.0),   // Normal
        Body::new_sphere([0.0, 1000.0, 0.0], 100.0, 1000.0), // Very large
        Body::new_box([0.0, 0.0, 1000.0], [0.01, 0.01, 0.01], 0.001), // Tiny box far away
        Body::new_box([0.0, 0.0, -1000.0], [100.0, 100.0, 100.0], 10000.0), // Huge box far away
    ];
    
    // Should handle extreme scales without issues
    assert_eq!(bodies.len(), 5);
    
    // Verify all position values are finite
    for body in &bodies {
        assert!(body.position[0].is_finite());
        assert!(body.position[1].is_finite());
        assert!(body.position[2].is_finite());
    }
}