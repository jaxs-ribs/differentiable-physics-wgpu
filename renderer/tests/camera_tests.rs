use physics_renderer::camera::Camera;
use glam::Mat4;

#[test]
fn test_camera_creation() {
    let camera = Camera::new(16.0 / 9.0);
    let matrix = camera.view_projection_matrix_transposed();
    
    // Check that the matrix is not identity
    let identity = Mat4::IDENTITY.to_cols_array_2d();
    assert_ne!(matrix, identity, "Camera matrix should not be identity");
    
    // Check that the matrix is valid (no NaN or infinity values)
    for row in &matrix {
        for &val in row {
            assert!(val.is_finite(), "Camera matrix contains non-finite values");
        }
    }
}

#[test]
fn test_camera_rotation() {
    let mut camera = Camera::new(16.0 / 9.0);
    let initial_matrix = camera.view_projection_matrix_transposed();
    
    // Rotate the camera
    camera.rotate(0.1, 0.1);
    let rotated_matrix = camera.view_projection_matrix_transposed();
    
    // Matrices should be different after rotation
    assert_ne!(initial_matrix, rotated_matrix, "Camera matrix should change after rotation");
}

#[test]
fn test_camera_zoom() {
    let mut camera = Camera::new(16.0 / 9.0);
    let initial_matrix = camera.view_projection_matrix_transposed();
    
    // Zoom the camera
    camera.zoom(5.0);
    let zoomed_matrix = camera.view_projection_matrix_transposed();
    
    // Matrices should be different after zoom
    assert_ne!(initial_matrix, zoomed_matrix, "Camera matrix should change after zoom");
}

#[test]
fn test_camera_aspect_ratio_update() {
    let mut camera = Camera::new(16.0 / 9.0);
    let initial_matrix = camera.view_projection_matrix_transposed();
    
    // Update aspect ratio
    camera.update_aspect_ratio(4.0 / 3.0);
    let updated_matrix = camera.view_projection_matrix_transposed();
    
    // Matrices should be different after aspect ratio change
    assert_ne!(initial_matrix, updated_matrix, "Camera matrix should change after aspect ratio update");
}

#[test]
fn test_camera_rotation_limits() {
    let mut camera = Camera::new(16.0 / 9.0);
    
    // Try to rotate beyond limits
    camera.rotate(0.0, 10.0); // Large vertical rotation
    let matrix1 = camera.view_projection_matrix_transposed();
    
    camera.rotate(0.0, -20.0); // Large rotation in opposite direction
    let matrix2 = camera.view_projection_matrix_transposed();
    
    // Check that the camera still produces valid matrices
    for row in &matrix1 {
        for &val in row {
            assert!(val.is_finite(), "Camera matrix contains non-finite values after extreme rotation");
        }
    }
    
    for row in &matrix2 {
        for &val in row {
            assert!(val.is_finite(), "Camera matrix contains non-finite values after extreme rotation");
        }
    }
}

#[test]
fn test_camera_minimum_zoom() {
    let mut camera = Camera::new(16.0 / 9.0);
    
    // Try to zoom very close
    camera.zoom(100.0);
    let matrix = camera.view_projection_matrix_transposed();
    
    // Check that the camera still produces valid matrices
    for row in &matrix {
        for &val in row {
            assert!(val.is_finite(), "Camera matrix contains non-finite values at minimum zoom");
        }
    }
}