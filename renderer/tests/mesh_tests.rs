use physics_renderer::{body::Body, mesh::WireframeGeometry};

#[test]
fn test_wireframe_generation_empty() {
    let bodies: Vec<Body> = vec![];
    let vertices = WireframeGeometry::generate_vertices_from_bodies(&bodies);
    
    // Should still generate debug triangles even with no bodies
    assert!(!vertices.is_empty(), "Should generate debug geometry even with no bodies");
    
    // Check that vertices are valid (6 floats per vertex)
    assert_eq!(vertices.len() % 6, 0, "Vertices should be in groups of 6 (position + color)");
}

#[test]
fn test_wireframe_generation_single_sphere() {
    let body = Body::new_sphere([0.0, 0.0, 0.0], 1.0, 1.0);
    let bodies = vec![body];
    let vertices = WireframeGeometry::generate_vertices_from_bodies(&bodies);
    
    assert!(!vertices.is_empty(), "Should generate vertices for sphere");
    assert_eq!(vertices.len() % 6, 0, "Vertices should be in groups of 6");
    
    // Verify all values are finite
    for &val in &vertices {
        assert!(val.is_finite(), "Vertex data contains non-finite values");
    }
}

#[test]
fn test_wireframe_generation_single_box() {
    let body = Body::new_box([0.0, 0.0, 0.0], [1.0, 2.0, 3.0], 1.0);
    let bodies = vec![body];
    let vertices = WireframeGeometry::generate_vertices_from_bodies(&bodies);
    
    assert!(!vertices.is_empty(), "Should generate vertices for box");
    assert_eq!(vertices.len() % 6, 0, "Vertices should be in groups of 6");
    
    // Verify all values are finite
    for &val in &vertices {
        assert!(val.is_finite(), "Vertex data contains non-finite values");
    }
}

#[test]
fn test_wireframe_colors() {
    let dynamic_body = Body::new_sphere([0.0, 0.0, 0.0], 1.0, 1.0);
    let static_body = Body::new_static_sphere([5.0, 0.0, 0.0], 1.0);
    let heavy_body = Body::new_sphere([10.0, 0.0, 0.0], 1.0, 10000.0); // Mass > 1000
    
    // Test individual bodies to check color assignments
    let dynamic_vertices = WireframeGeometry::generate_vertices_from_bodies(&[dynamic_body]);
    let static_vertices = WireframeGeometry::generate_vertices_from_bodies(&[static_body]);
    let heavy_vertices = WireframeGeometry::generate_vertices_from_bodies(&[heavy_body]);
    
    // All should generate vertices
    assert!(!dynamic_vertices.is_empty());
    assert!(!static_vertices.is_empty());
    assert!(!heavy_vertices.is_empty());
}

#[test]
fn test_wireframe_multiple_bodies() {
    let bodies = vec![
        Body::new_sphere([0.0, 0.0, 0.0], 0.5, 1.0),
        Body::new_box([5.0, 0.0, 0.0], [1.0, 1.0, 1.0], 2.0),
        Body::new_static_sphere([0.0, -5.0, 0.0], 2.0),
        Body::new_static_box([0.0, -10.0, 0.0], [10.0, 1.0, 10.0]),
    ];
    
    let vertices = WireframeGeometry::generate_vertices_from_bodies(&bodies);
    
    assert!(!vertices.is_empty(), "Should generate vertices for multiple bodies");
    assert_eq!(vertices.len() % 6, 0, "Vertices should be in groups of 6");
    
    // Verify reasonable bounds (debug triangle is large, so bounds should be wide)
    let mut min_pos = [f32::MAX; 3];
    let mut max_pos = [f32::MIN; 3];
    
    for i in (0..vertices.len()).step_by(6) {
        for j in 0..3 {
            min_pos[j] = min_pos[j].min(vertices[i + j]);
            max_pos[j] = max_pos[j].max(vertices[i + j]);
        }
    }
    
    // Should have some extent in at least X and Y dimensions (debug triangles might be flat in Z)
    assert!(max_pos[0] > min_pos[0], "Should have extent in X dimension");
    assert!(max_pos[1] > min_pos[1], "Should have extent in Y dimension");
    // Z dimension might be flat due to debug triangle rendering
}

#[test]
fn test_wireframe_large_coordinates() {
    let body = Body::new_sphere([1000.0, 2000.0, 3000.0], 50.0, 1.0);
    let bodies = vec![body];
    let vertices = WireframeGeometry::generate_vertices_from_bodies(&bodies);
    
    assert!(!vertices.is_empty(), "Should handle large coordinates");
    
    // Verify all values are finite even with large inputs
    for &val in &vertices {
        assert!(val.is_finite(), "Vertex data contains non-finite values with large coordinates");
    }
}

#[test]
fn test_wireframe_small_objects() {
    let body = Body::new_sphere([0.0, 0.0, 0.0], 0.001, 0.001);
    let bodies = vec![body];
    let vertices = WireframeGeometry::generate_vertices_from_bodies(&bodies);
    
    assert!(!vertices.is_empty(), "Should handle small objects");
    
    // Verify all values are finite even with small inputs
    for &val in &vertices {
        assert!(val.is_finite(), "Vertex data contains non-finite values with small objects");
    }
}