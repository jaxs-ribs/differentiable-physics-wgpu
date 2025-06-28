use std::fs;
use std::path::Path;

#[test]
fn test_shader_file_exists() {
    let shader_path = Path::new("src/shaders/wireframe.wgsl");
    assert!(shader_path.exists(), "Shader file should exist at {:?}", shader_path);
}

#[test]
fn test_shader_valid_wgsl() {
    let shader_path = Path::new("src/shaders/wireframe.wgsl");
    let shader_content = fs::read_to_string(shader_path).expect("Failed to read shader file");
    
    // Basic validation - check for required shader entry points
    assert!(shader_content.contains("@vertex"), "Shader should contain vertex entry point");
    assert!(shader_content.contains("@fragment"), "Shader should contain fragment entry point");
    assert!(shader_content.contains("vs_main"), "Shader should have vs_main function");
    assert!(shader_content.contains("fs_main"), "Shader should have fs_main function");
    
    // Check for required structures
    assert!(shader_content.contains("ViewProjection"), "Shader should define ViewProjection struct");
    assert!(shader_content.contains("ColorUniform"), "Shader should define ColorUniform struct");
    assert!(shader_content.contains("VertexInput"), "Shader should define VertexInput struct");
    assert!(shader_content.contains("VertexOutput"), "Shader should define VertexOutput struct");
    
    // Check for uniforms
    assert!(shader_content.contains("@group(0) @binding(0)"), "Shader should have binding 0");
    assert!(shader_content.contains("@group(0) @binding(1)"), "Shader should have binding 1");
    
    // Check vertex attributes
    assert!(shader_content.contains("@location(0) position"), "Shader should have position attribute");
    assert!(shader_content.contains("@location(1) color"), "Shader should have color attribute");
}

#[test]
fn test_shader_matrix_multiplication() {
    let shader_path = Path::new("src/shaders/wireframe.wgsl");
    let shader_content = fs::read_to_string(shader_path).expect("Failed to read shader file");
    
    // Check that vertex shader applies view-projection matrix
    assert!(shader_content.contains("view_proj.matrix"), "Shader should use view projection matrix");
    assert!(shader_content.contains("vec4<f32>"), "Shader should use vec4 for homogeneous coordinates");
}

#[test]
fn test_shader_color_passthrough() {
    let shader_path = Path::new("src/shaders/wireframe.wgsl");
    let shader_content = fs::read_to_string(shader_path).expect("Failed to read shader file");
    
    // Check that colors are passed from vertex to fragment shader
    assert!(shader_content.contains("output.color = input.color"), "Vertex shader should pass through color");
    assert!(shader_content.contains("return vec4<f32>(input.color, 1.0)"), "Fragment shader should output color with alpha");
}