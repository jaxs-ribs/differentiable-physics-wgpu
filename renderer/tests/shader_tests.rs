use std::fs;
use std::path::Path;

#[test]
fn test_shader_file_exists() {
    let shader_path = Path::new("src/shaders/sdf.wgsl");
    assert!(shader_path.exists(), "Shader file should exist at {:?}", shader_path);
}

#[test]
fn test_shader_valid_wgsl() {
    let shader_path = Path::new("src/shaders/sdf.wgsl");
    let shader_content = fs::read_to_string(shader_path).expect("Failed to read shader file");
    
    // Basic validation - check for required shader entry points
    assert!(shader_content.contains("@vertex"), "Shader should contain vertex entry point");
    assert!(shader_content.contains("@fragment"), "Shader should contain fragment entry point");
    assert!(shader_content.contains("vs_main"), "Shader should have vs_main function");
    assert!(shader_content.contains("fs_main"), "Shader should have fs_main function");
    
    // Check for required structures
    assert!(shader_content.contains("ViewProjection"), "Shader should define ViewProjection struct");
    assert!(shader_content.contains("Body"), "Shader should define Body struct");
    
    // Check for uniforms
    assert!(shader_content.contains("@group(0) @binding(0)"), "Shader should have binding 0");
    assert!(shader_content.contains("@group(0) @binding(1)"), "Shader should have binding 1");
    
    // Check SDF functions
    assert!(shader_content.contains("sdSphere"), "Shader should have sphere SDF function");
    assert!(shader_content.contains("sdBox"), "Shader should have box SDF function");
    assert!(shader_content.contains("sdCapsule"), "Shader should have capsule SDF function");
}

#[test]
fn test_shader_raymarching() {
    let shader_path = Path::new("src/shaders/sdf.wgsl");
    let shader_content = fs::read_to_string(shader_path).expect("Failed to read shader file");
    
    // Check raymarching implementation
    assert!(shader_content.contains("map_scene"), "Shader should have scene mapping function");
    assert!(shader_content.contains("calculate_normal"), "Shader should have normal calculation");
    assert!(shader_content.contains("raymarch"), "Shader should have raymarching function");
}

#[test]
fn test_shader_sdf_shapes() {
    let shader_path = Path::new("src/shaders/sdf.wgsl");
    let shader_content = fs::read_to_string(shader_path).expect("Failed to read shader file");
    
    // Check shape constants
    assert!(shader_content.contains("SHAPE_SPHERE"), "Shader should define sphere shape constant");
    assert!(shader_content.contains("SHAPE_BOX"), "Shader should define box shape constant");
    assert!(shader_content.contains("SHAPE_CAPSULE"), "Shader should define capsule shape constant");
}