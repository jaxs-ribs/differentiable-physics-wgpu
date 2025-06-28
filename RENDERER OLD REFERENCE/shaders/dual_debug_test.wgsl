struct ViewProjection {
    matrix: mat4x4<f32>,
}

struct ColorUniform {
    color: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> view_proj: ViewProjection;

@group(0) @binding(1)
var<uniform> color_uniform: ColorUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    
    // DEBUG: Create a simple triangle that fills the screen
    // Map vertex positions to NDC space directly for testing
    if (input.position.x < -50.0) {
        output.position = vec4<f32>(-1.0, -1.0, 0.5, 1.0);
    } else if (input.position.x > 50.0) {
        output.position = vec4<f32>(1.0, -1.0, 0.5, 1.0);
    } else {
        output.position = vec4<f32>(0.0, 1.0, 0.5, 1.0);
    }
    
    output.color = input.color;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Use bright colors for visibility
    return vec4<f32>(input.color, 1.0);
}