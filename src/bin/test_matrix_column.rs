use physics_core::test_utils::math::{MatrixOperations, print_matrix};

fn main() {
    println!("Testing column-major vs row-major matrix multiplication\n");
    
    // Simple identity matrix test
    let _identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    
    // Create a simple translation matrix
    let translation = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [5.0, 10.0, 15.0, 1.0], // Translation in row-major
    ];
    
    println!("Translation matrix (row-major):");
    print_matrix(&translation);
    
    // Create a simple scale matrix
    let scale = [
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 3.0, 0.0, 0.0],
        [0.0, 0.0, 4.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    
    println!("\nScale matrix:");
    print_matrix(&scale);
    
    // Multiply scale * translation (apply translation first, then scale)
    let result = MatrixOperations::matrix_multiply(&scale, &translation);
    
    println!("\nScale * Translation (row-major):");
    print_matrix(&result);
    
    // Test with a point
    let point = [1.0, 1.0, 1.0, 1.0];
    println!("\nTest point: [{:.1}, {:.1}, {:.1}, {:.1}]", point[0], point[1], point[2], point[3]);
    
    // Apply translation first
    let mut translated = [0.0; 4];
    for i in 0..4 {
        for j in 0..4 {
            translated[i] += translation[i][j] * point[j];
        }
    }
    println!("\nAfter translation: [{:.1}, {:.1}, {:.1}, {:.1}]", 
        translated[0], translated[1], translated[2], translated[3]);
    
    // Then apply scale
    let mut final_result = [0.0; 4];
    for i in 0..4 {
        for j in 0..4 {
            final_result[i] += scale[i][j] * translated[j];
        }
    }
    println!("After scale: [{:.1}, {:.1}, {:.1}, {:.1}]", 
        final_result[0], final_result[1], final_result[2], final_result[3]);
    
    // Apply combined matrix
    let mut combined_result = [0.0; 4];
    for i in 0..4 {
        for j in 0..4 {
            combined_result[i] += result[i][j] * point[j];
        }
    }
    println!("\nUsing combined matrix: [{:.1}, {:.1}, {:.1}, {:.1}]", 
        combined_result[0], combined_result[1], combined_result[2], combined_result[3]);
    
    // Now test actual view-projection matrix
    println!("\n\nTesting View-Projection Matrix:");
    println!("================================");
    
    let aspect = 1024.0 / 768.0;
    let projection = MatrixOperations::perspective_matrix(45.0_f32.to_radians(), aspect, 0.1, 1000.0);
    let view = MatrixOperations::look_at_matrix(
        [0.0, 20.0, 50.0],  // eye position
        [0.0, 5.0, 0.0],    // look at
        [0.0, 1.0, 0.0],    // up
    );
    
    let view_proj = MatrixOperations::matrix_multiply(&projection, &view);
    
    println!("\nView-Projection Matrix:");
    print_matrix(&view_proj);
    
    // Test a point at the origin
    let test_point = [0.0, 0.0, 0.0, 1.0];
    let mut transformed = [0.0; 4];
    
    for i in 0..4 {
        for j in 0..4 {
            transformed[i] += view_proj[i][j] * test_point[j];
        }
    }
    
    println!("\nOrigin transformed: [{:.3}, {:.3}, {:.3}, {:.3}]",
        transformed[0], transformed[1], transformed[2], transformed[3]);
    
    let clip_x = transformed[0] / transformed[3];
    let clip_y = transformed[1] / transformed[3];
    let clip_z = transformed[2] / transformed[3];
    
    println!("Clip space: [{:.3}, {:.3}, {:.3}]", clip_x, clip_y, clip_z);
    
    // For column-major (what WGSL expects), we need to transpose
    let view_proj_transposed = MatrixOperations::transpose_matrix(&view_proj);
    
    println!("\nTransposed for GPU (column-major):");
    print_matrix(&view_proj_transposed);
}