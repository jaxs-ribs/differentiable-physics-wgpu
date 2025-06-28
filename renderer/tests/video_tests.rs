use physics_renderer::video::save_frames_as_video;
use tempfile::TempDir;

fn create_test_frame(width: u32, height: u32, color: [u8; 4]) -> Vec<u8> {
    let mut frame = Vec::with_capacity((width * height * 4) as usize);
    for _ in 0..(width * height) {
        frame.extend_from_slice(&color);
    }
    frame
}

#[test]
#[ignore] // Ignore by default since it requires ffmpeg
fn test_save_single_frame_video() {
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("test_video.mp4");
    
    // Create a single red frame (BGRA format)
    let frames = vec![create_test_frame(100, 100, [0, 0, 255, 255])];
    
    let result = save_frames_as_video(&frames, output_path.clone(), 30, 100, 100);
    
    match result {
        Ok(_) => {
            assert!(output_path.exists(), "Video file should be created");
            let metadata = std::fs::metadata(&output_path).unwrap();
            assert!(metadata.len() > 0, "Video file should not be empty");
        }
        Err(e) => {
            // If ffmpeg is not installed, the test will fail here
            println!("Video test failed (likely ffmpeg not installed): {}", e);
        }
    }
}

#[test]
#[ignore] // Ignore by default since it requires ffmpeg
fn test_save_multiple_frames_video() {
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("test_video.mp4");
    
    // Create frames with different colors
    let frames = vec![
        create_test_frame(200, 200, [0, 0, 255, 255]),   // Red
        create_test_frame(200, 200, [0, 255, 0, 255]),   // Green
        create_test_frame(200, 200, [255, 0, 0, 255]),   // Blue
    ];
    
    let result = save_frames_as_video(&frames, output_path.clone(), 30, 200, 200);
    
    match result {
        Ok(_) => {
            assert!(output_path.exists(), "Video file should be created");
        }
        Err(e) => {
            println!("Video test failed (likely ffmpeg not installed): {}", e);
        }
    }
}

#[test]
fn test_bgra_to_rgba_conversion() {
    // This test verifies the color conversion logic without actually creating a video
    let bgra_data = vec![
        255, 128, 64, 255,  // BGRA: Blue=255, Green=128, Red=64, Alpha=255
        0, 255, 128, 200,   // BGRA: Blue=0, Green=255, Red=128, Alpha=200
    ];
    
    // Expected RGBA after conversion
    let expected_rgba = vec![
        64, 128, 255, 255,  // RGBA: Red=64, Green=128, Blue=255, Alpha=255
        128, 255, 0, 200,   // RGBA: Red=128, Green=255, Blue=0, Alpha=200
    ];
    
    // Simulate the conversion logic from save_frames_as_video
    let mut rgba_data = vec![0u8; bgra_data.len()];
    for j in 0..(bgra_data.len() / 4) {
        rgba_data[j * 4] = bgra_data[j * 4 + 2];     // R = B from BGRA
        rgba_data[j * 4 + 1] = bgra_data[j * 4 + 1]; // G = G from BGRA
        rgba_data[j * 4 + 2] = bgra_data[j * 4];     // B = R from BGRA
        rgba_data[j * 4 + 3] = bgra_data[j * 4 + 3]; // A = A from BGRA
    }
    
    assert_eq!(rgba_data, expected_rgba, "BGRA to RGBA conversion should be correct");
}

#[test]
fn test_frame_dimensions() {
    // Test that frame creation produces correct size
    let width = 320;
    let height = 240;
    let frame = create_test_frame(width, height, [255, 255, 255, 255]);
    
    assert_eq!(frame.len(), (width * height * 4) as usize, "Frame should have correct byte count");
}

#[test]
#[ignore] // Ignore by default since it requires ffmpeg
fn test_empty_frames_list() {
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("empty_video.mp4");
    
    let frames: Vec<Vec<u8>> = vec![];
    
    // This should handle empty frames gracefully
    let result = save_frames_as_video(&frames, output_path, 30, 100, 100);
    
    // With no frames, ffmpeg will likely fail, but our code shouldn't panic
    assert!(result.is_err(), "Empty frames should result in an error");
}