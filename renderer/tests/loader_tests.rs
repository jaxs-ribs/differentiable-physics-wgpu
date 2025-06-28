use physics_renderer::loader::TrajectoryLoader;
use std::io::Write;
use tempfile::NamedTempFile;

fn create_test_npy_file(num_frames: usize, num_bodies: usize) -> NamedTempFile {
    let mut file = NamedTempFile::new().unwrap();
    
    // NPY format header
    let shape = format!("({}, {})", num_frames, num_bodies * 18);
    let header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {}}}", shape);
    let header_len = header.len();
    let padding = (16 - (10 + header_len) % 16) % 16;
    let header_with_padding = format!("{}{}", header, " ".repeat(padding));
    
    // Write magic string and version
    file.write_all(b"\x93NUMPY").unwrap();
    file.write_all(&[1u8, 0u8]).unwrap(); // Version 1.0
    
    // Write header length (little-endian)
    let total_header_len = header_with_padding.len() + 1; // +1 for newline
    file.write_all(&(total_header_len as u16).to_le_bytes()).unwrap();
    
    // Write header
    file.write_all(header_with_padding.as_bytes()).unwrap();
    file.write_all(b"\n").unwrap();
    
    // Write data
    for frame in 0..num_frames {
        for body_idx in 0..num_bodies {
            // Position
            file.write_all(&(body_idx as f32).to_le_bytes()).unwrap();
            file.write_all(&(frame as f32).to_le_bytes()).unwrap();
            file.write_all(&0.0f32.to_le_bytes()).unwrap();
            
            // Velocity
            file.write_all(&1.0f32.to_le_bytes()).unwrap();
            file.write_all(&0.0f32.to_le_bytes()).unwrap();
            file.write_all(&0.0f32.to_le_bytes()).unwrap();
            
            // Orientation (quaternion)
            file.write_all(&1.0f32.to_le_bytes()).unwrap();
            file.write_all(&0.0f32.to_le_bytes()).unwrap();
            file.write_all(&0.0f32.to_le_bytes()).unwrap();
            file.write_all(&0.0f32.to_le_bytes()).unwrap();
            
            // Angular velocity
            file.write_all(&0.0f32.to_le_bytes()).unwrap();
            file.write_all(&0.0f32.to_le_bytes()).unwrap();
            file.write_all(&0.0f32.to_le_bytes()).unwrap();
            
            // Mass
            file.write_all(&1.0f32.to_le_bytes()).unwrap();
            
            // Shape type (0 = sphere)
            file.write_all(&0.0f32.to_le_bytes()).unwrap();
            
            // Shape params (radius for sphere)
            file.write_all(&0.5f32.to_le_bytes()).unwrap();
            file.write_all(&0.0f32.to_le_bytes()).unwrap();
            file.write_all(&0.0f32.to_le_bytes()).unwrap();
        }
    }
    
    file.flush().unwrap();
    file
}

#[test]
fn test_load_trajectory() {
    let file = create_test_npy_file(10, 5);
    let path = file.path().to_path_buf();
    
    let run = TrajectoryLoader::load_trajectory(&path).unwrap();
    let metadata = TrajectoryLoader::get_metadata(&run);
    
    assert_eq!(metadata.num_frames, 10);
    assert_eq!(metadata.num_bodies, 5);
    assert_eq!(metadata.file_path, path.to_string_lossy());
}

#[test]
fn test_get_bodies_at_frame() {
    let file = create_test_npy_file(5, 3);
    let path = file.path().to_path_buf();
    
    let run = TrajectoryLoader::load_trajectory(&path).unwrap();
    
    // Test first frame
    let bodies = TrajectoryLoader::get_bodies_at_frame(&run, 0).unwrap();
    assert_eq!(bodies.len(), 3);
    
    // Check first body
    assert_eq!(bodies[0].position[0], 0.0); // body_idx 0
    assert_eq!(bodies[0].position[1], 0.0); // frame 0
    assert_eq!(bodies[0].position[2], 0.0);
    
    // Test later frame
    let bodies = TrajectoryLoader::get_bodies_at_frame(&run, 2).unwrap();
    assert_eq!(bodies.len(), 3);
    
    // Check second body
    assert_eq!(bodies[1].position[0], 1.0); // body_idx 1
    assert_eq!(bodies[1].position[1], 2.0); // frame 2
    assert_eq!(bodies[1].position[2], 0.0);
}

#[test]
fn test_frame_bounds() {
    let file = create_test_npy_file(5, 2);
    let path = file.path().to_path_buf();
    
    let run = TrajectoryLoader::load_trajectory(&path).unwrap();
    
    // Valid frame
    assert!(TrajectoryLoader::get_bodies_at_frame(&run, 4).is_ok());
    
    // Out of bounds frame
    assert!(TrajectoryLoader::get_bodies_at_frame(&run, 5).is_err());
}

#[test]
fn test_empty_trajectory() {
    let file = create_test_npy_file(0, 0);
    let path = file.path().to_path_buf();
    
    let run = TrajectoryLoader::load_trajectory(&path).unwrap();
    let metadata = TrajectoryLoader::get_metadata(&run);
    
    assert_eq!(metadata.num_frames, 0);
    assert_eq!(metadata.num_bodies, 0);
}

#[test]
fn test_single_frame_single_body() {
    let file = create_test_npy_file(1, 1);
    let path = file.path().to_path_buf();
    
    let run = TrajectoryLoader::load_trajectory(&path).unwrap();
    let metadata = TrajectoryLoader::get_metadata(&run);
    
    assert_eq!(metadata.num_frames, 1);
    assert_eq!(metadata.num_bodies, 1);
    
    let bodies = TrajectoryLoader::get_bodies_at_frame(&run, 0).unwrap();
    assert_eq!(bodies.len(), 1);
}

#[test]
fn test_body_properties() {
    let file = create_test_npy_file(1, 1);
    let path = file.path().to_path_buf();
    
    let run = TrajectoryLoader::load_trajectory(&path).unwrap();
    let bodies = TrajectoryLoader::get_bodies_at_frame(&run, 0).unwrap();
    
    let body = &bodies[0];
    
    // Check velocity
    assert_eq!(body.velocity[0], 1.0);
    assert_eq!(body.velocity[1], 0.0);
    assert_eq!(body.velocity[2], 0.0);
    
    // Check orientation (quaternion)
    assert_eq!(body.orientation[0], 1.0);
    assert_eq!(body.orientation[1], 0.0);
    assert_eq!(body.orientation[2], 0.0);
    assert_eq!(body.orientation[3], 0.0);
    
    // Check mass
    assert_eq!(body.mass_data[0], 1.0);
    assert_eq!(body.mass_data[1], 1.0); // inverse mass
    
    // Check shape
    assert_eq!(body.shape_data[0], 0); // sphere
    assert_eq!(body.shape_params[0], 0.5); // radius
}

#[test]
fn test_metadata_duration() {
    let file = create_test_npy_file(120, 10); // 2 seconds at 60 FPS
    let path = file.path().to_path_buf();
    
    let run = TrajectoryLoader::load_trajectory(&path).unwrap();
    let metadata = TrajectoryLoader::get_metadata(&run);
    
    assert_eq!(metadata.duration_seconds, 2.0);
}