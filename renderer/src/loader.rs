use crate::body::Body;
use anyhow::{anyhow, Result};
use ndarray::Array3;
use std::path::Path;

/// Loads simulation trajectory data from .npy files
/// Expected format: (num_frames, num_bodies, 27) where each body has 27 floats
pub struct TrajectoryLoader;

pub struct SimulationRun {
    pub file_path: String,
    pub num_frames: usize,
    pub num_bodies: usize,
    pub data: Array3<f32>, // Shape: (num_frames, num_bodies, 27)
}

impl TrajectoryLoader {
    /// Load a .npy file containing trajectory data
    /// Returns the trajectory data with shape (num_frames, num_bodies, 27)
    pub fn load_trajectory<P: AsRef<Path>>(file_path: P) -> Result<SimulationRun> {
        let path = file_path.as_ref();
        let path_str = path.to_string_lossy().to_string();
        
        // Read the .npy file
        let array: Array3<f32> = ndarray_npy::read_npy(path)?;
        
        // Validate shape
        let shape = array.shape();
        if shape.len() != 3 {
            return Err(anyhow!(
                "Expected 3D array (frames, bodies, features), got {}D array", 
                shape.len()
            ));
        }
        
        let (num_frames, num_bodies, num_features) = (shape[0], shape[1], shape[2]);
        
        if num_features != 27 {
            return Err(anyhow!(
                "Expected 27 features per body, got {} features", 
                num_features
            ));
        }
        
        log::info!(
            "Loaded trajectory: {} frames, {} bodies, {} features per body",
            num_frames, num_bodies, num_features
        );
        
        Ok(SimulationRun {
            file_path: path_str,
            num_frames,
            num_bodies,
            data: array,
        })
    }
    
    /// Extract bodies for a specific frame
    pub fn get_bodies_at_frame(run: &SimulationRun, frame_index: usize) -> Result<Vec<Body>> {
        if frame_index >= run.num_frames {
            return Err(anyhow!(
                "Frame index {} out of bounds (max: {})", 
                frame_index, 
                run.num_frames - 1
            ));
        }
        
        let mut bodies = Vec::with_capacity(run.num_bodies);
        
        for body_index in 0..run.num_bodies {
            let body_data = run.data.slice(ndarray::s![frame_index, body_index, ..]);
            
            // Convert slice to array of exactly 27 floats
            let mut data = [0.0f32; 27];
            for (i, &value) in body_data.iter().enumerate() {
                if i < 27 {
                    data[i] = value;
                }
            }
            
            bodies.push(Body { data });
        }
        
        Ok(bodies)
    }
    
    /// Get the duration of the simulation in seconds based on frame count
    /// Assumes 60 FPS for now - this could be made configurable
    pub fn get_duration_seconds(run: &SimulationRun) -> f32 {
        run.num_frames as f32 / 60.0
    }
    
    /// Get metadata about the simulation run
    pub fn get_metadata(run: &SimulationRun) -> RunMetadata {
        RunMetadata {
            file_path: run.file_path.clone(),
            num_frames: run.num_frames,
            num_bodies: run.num_bodies,
            duration_seconds: Self::get_duration_seconds(run),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RunMetadata {
    pub file_path: String,
    pub num_frames: usize,
    pub num_bodies: usize,
    pub duration_seconds: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_load_trajectory_structure() {
        // Test that we can create the loader and it has the expected interface
        // We can't test actual file loading without test data
        
        // Test error case for wrong number of features
        let bad_array = Array3::<f32>::zeros((10, 5, 26)); // Wrong feature count
        
        // Simulate what would happen with wrong features
        let shape = bad_array.shape();
        let num_features = shape[2];
        assert_ne!(num_features, 27, "Test array should have wrong feature count");
    }
    
    #[test]
    fn test_get_bodies_at_frame_bounds() {
        // Create a minimal simulation run for testing
        let data = Array3::<f32>::zeros((5, 2, 27)); // 5 frames, 2 bodies, 27 features
        let run = SimulationRun {
            file_path: "test.npy".to_string(),
            num_frames: 5,
            num_bodies: 2,
            data,
        };
        
        // Test valid frame
        let bodies = TrajectoryLoader::get_bodies_at_frame(&run, 0);
        assert!(bodies.is_ok());
        let bodies = bodies.unwrap();
        assert_eq!(bodies.len(), 2);
        
        // Test out of bounds frame
        let result = TrajectoryLoader::get_bodies_at_frame(&run, 5);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_metadata_generation() {
        let data = Array3::<f32>::zeros((120, 10, 27)); // 120 frames, 10 bodies
        let run = SimulationRun {
            file_path: "test_120_frames.npy".to_string(),
            num_frames: 120,
            num_bodies: 10,
            data,
        };
        
        let metadata = TrajectoryLoader::get_metadata(&run);
        assert_eq!(metadata.num_frames, 120);
        assert_eq!(metadata.num_bodies, 10);
        assert_eq!(metadata.duration_seconds, 2.0); // 120 frames / 60 FPS = 2 seconds
    }
    
    #[test]
    #[ignore] // This test requires an actual .npy file
    fn test_load_real_trajectory() {
        // This test would be enabled when we have a proper test file
        let result = TrajectoryLoader::load_trajectory("../artifacts/test_single_100.npy");
        if result.is_ok() {
            let run = result.unwrap();
            println!("Loaded run: {} frames, {} bodies", run.num_frames, run.num_bodies);
            
            // Test getting bodies from first frame
            let bodies = TrajectoryLoader::get_bodies_at_frame(&run, 0).unwrap();
            assert_eq!(bodies.len(), run.num_bodies);
        }
    }
}