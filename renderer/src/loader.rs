use crate::body::Body;
use anyhow::{anyhow, Result};
use std::path::Path;

/// Loads simulation trajectory data from .npy files
pub struct TrajectoryLoader;

pub struct TrajectoryRun {
    pub file_path: String,
    pub num_frames: usize,
    pub num_bodies: usize,
    pub data: Vec<f32>, // Flattened data
}

impl TrajectoryLoader {
    /// Load a .npy file containing trajectory data
    /// Supports both 2D (frames, bodies*18) and 3D (frames, bodies, 18) formats
    pub fn load_trajectory<P: AsRef<Path>>(file_path: P) -> Result<TrajectoryRun> {
        use npyz::NpyFile;
        use std::io::Read;
        
        let path = file_path.as_ref();
        let path_str = path.to_string_lossy().to_string();
        
        let mut file = std::fs::File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let npy = NpyFile::new(&buffer[..])?;
        
        let shape = npy.shape();
        if shape.len() != 2 {
            return Err(anyhow!(
                "Expected a 2D array (frames, data), but got shape: {:?}", 
                shape
            ));
        }
        
        let num_frames = shape[0] as usize;
        let data_per_frame = shape[1] as usize;
        let floats_per_body = 18;
        
        if data_per_frame % floats_per_body != 0 {
            return Err(anyhow!(
                "Frame data size ({}) is not divisible by floats per body ({})", 
                data_per_frame, floats_per_body
            ));
        }
        
        let num_bodies = data_per_frame / floats_per_body;
        let data: Vec<f32> = npy.into_vec()?;
        
        log::info!(
            "Loaded trajectory: {} frames, {} bodies",
            num_frames, num_bodies
        );
        
        Ok(TrajectoryRun {
            file_path: path_str,
            num_frames,
            num_bodies,
            data,
        })
    }
    
    /// Extract bodies for a specific frame
    pub fn get_bodies_at_frame(run: &TrajectoryRun, frame_index: usize) -> Result<Vec<Body>> {
        if frame_index >= run.num_frames {
            return Err(anyhow!(
                "Frame index {} out of bounds (max: {})", 
                frame_index, 
                run.num_frames - 1
            ));
        }
        
        let mut bodies = Vec::with_capacity(run.num_bodies);
        let floats_per_body = 18;
        let data_per_frame = run.num_bodies * floats_per_body;
        
        for body_idx in 0..run.num_bodies {
            let frame_offset = frame_index * data_per_frame;
            let body_offset = body_idx * floats_per_body;
            let offset = frame_offset + body_offset;
            
            let position = [run.data[offset], run.data[offset + 1], run.data[offset + 2], 0.0];
            let velocity = [run.data[offset + 3], run.data[offset + 4], run.data[offset + 5], 0.0];
            let orientation = [
                run.data[offset + 6],
                run.data[offset + 7],
                run.data[offset + 8],
                run.data[offset + 9],
            ];
            let angular_vel = [run.data[offset + 10], run.data[offset + 11], run.data[offset + 12], 0.0];
            let mass = run.data[offset + 13];
            let shape_type = run.data[offset + 14] as u32;
            let shape_params = [run.data[offset + 15], run.data[offset + 16], run.data[offset + 17], 0.0];
            
            bodies.push(Body {
                position,
                velocity,
                orientation,
                angular_vel,
                mass_data: [mass, if mass > 0.0 { 1.0 / mass } else { 0.0 }, 0.0, 0.0],
                shape_data: [shape_type, 0, 0, 0],
                shape_params,
            });
        }
        
        Ok(bodies)
    }
    
    /// Get the duration of the simulation in seconds based on frame count
    /// Assumes 60 FPS for now - this could be made configurable
    pub fn get_duration_seconds(run: &TrajectoryRun) -> f32 {
        run.num_frames as f32 / 60.0
    }
    
    /// Get metadata about the simulation run
    pub fn get_metadata(run: &TrajectoryRun) -> RunMetadata {
        RunMetadata {
            file_path: run.file_path.clone(),
            num_frames: run.num_frames,
            num_bodies: run.num_bodies,
            duration_seconds: TrajectoryLoader::get_duration_seconds(run),
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
    fn test_metadata_generation() {
        let data = vec![0.0; 120 * 10 * 18]; // 120 frames, 10 bodies, 18 floats per body
        let run = TrajectoryRun {
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
    fn test_get_bodies_at_frame_bounds() {
        let data = vec![0.0; 5 * 2 * 18]; // 5 frames, 2 bodies
        let run = TrajectoryRun {
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
}