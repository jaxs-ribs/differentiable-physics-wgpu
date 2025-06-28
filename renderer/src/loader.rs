//! NPY file loading for physics simulation trajectories.
//!
//! This module handles loading and parsing of NumPy array files containing
//! physics simulation data. It converts the flat array format into structured
//! `Body` instances for rendering.
//!
//! # File Format
//! The NPY files are expected to contain 2D arrays with shape (frames, bodies*18)
//! where each body is represented by 18 float values:
//! - Position (3 floats)
//! - Velocity (3 floats)
//! - Orientation quaternion (4 floats)
//! - Angular velocity (3 floats)
//! - Mass (1 float)
//! - Shape type (1 float: 0=sphere, 2=box)
//! - Shape parameters (3 floats: radius or half-extents)
//!
//! # Design
//! The loader is implemented as a stateless utility struct following the
//! static method pattern. This ensures thread safety and prevents unnecessary
//! state management for what is essentially a data transformation operation.

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
        validate_frame_index(frame_index, run.num_frames)?;
        
        let mut bodies = Vec::with_capacity(run.num_bodies);
        let floats_per_body = 18;
        let frame_offset = calculate_frame_offset(frame_index, run.num_bodies, floats_per_body);
        
        for body_idx in 0..run.num_bodies {
            let body_offset = body_idx * floats_per_body;
            let offset = frame_offset + body_offset;
            let body = parse_body_at_offset(&run.data, offset);
            bodies.push(body);
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

fn validate_frame_index(frame_index: usize, num_frames: usize) -> Result<()> {
    if frame_index >= num_frames {
        return Err(anyhow!(
            "Frame index {} out of bounds (max: {})", 
            frame_index, 
            num_frames - 1
        ));
    }
    Ok(())
}

fn calculate_frame_offset(frame_index: usize, num_bodies: usize, floats_per_body: usize) -> usize {
    frame_index * num_bodies * floats_per_body
}

fn parse_body_at_offset(data: &[f32], offset: usize) -> Body {
    let position = extract_vec3(data, offset);
    let velocity = extract_vec3(data, offset + 3);
    let orientation = extract_vec4(data, offset + 6);
    let angular_vel = extract_vec3(data, offset + 10);
    let mass = data[offset + 13];
    let shape_type = data[offset + 14] as u32;
    let shape_params = extract_vec3(data, offset + 15);
    
    Body {
        position,
        velocity,
        orientation,
        angular_vel,
        mass_data: compute_mass_data(mass),
        shape_data: [shape_type, 0, 0, 0],
        shape_params,
    }
}

fn extract_vec3(data: &[f32], offset: usize) -> [f32; 4] {
    [data[offset], data[offset + 1], data[offset + 2], 0.0]
}

fn extract_vec4(data: &[f32], offset: usize) -> [f32; 4] {
    [data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]
}

fn compute_mass_data(mass: f32) -> [f32; 4] {
    let inverse_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
    [mass, inverse_mass, 0.0, 0.0]
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