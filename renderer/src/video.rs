//! Video recording and export functionality.
//!
//! This module provides video recording capabilities for the renderer, allowing
//! simulation playback to be exported as MP4 files. It handles frame capture,
//! color space conversion, and FFmpeg integration.
//!
//! # Process
//! 1. Capture frames from the renderer (BGRA format)
//! 2. Convert BGRA to RGBA for image encoding
//! 3. Save frames as temporary PNG files
//! 4. Use FFmpeg to encode PNGs into MP4
//! 5. Clean up temporary files
//!
//! # Requirements
//! - FFmpeg must be installed and available in PATH
//! - Sufficient disk space for temporary PNG files
//!
//! # Design Notes
//! The module uses a two-stage process (PNG then MP4) for compatibility
//! and quality. Direct piping to FFmpeg is possible but more complex
//! and error-prone across different platforms.

use std::{path::PathBuf, process::Command};
use image::{ImageBuffer, Rgba};

pub fn save_frames_as_video(
    frames: &[Vec<u8>],
    output_path: PathBuf,
    fps: u32,
    width: u32,
    height: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Saving {} frames to video...", frames.len());
    
    // Create temp directory
    let temp_dir = std::env::temp_dir().join(format!("physics_recording_{}", std::process::id()));
    std::fs::create_dir_all(&temp_dir)?;
    
    // Save frames as PNG files
    for (i, frame_data) in frames.iter().enumerate() {
        let filename = temp_dir.join(format!("frame_{:04}.png", i));
        
        // Convert BGRA to RGBA
        let mut rgba_data = vec![0u8; (width * height * 4) as usize];
        for j in 0..((width * height) as usize) {
            rgba_data[j * 4] = frame_data[j * 4 + 2];     // R
            rgba_data[j * 4 + 1] = frame_data[j * 4 + 1]; // G
            rgba_data[j * 4 + 2] = frame_data[j * 4];     // B
            rgba_data[j * 4 + 3] = frame_data[j * 4 + 3]; // A
        }
        
        let img = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(width, height, rgba_data)
            .ok_or("Failed to create image buffer")?;
        img.save(&filename)?;
    }
    
    // Run ffmpeg to create video
    let status = Command::new("ffmpeg")
        .args(&[
            "-y", // Overwrite output
            "-r", &fps.to_string(),
            "-i", &temp_dir.join("frame_%04d.png").to_string_lossy(),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "28",
            "-pix_fmt", "yuv420p",
            "-vf", "scale=800:600", // Ensure output size
            &output_path.to_string_lossy(),
        ])
        .status()?;
    
    if !status.success() {
        return Err("FFmpeg encoding failed".into());
    }
    
    // Clean up
    std::fs::remove_dir_all(&temp_dir)?;
    println!("Video saved to: {}", output_path.display());
    
    Ok(())
}