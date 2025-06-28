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
    
    let temp_dir = create_temporary_directory()?;
    save_frames_as_pngs(frames, &temp_dir, width, height)?;
    encode_pngs_to_video(&temp_dir, &output_path, fps, width, height)?;
    cleanup_temporary_directory(&temp_dir)?;
    
    println!("Video saved to: {}", output_path.display());
    Ok(())
}

fn create_temporary_directory() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let temp_dir = std::env::temp_dir().join(format!("physics_recording_{}", std::process::id()));
    std::fs::create_dir_all(&temp_dir)?;
    Ok(temp_dir)
}

fn save_frames_as_pngs(
    frames: &[Vec<u8>],
    temp_dir: &PathBuf,
    width: u32,
    height: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    for (i, frame_data) in frames.iter().enumerate() {
        let filename = temp_dir.join(format!("frame_{:04}.png", i));
        let rgba_data = convert_bgra_to_rgba(frame_data, width, height);
        save_png(&filename, &rgba_data, width, height)?;
    }
    Ok(())
}

fn convert_bgra_to_rgba(bgra_data: &[u8], width: u32, height: u32) -> Vec<u8> {
    let pixel_count = (width * height) as usize;
    let mut rgba_data = vec![0u8; pixel_count * 4];
    
    for i in 0..pixel_count {
        copy_pixel_bgra_to_rgba(&bgra_data[i * 4..], &mut rgba_data[i * 4..]);
    }
    
    rgba_data
}

fn copy_pixel_bgra_to_rgba(bgra: &[u8], rgba: &mut [u8]) {
    rgba[0] = bgra[2]; // R <- B
    rgba[1] = bgra[1]; // G <- G
    rgba[2] = bgra[0]; // B <- R
    rgba[3] = bgra[3]; // A <- A
}

fn save_png(
    filename: &PathBuf,
    rgba_data: &[u8],
    width: u32,
    height: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let img = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(width, height, rgba_data.to_vec())
        .ok_or("Failed to create image buffer")?;
    img.save(filename)?;
    Ok(())
}

fn encode_pngs_to_video(
    temp_dir: &PathBuf,
    output_path: &PathBuf,
    fps: u32,
    width: u32,
    height: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let status = Command::new("ffmpeg")
        .args(&[
            "-y", // Overwrite output
            "-r", &fps.to_string(),
            "-i", &temp_dir.join("frame_%04d.png").to_string_lossy(),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "28",
            "-pix_fmt", "yuv420p",
            "-vf", &format!("scale={}:{}", width, height),
            &output_path.to_string_lossy(),
        ])
        .status()?;
    
    if !status.success() {
        return Err("FFmpeg encoding failed".into());
    }
    
    Ok(())
}

fn cleanup_temporary_directory(temp_dir: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::remove_dir_all(temp_dir)?;
    Ok(())
}