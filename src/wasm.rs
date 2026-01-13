//! WebAssembly bindings for LLIC compression library.
//!
//! This module provides JavaScript-friendly APIs for compressing and
//! decompressing grayscale images using the LLIC format.

use wasm_bindgen::prelude::*;
use crate::{LlicContext, Quality, Mode};

/// Decompress LLIC-compressed grayscale image data.
///
/// # Arguments
/// * `compressed_data` - The LLIC compressed data (v3 format)
/// * `width` - Image width (must be multiple of 4)
/// * `height` - Image height (must be multiple of 4)
///
/// # Returns
/// Decompressed grayscale pixel data as Uint8Array
#[wasm_bindgen]
pub fn decompress_gray8(
    compressed_data: &[u8],
    width: u32,
    height: u32,
) -> Result<Vec<u8>, JsError> {
    let context = LlicContext::new(width, height, width, Some(1))
        .map_err(|e| JsError::new(&e.to_string()))?;

    let mut output = vec![0u8; (width * height) as usize];
    context.decompress_gray8(compressed_data, &mut output)
        .map_err(|e| JsError::new(&e.to_string()))?;

    Ok(output)
}

/// Compress grayscale image data using LLIC lossless compression.
///
/// # Arguments
/// * `image_data` - Raw grayscale pixel data (row-major, 8-bit)
/// * `width` - Image width (must be multiple of 4)
/// * `height` - Image height (must be multiple of 4)
///
/// # Returns
/// LLIC compressed data as Uint8Array
#[wasm_bindgen]
pub fn compress_gray8(
    image_data: &[u8],
    width: u32,
    height: u32,
) -> Result<Vec<u8>, JsError> {
    let context = LlicContext::new(width, height, width, Some(1))
        .map_err(|e| JsError::new(&e.to_string()))?;

    let max_size = context.compressed_buffer_size();
    let mut output = vec![0u8; max_size];

    let compressed_size = context.compress_gray8(
        image_data,
        Quality::Lossless,
        Mode::Default,
        &mut output,
    ).map_err(|e| JsError::new(&e.to_string()))?;

    output.truncate(compressed_size);
    Ok(output)
}

/// Get the required buffer size for compressed output.
///
/// # Arguments
/// * `width` - Image width
/// * `height` - Image height
///
/// # Returns
/// Maximum possible compressed size in bytes
#[wasm_bindgen]
pub fn compressed_buffer_size(width: u32, height: u32) -> Result<usize, JsError> {
    let context = LlicContext::new(width, height, width, Some(1))
        .map_err(|e| JsError::new(&e.to_string()))?;
    Ok(context.compressed_buffer_size())
}

/// Get library version information.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Check if SIMD is enabled in this build.
#[wasm_bindgen]
pub fn has_simd() -> bool {
    cfg!(target_feature = "simd128")
}

/// Get build info string.
#[wasm_bindgen]
pub fn build_info() -> String {
    let simd = if cfg!(target_feature = "simd128") { "SIMD" } else { "scalar" };
    let profile = if cfg!(debug_assertions) { "debug" } else { "release" };
    format!("{} ({}, {})", env!("CARGO_PKG_VERSION"), simd, profile)
}
