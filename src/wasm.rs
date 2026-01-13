//! WebAssembly bindings for LLIC compression library.
//!
//! LLIC (Lossless/Lossy Image Compression) is optimized for grayscale images.
//! This module provides JavaScript-friendly APIs for use in browsers and Node.js.

use wasm_bindgen::prelude::*;
use crate::{LlicContext, Quality, Mode};

/// Compress grayscale image data using LLIC compression.
///
/// The input must be raw 8-bit grayscale pixels in row-major order.
/// Image dimensions must be multiples of 4.
///
/// @param data - Raw grayscale pixels (Uint8Array, 8-bit, row-major order)
/// @param width - Image width in pixels (must be multiple of 4)
/// @param height - Image height in pixels (must be multiple of 4)
/// @param quality - Quality level: 'lossless', 'very_high', 'high', 'medium', or 'low'
/// @returns Compressed data as Uint8Array
/// @throws Error if dimensions are invalid or quality is unrecognized
///
/// @example
/// ```js
/// const lossless = compress(grayPixels, 256, 256, 'lossless');
/// const lossy = compress(grayPixels, 256, 256, 'high');
/// ```
#[wasm_bindgen]
pub fn compress(
    data: &[u8],
    width: u32,
    height: u32,
    quality: &str,
) -> Result<Vec<u8>, JsError> {
    let quality_level = match quality {
        "lossless" => Quality::Lossless,
        "very_high" => Quality::VeryHigh,
        "high" => Quality::High,
        "medium" => Quality::Medium,
        "low" => Quality::Low,
        _ => return Err(JsError::new(&format!(
            "Invalid quality '{}'. Use: 'lossless', 'very_high', 'high', 'medium', or 'low'",
            quality
        ))),
    };

    let context = LlicContext::new(width, height, width, Some(1))
        .map_err(|e| JsError::new(&e.to_string()))?;

    let max_size = context.compressed_buffer_size();
    let mut output = vec![0u8; max_size];

    // Use Default mode for lossless, Fast mode for lossy
    let mode = if quality_level == Quality::Lossless { Mode::Default } else { Mode::Fast };

    let compressed_size = context.compress_gray8(
        data,
        quality_level,
        mode,
        &mut output,
    ).map_err(|e| JsError::new(&e.to_string()))?;

    output.truncate(compressed_size);
    Ok(output)
}

// Keep legacy functions for backwards compatibility

/// @deprecated Use `compress(data, width, height, 'lossless')` instead
#[wasm_bindgen]
pub fn lossless_compress(
    data: &[u8],
    width: u32,
    height: u32,
) -> Result<Vec<u8>, JsError> {
    compress(data, width, height, "lossless")
}

/// @deprecated Use `compress(data, width, height, quality)` instead
#[wasm_bindgen]
pub fn lossy_compress(
    data: &[u8],
    width: u32,
    height: u32,
    quality: &str,
) -> Result<Vec<u8>, JsError> {
    compress(data, width, height, quality)
}

/// Decompress LLIC-compressed grayscale image data.
///
/// Automatically detects whether the data is lossless or lossy compressed.
/// Image dimensions must match the original compressed image.
///
/// @param data - LLIC compressed data (Uint8Array)
/// @param width - Original image width in pixels (must be multiple of 4)
/// @param height - Original image height in pixels (must be multiple of 4)
/// @returns Decompressed grayscale pixels as Uint8Array
/// @throws Error if data is corrupted or dimensions don't match
///
/// @example
/// ```js
/// const pixels = decompress(compressedData, 256, 256);
/// ```
#[wasm_bindgen]
pub fn decompress(
    data: &[u8],
    width: u32,
    height: u32,
) -> Result<Vec<u8>, JsError> {
    let context = LlicContext::new(width, height, width, Some(1))
        .map_err(|e| JsError::new(&e.to_string()))?;

    let mut output = vec![0u8; (width * height) as usize];
    context.decompress_gray8(data, &mut output)
        .map_err(|e| JsError::new(&e.to_string()))?;

    Ok(output)
}

/// Get the library version string.
///
/// @returns Version string (e.g., "0.2.0")
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Check if SIMD optimizations are enabled in this build.
///
/// SIMD builds are significantly faster (up to 4x) but require
/// browser support for WebAssembly SIMD (Chrome 91+, Firefox 89+, Safari 16.4+).
///
/// @returns true if SIMD is enabled, false otherwise
#[wasm_bindgen]
pub fn has_simd() -> bool {
    cfg!(target_feature = "simd128")
}

/// Get detailed build information.
///
/// Returns version, SIMD status, and build profile.
///
/// @returns Build info string (e.g., "0.2.0 (SIMD, release)")
#[wasm_bindgen]
pub fn build_info() -> String {
    let simd = if cfg!(target_feature = "simd128") { "SIMD" } else { "scalar" };
    let profile = if cfg!(debug_assertions) { "debug" } else { "release" };
    format!("{} ({}, {})", env!("CARGO_PKG_VERSION"), simd, profile)
}
