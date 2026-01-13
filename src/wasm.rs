//! WebAssembly bindings for LLIC compression library.
//!
//! LLIC (Lossless/Lossy Image Compression) is optimized for grayscale images.
//! This module provides JavaScript-friendly APIs for use in browsers and Node.js.

use wasm_bindgen::prelude::*;
use crate::{LlicContext, Quality, Mode};

/// Compress grayscale image data using lossless LLIC compression.
///
/// The input must be raw 8-bit grayscale pixels in row-major order.
/// Image dimensions must be multiples of 4.
///
/// @param data - Raw grayscale pixels (Uint8Array, 8-bit, row-major order)
/// @param width - Image width in pixels (must be multiple of 4)
/// @param height - Image height in pixels (must be multiple of 4)
/// @returns Compressed data as Uint8Array
/// @throws Error if dimensions are invalid or not multiples of 4
///
/// @example
/// ```js
/// const compressed = lossless_compress(grayPixels, 256, 256);
/// ```
#[wasm_bindgen]
pub fn lossless_compress(
    data: &[u8],
    width: u32,
    height: u32,
) -> Result<Vec<u8>, JsError> {
    let context = LlicContext::new(width, height, width, Some(1))
        .map_err(|e| JsError::new(&e.to_string()))?;

    let max_size = context.compressed_buffer_size();
    let mut output = vec![0u8; max_size];

    let compressed_size = context.compress_gray8(
        data,
        Quality::Lossless,
        Mode::Default,
        &mut output,
    ).map_err(|e| JsError::new(&e.to_string()))?;

    output.truncate(compressed_size);
    Ok(output)
}

/// Compress grayscale image data using lossy LLIC compression.
///
/// Lossy compression achieves higher compression ratios at the cost of
/// some image quality. The quality parameter controls the trade-off.
///
/// @param data - Raw grayscale pixels (Uint8Array, 8-bit, row-major order)
/// @param width - Image width in pixels (must be multiple of 4)
/// @param height - Image height in pixels (must be multiple of 4)
/// @param quality - Quality level: 'very_high', 'high', 'medium', or 'low'
/// @returns Compressed data as Uint8Array
/// @throws Error if dimensions are invalid or quality is unrecognized
///
/// @example
/// ```js
/// const compressed = lossy_compress(grayPixels, 256, 256, 'high');
/// ```
#[wasm_bindgen]
pub fn lossy_compress(
    _data: &[u8],
    _width: u32,
    _height: u32,
    quality: &str,
) -> Result<Vec<u8>, JsError> {
    // Validate quality parameter
    let _quality = match quality {
        "very_high" => Quality::VeryHigh,
        "high" => Quality::High,
        "medium" => Quality::Medium,
        "low" => Quality::Low,
        _ => return Err(JsError::new(&format!(
            "Invalid quality '{}'. Use: 'very_high', 'high', 'medium', or 'low'",
            quality
        ))),
    };

    Err(JsError::new("Lossy compression not yet implemented in WASM bindings"))
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
