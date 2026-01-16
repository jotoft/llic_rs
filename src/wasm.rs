//! WebAssembly bindings for LLIC compression library.
//!
//! LLIC (Low Latency Image Codec) is optimized for grayscale images.
//! This module provides JavaScript-friendly APIs for use in browsers and Node.js.

use crate::{LlicContext, Mode, Quality};
use wasm_bindgen::prelude::*;

/// Compress grayscale image data using LLIC compression.
///
/// The input must be raw 8-bit grayscale pixels in row-major order.
/// Image dimensions must be multiples of 4.
///
/// @param data - Raw grayscale pixels (Uint8Array, 8-bit, row-major order)
/// @param width - Image width in pixels (must be multiple of 4)
/// @param height - Image height in pixels (must be multiple of 4)
/// @param quality - Quality level: 'lossless', 'lossless_entropy', 'very_high', 'high', 'medium', 'low', or 'very_low'
/// @param mode - Optional compression mode: 'default', 'fast', or 'dynamic' (defaults to 'default' for lossless, 'fast' for lossy)
/// @returns Compressed data as Uint8Array
/// @throws Error if dimensions are invalid or quality/mode is unrecognized
///
/// Quality levels:
/// - 'lossless': Tile-based lossless compression (recommended, format v4)
/// - 'lossless_entropy': Legacy entropy-coded lossless (for compatibility)
/// - 'very_high': Near-lossless, max error ±2
/// - 'high': High quality, max error ±4
/// - 'medium': Medium quality, max error ±8
/// - 'low': Low quality, max error ±16
/// - 'very_low': Very low quality, max error ±32
///
/// Compression modes:
/// - 'default': Balance between speed and size (compresses tile headers)
/// - 'fast': Fastest compression/decompression (skips header compression, slightly larger output)
/// - 'dynamic': Best compression ratio (adaptive predictor, compresses headers, slower)
///
/// @example
/// ```js
/// const lossless = compress(grayPixels, 256, 256, 'lossless');
/// const fast = compress(grayPixels, 256, 256, 'high', 'fast');
/// const best = compress(grayPixels, 256, 256, 'lossless', 'dynamic');
/// ```
#[wasm_bindgen]
pub fn compress(
    data: &[u8],
    width: u32,
    height: u32,
    quality: &str,
    mode: Option<String>,
) -> Result<Vec<u8>, JsError> {
    let quality_level = match quality {
        "lossless" => Quality::Lossless,
        "lossless_entropy" => Quality::LosslessEntropy,
        "very_high" => Quality::VeryHigh,
        "high" => Quality::High,
        "medium" => Quality::Medium,
        "low" => Quality::Low,
        "very_low" => Quality::VeryLow,
        _ => return Err(JsError::new(&format!(
            "Invalid quality '{}'. Use: 'lossless', 'lossless_entropy', 'very_high', 'high', 'medium', 'low', or 'very_low'",
            quality
        ))),
    };

    // Parse mode if provided, otherwise use smart defaults
    let compression_mode = if let Some(mode_str) = mode {
        match mode_str.as_str() {
            "default" => Mode::Default,
            "fast" => Mode::Fast,
            "dynamic" => Mode::Dynamic,
            _ => {
                return Err(JsError::new(&format!(
                    "Invalid mode '{}'. Use: 'default', 'fast', or 'dynamic'",
                    mode_str
                )))
            }
        }
    } else {
        // Default mode selection: Default for lossless, Fast for lossy
        if quality_level == Quality::Lossless || quality_level == Quality::LosslessEntropy {
            Mode::Default
        } else {
            Mode::Fast
        }
    };

    let context = LlicContext::new(width, height, width, Some(1))
        .map_err(|e| JsError::new(&e.to_string()))?;

    let max_size = context.compressed_buffer_size();
    let mut output = vec![0u8; max_size];

    let compressed_size = context
        .compress_gray8(data, quality_level, compression_mode, &mut output)
        .map_err(|e| JsError::new(&e.to_string()))?;

    output.truncate(compressed_size);
    Ok(output)
}

// Keep legacy functions for backwards compatibility

/// @deprecated Use `compress(data, width, height, 'lossless')` instead
#[wasm_bindgen]
pub fn lossless_compress(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, JsError> {
    compress(data, width, height, "lossless", None)
}

/// @deprecated Use `compress(data, width, height, quality)` instead
#[wasm_bindgen]
pub fn lossy_compress(
    data: &[u8],
    width: u32,
    height: u32,
    quality: &str,
) -> Result<Vec<u8>, JsError> {
    compress(data, width, height, quality, None)
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
pub fn decompress(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, JsError> {
    let context = LlicContext::new(width, height, width, Some(1))
        .map_err(|e| JsError::new(&e.to_string()))?;

    let mut output = vec![0u8; (width * height) as usize];
    context
        .decompress_gray8(data, &mut output)
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
    let simd = if cfg!(target_feature = "simd128") {
        "SIMD"
    } else {
        "scalar"
    };
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    format!("{} ({}, {})", env!("CARGO_PKG_VERSION"), simd, profile)
}

/// Extract tile metadata from compressed data for visualization.
///
/// Returns a flat array of [min, dist, bits] for each 4x4 tile.
/// The array length is `(width/4) * (height/4) * 3`.
///
/// This is useful for visualizing the tile-based compression:
/// - `min`: Minimum pixel value in the 4x4 tile
/// - `dist`: Range (max - min) of pixel values in the tile
/// - `bits`: Number of bits used per pixel in this tile (0-8)
///
/// @param compressed_data - LLIC compressed data (must be tile-based, not entropy-coded)
/// @param width - Image width in pixels (must be multiple of 4)
/// @param height - Image height in pixels (must be multiple of 4)
/// @returns Flat Uint8Array: [min0, dist0, bits0, min1, dist1, bits1, ...]
/// @throws Error if data is not tile-based compressed or is invalid
///
/// @example
/// ```js
/// const metadata = get_tile_metadata(compressed, 256, 256);
/// // metadata.length == (256/4) * (256/4) * 3 == 12288
/// // Tile at (tx, ty): idx = (ty * (width/4) + tx) * 3
/// // min = metadata[idx], dist = metadata[idx+1], bits = metadata[idx+2]
/// ```
#[wasm_bindgen]
pub fn get_tile_metadata(
    compressed_data: &[u8],
    width: u32,
    height: u32,
) -> Result<Vec<u8>, JsError> {
    if compressed_data.is_empty() {
        return Err(JsError::new("Empty compressed data"));
    }

    // Parse format header to find the tile block data
    let version = compressed_data[0];
    if version < 4 {
        return Err(JsError::new(
            "Tile metadata only available for format v4+ (tile-based compression)",
        ));
    }

    let num_blocks = compressed_data[1] as usize;
    let tile_based = compressed_data[2] != 0;
    let mode = compressed_data[3];
    let use_dynamic_predictor = mode == 2; // Mode::Dynamic = 2

    if !tile_based {
        return Err(JsError::new(
            "Tile metadata not available for entropy-coded compression",
        ));
    }

    // Parse block sizes (format v4: block sizes start at offset 4)
    let mut block_sizes = Vec::with_capacity(num_blocks);
    let mut pos = 4usize;
    for _ in 0..num_blocks {
        if pos + 4 > compressed_data.len() {
            return Err(JsError::new(
                "Invalid compressed data: truncated block sizes",
            ));
        }
        let size = u32::from_le_bytes([
            compressed_data[pos],
            compressed_data[pos + 1],
            compressed_data[pos + 2],
            compressed_data[pos + 3],
        ]) as usize;
        block_sizes.push(size);
        pos += 4;
    }

    // For single-block data, extract metadata directly
    // For multi-block data, we need to combine all blocks
    // For simplicity, we'll handle the common single-block case and
    // use the last non-zero block for multi-block (which has the full data)
    let block_data_start = pos;

    // Find the block with actual data (for multi-thread output, it's usually the last block)
    let mut data_offset = block_data_start;
    let mut tile_block_data = &compressed_data[0..0];

    for (i, &size) in block_sizes.iter().enumerate() {
        if size > 0 {
            // For single block case or the main data block
            if num_blocks == 1 || i == block_sizes.len() - 1 {
                tile_block_data = &compressed_data[data_offset..data_offset + size];
            }
        }
        data_offset += size;
    }

    if tile_block_data.is_empty() {
        return Err(JsError::new("No tile block data found"));
    }

    // Extract metadata using the lossy module
    let metadata =
        crate::lossy::extract_tile_metadata(tile_block_data, width, height, use_dynamic_predictor)
            .map_err(|e| JsError::new(&format!("Failed to extract tile metadata: {}", e)))?;

    // Flatten to [min, dist, bits, min, dist, bits, ...]
    let mut result = Vec::with_capacity(metadata.len() * 3);
    for (min, dist, bits) in metadata {
        result.push(min);
        result.push(dist);
        result.push(bits);
    }

    Ok(result)
}

/// Compute prediction residual for grayscale image data.
///
/// Returns the difference between actual pixel values and predicted values,
/// mapped to 0-255 where 128 = zero residual (perfect prediction).
///
/// This is useful for visualizing where the compression predictor struggles:
/// - Values near 128 (white in visualization) = good prediction
/// - Values far from 128 (blue/red) = high prediction error
///
/// @param data - Raw grayscale pixels (Uint8Array, 8-bit, row-major order)
/// @param width - Image width in pixels
/// @param height - Image height in pixels
/// @returns Residual data as Uint8Array (128 = zero, <128 = negative, >128 = positive)
///
/// @example
/// ```js
/// const residual = get_prediction_residual(grayPixels, 256, 256);
/// // residual[i] == 128 means perfect prediction at pixel i
/// ```
#[wasm_bindgen]
pub fn get_prediction_residual(data: &[u8], width: u32, height: u32) -> Vec<u8> {
    crate::compute_prediction_residual(data, width as usize, height as usize)
}
