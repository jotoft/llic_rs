//! Lossless compression using the u8v1 entropy coder.
//!
//! This is the inverse of u8v1_decompress. It takes raw pixel data and produces
//! a compressed bitstream that can be decoded by the decompressor.
//!
//! The encoding process:
//! 1. Apply delta encoding with prediction (same predictor as decoder)
//! 2. Encode delta values using the compression table (pairs of symbols)
//! 3. Pack bits into the output stream
//!
//! Dynamic mode:
//! - For each row (except first), chooses between average and MED predictor
//! - Stores a bitmap indicating which predictor was used for each row
//! - Improves compression for images with edges/sharp transitions

use super::bit_writer::BitWriter;
use super::tables::get_compress_table_entry;

/// MED (Median Edge Detector) predictor used in JPEG-LS and Dynamic mode.
///
/// Returns a prediction based on:
/// - If top_left >= max(left, top): min(left, top)
/// - If top_left <= min(left, top): max(left, top)
/// - Otherwise: left + top - top_left
#[inline]
fn med_predictor(left: u8, top: u8, top_left: u8) -> u8 {
    let a = left as i32;
    let b = top as i32;
    let c = top_left as i32;
    let min_ab = a.min(b);
    let max_ab = a.max(b);

    if c >= max_ab {
        min_ab as u8
    } else if c <= min_ab {
        max_ab as u8
    } else {
        (a + b - c) as u8
    }
}

/// Estimate how many bits a row of delta values would take to compress.
///
/// Uses the compression table to estimate the output size without actually compressing.
#[inline]
fn estimate_row_bits(deltas: &[u8]) -> u64 {
    let mut bits: u64 = 0;
    let width = deltas.len();
    let mut x = 0;

    // Process pairs of symbols
    while x + 1 < width {
        let index = (deltas[x] as usize) | ((deltas[x + 1] as usize) << 8);
        let entry = get_compress_table_entry(index);
        // Extract bits count from packed entry (lower 6 bits)
        bits += (entry & 0x3F) as u64;
        x += 2;
    }

    // Handle odd trailing symbol (use escape code: 13 bits)
    if x < width {
        // For single symbols, estimate using the table with sym2=0, then subtract 2 bits
        let index = deltas[x] as usize;
        let entry = get_compress_table_entry(index);
        let entry_bits = (entry & 0x3F) as u64;
        bits += entry_bits.saturating_sub(2);
    }

    bits
}

/// Compute MED residuals for a row.
///
/// Returns residuals computed using MED predictor instead of average predictor.
fn compute_med_residual_row(current_row: &[u8], prev_row: &[u8], width: usize, out_row: &mut [u8]) {
    if width == 0 {
        return;
    }

    // First pixel: use only top as predictor (same as average for first pixel)
    out_row[0] = current_row[0].wrapping_sub(prev_row[0]);

    // Remaining pixels: use MED predictor
    for x in 1..width {
        let left = current_row[x - 1];
        let top = prev_row[x];
        let top_left = prev_row[x - 1];
        let predictor = med_predictor(left, top, top_left);
        out_row[x] = current_row[x].wrapping_sub(predictor);
    }
}

/// Compresses grayscale image data using the u8v1 entropy coder.
///
/// # Arguments
/// * `src_image` - Source pixel data (row-major, 8-bit grayscale)
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `bytes_per_line` - Stride between rows (usually equal to width)
///
/// # Returns
/// Compressed data as a byte vector, or error if compression fails.
pub fn compress(
    src_image: &[u8],
    width: u32,
    height: u32,
    bytes_per_line: u32,
) -> crate::Result<Vec<u8>> {
    if width == 0 || height == 0 {
        return Err(crate::LlicError::InvalidArgument);
    }

    let width = width as usize;
    let height = height as usize;
    let bytes_per_line = bytes_per_line as usize;

    // Verify source buffer size
    if src_image.len() < (height - 1) * bytes_per_line + width {
        return Err(crate::LlicError::InvalidArgument);
    }

    // Estimate output size (compressed should be smaller, but allocate generously)
    let estimated_size = width * height * 2;
    let mut writer = BitWriter::with_capacity(estimated_size);

    // Buffer for delta-encoded values for current row
    let mut delta_buffer = vec![0u8; width];

    // Process first row (horizontal delta only)
    {
        let src_row = &src_image[..width];
        delta_buffer[0] = src_row[0];
        for x in 1..width {
            delta_buffer[x] = src_row[x].wrapping_sub(src_row[x - 1]);
        }
        encode_row(&delta_buffer, &mut writer);
    }

    // Process remaining rows (combined predictor: average of left and top)
    for y in 1..height {
        let row_offset = y * bytes_per_line;
        let prev_row_offset = (y - 1) * bytes_per_line;
        let src_row = &src_image[row_offset..row_offset + width];
        let prev_row = &src_image[prev_row_offset..prev_row_offset + width];

        // First pixel uses only top predictor
        delta_buffer[0] = src_row[0].wrapping_sub(prev_row[0]);

        // Remaining pixels use average of left and top
        for x in 1..width {
            let left = src_row[x - 1];
            let top = prev_row[x];
            let avg = ((left as u16 + top as u16) >> 1) as u8;
            delta_buffer[x] = src_row[x].wrapping_sub(avg);
        }

        encode_row(&delta_buffer, &mut writer);
    }

    Ok(writer.finish())
}

/// Align a value up to the next multiple of 4.
#[inline]
fn align_to_4(value: usize) -> usize {
    (value + 3) & !3
}

/// Compresses grayscale image data using the u8v1 entropy coder with dynamic predictor.
///
/// Dynamic mode chooses between average and MED predictors on a per-row basis,
/// selecting whichever yields better compression for each row.
///
/// # Arguments
/// * `src_image` - Source pixel data (row-major, 8-bit grayscale)
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `bytes_per_line` - Stride between rows (usually equal to width)
///
/// # Returns
/// Compressed data as a byte vector. Format:
/// - First `align_to_4(height.div_ceil(8))` bytes: predictor bitmap (1 bit per row)
/// - Remaining bytes: compressed bitstream
pub fn compress_dynamic(
    src_image: &[u8],
    width: u32,
    height: u32,
    bytes_per_line: u32,
) -> crate::Result<Vec<u8>> {
    if width == 0 || height == 0 {
        return Err(crate::LlicError::InvalidArgument);
    }

    let width = width as usize;
    let height = height as usize;
    let bytes_per_line = bytes_per_line as usize;

    // Verify source buffer size
    if src_image.len() < (height - 1) * bytes_per_line + width {
        return Err(crate::LlicError::InvalidArgument);
    }

    // Calculate predictor bitmap size (1 bit per row, aligned to 4 bytes)
    let predictor_bytes = height.div_ceil(8);
    let predictor_storage_bytes = align_to_4(predictor_bytes);

    // Allocate output with predictor bitmap at the start
    let estimated_size = predictor_storage_bytes + width * height * 2;
    let mut output = vec![0u8; estimated_size];

    // Predictor bitmap (all zeros initially = all average predictor)
    let predictor_bits = &mut output[..predictor_storage_bytes];

    // Create BitWriter for compressed data after predictor bitmap
    let mut writer = BitWriter::with_capacity(width * height * 2);

    // Buffers for delta-encoded values
    let mut avg_delta_buffer = vec![0u8; width];
    let mut med_delta_buffer = vec![0u8; width];

    // Process first row (horizontal delta only, no predictor choice)
    {
        let src_row = &src_image[..width];
        avg_delta_buffer[0] = src_row[0];
        for x in 1..width {
            avg_delta_buffer[x] = src_row[x].wrapping_sub(src_row[x - 1]);
        }
        encode_row(&avg_delta_buffer, &mut writer);
        // Row 0 always uses "average" (actually horizontal delta), so bit stays 0
    }

    // Process remaining rows with dynamic predictor selection
    for y in 1..height {
        let row_offset = y * bytes_per_line;
        let prev_row_offset = (y - 1) * bytes_per_line;
        let src_row = &src_image[row_offset..row_offset + width];
        let prev_row = &src_image[prev_row_offset..prev_row_offset + width];

        // Compute average predictor residuals
        avg_delta_buffer[0] = src_row[0].wrapping_sub(prev_row[0]);
        for x in 1..width {
            let left = src_row[x - 1];
            let top = prev_row[x];
            let avg = ((left as u16 + top as u16) >> 1) as u8;
            avg_delta_buffer[x] = src_row[x].wrapping_sub(avg);
        }

        // Compute MED predictor residuals
        compute_med_residual_row(src_row, prev_row, width, &mut med_delta_buffer);

        // Estimate bits for each predictor
        let avg_bits = estimate_row_bits(&avg_delta_buffer);
        let med_bits = estimate_row_bits(&med_delta_buffer);

        // Choose the better predictor
        let use_med = med_bits < avg_bits;

        if use_med {
            // Set the predictor bit for this row
            predictor_bits[y / 8] |= 1 << (y % 8);
            encode_row(&med_delta_buffer, &mut writer);
        } else {
            encode_row(&avg_delta_buffer, &mut writer);
        }
    }

    // Finalize the output
    let compressed_data = writer.finish();
    let total_size = predictor_storage_bytes + compressed_data.len();
    output.truncate(total_size);
    output[predictor_storage_bytes..].copy_from_slice(&compressed_data);

    Ok(output)
}

/// Encode a row of delta values to the bit stream.
///
/// Uses the compression table which encodes pairs of symbols for efficiency.
#[inline]
fn encode_row(deltas: &[u8], writer: &mut BitWriter) {
    let width = deltas.len();
    let mut x = 0;

    // Process pairs of symbols
    while x + 1 < width {
        let sym1 = deltas[x];
        let sym2 = deltas[x + 1];
        encode_pair(sym1, sym2, writer);
        x += 2;
    }

    // Handle odd trailing symbol
    if x < width {
        encode_single(deltas[x], writer);
    }
}

/// Encode a pair of symbols using the compression table.
#[inline]
fn encode_pair(sym1: u8, sym2: u8, writer: &mut BitWriter) {
    // Table index: sym1 | (sym2 << 8)
    let index = (sym1 as usize) | ((sym2 as usize) << 8);
    let entry = get_compress_table_entry(index);
    writer.write_packed(entry);
}

/// Encode a single symbol.
///
/// For single symbols, we use the compression table with sym2=0,
/// but we need to handle this carefully to ensure correct decoding.
#[inline]
fn encode_single(sym: u8, writer: &mut BitWriter) {
    // Use 13-bit escape code format for single symbols at end of row
    // Format: 11111 + 8-bit symbol = 13 bits
    // MSB-justified: 0xF800_0000 | (sym << 19)
    let code = 0xF800_0000 | ((sym as u32) << 19);
    writer.write_bits(code, 13);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy_coder::{decompress, decompress_dynamic};

    #[test]
    fn test_compress_zeros_4x4() {
        let input = vec![0u8; 16];
        let compressed = compress(&input, 4, 4, 4).unwrap();

        // Decompress and verify roundtrip
        let mut output = vec![0u8; 16];
        decompress(&compressed, 4, 4, 4, &mut output).unwrap();

        assert_eq!(input, output, "Roundtrip failed for zeros");
    }

    #[test]
    fn test_compress_uniform_128() {
        let input = vec![128u8; 16];
        let compressed = compress(&input, 4, 4, 4).unwrap();

        let mut output = vec![0u8; 16];
        decompress(&compressed, 4, 4, 4, &mut output).unwrap();

        assert_eq!(input, output, "Roundtrip failed for uniform 128");
    }

    #[test]
    fn test_compress_gradient_4x4() {
        // Simple 4x4 gradient: 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
        let input: Vec<u8> = (0..16).collect();
        let compressed = compress(&input, 4, 4, 4).unwrap();

        let mut output = vec![0u8; 16];
        decompress(&compressed, 4, 4, 4, &mut output).unwrap();

        assert_eq!(input, output, "Roundtrip failed for gradient");
    }

    #[test]
    fn test_compress_dynamic_gradient_64x64() {
        // Create a 64x64 gradient image
        let width = 64u32;
        let height = 64u32;
        let mut input = vec![0u8; (width * height) as usize];
        for y in 0..height {
            for x in 0..width {
                input[(y * width + x) as usize] = ((x + y) % 256) as u8;
            }
        }

        let compressed = compress_dynamic(&input, width, height, width).unwrap();

        let mut output = vec![0u8; (width * height) as usize];
        decompress_dynamic(&compressed, width, height, width, &mut output).unwrap();

        assert_eq!(input, output, "Dynamic roundtrip failed for gradient");
    }

    #[test]
    fn test_compress_dynamic_edges() {
        // Create an image with sharp edges (benefits from MED predictor)
        let width = 64u32;
        let height = 64u32;
        let mut input = vec![0u8; (width * height) as usize];

        // Create vertical stripes
        for y in 0..height {
            for x in 0..width {
                input[(y * width + x) as usize] = if x % 8 < 4 { 0 } else { 255 };
            }
        }

        let compressed = compress_dynamic(&input, width, height, width).unwrap();

        let mut output = vec![0u8; (width * height) as usize];
        decompress_dynamic(&compressed, width, height, width, &mut output).unwrap();

        assert_eq!(input, output, "Dynamic roundtrip failed for edges");
    }

    #[test]
    fn test_dynamic_vs_regular_compression() {
        // For smooth gradients, dynamic mode should produce similar or better compression
        let width = 64u32;
        let height = 64u32;
        let mut input = vec![0u8; (width * height) as usize];
        for y in 0..height {
            for x in 0..width {
                input[(y * width + x) as usize] = ((x + y) % 256) as u8;
            }
        }

        let regular = compress(&input, width, height, width).unwrap();
        let dynamic = compress_dynamic(&input, width, height, width).unwrap();

        // Dynamic includes predictor bitmap overhead, so it might be slightly larger
        // for smooth images where MED doesn't help much
        // But both should decompress correctly
        let mut regular_output = vec![0u8; (width * height) as usize];
        let mut dynamic_output = vec![0u8; (width * height) as usize];

        decompress(&regular, width, height, width, &mut regular_output).unwrap();
        decompress_dynamic(&dynamic, width, height, width, &mut dynamic_output).unwrap();

        assert_eq!(input, regular_output, "Regular roundtrip failed");
        assert_eq!(input, dynamic_output, "Dynamic roundtrip failed");

        println!("Regular compression size: {} bytes", regular.len());
        println!(
            "Dynamic compression size: {} bytes (includes {} bytes predictor bitmap)",
            dynamic.len(),
            align_to_4((height as usize).div_ceil(8))
        );
    }
}
