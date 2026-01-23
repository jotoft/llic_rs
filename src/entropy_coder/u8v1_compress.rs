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
/// Optimized version using pointer reads like C++.
#[inline]
fn estimate_row_bits(deltas: &[u8]) -> u64 {
    let mut bits: u64 = 0;
    let width = deltas.len();
    let num_pairs = width / 2;

    // Cast to u16 pointer like C++
    let ptr = deltas.as_ptr() as *const u16;

    for i in 0..num_pairs {
        // SAFETY: i < num_pairs ensures we're within bounds
        let index = unsafe { *ptr.add(i) as usize };
        let entry = get_compress_table_entry(index);
        bits += (entry & 0x3F) as u64;
    }

    // Handle odd trailing symbol
    if (width & 1) != 0 {
        let index = deltas[width - 1] as usize;
        let entry = get_compress_table_entry(index);
        // Subtract 2 bits (the encoding of the dummy zero)
        bits += ((entry & 0x3F) - 2) as u64;
    }

    bits
}

/// Compute MED residuals for a row.
///
/// Returns residuals computed using MED predictor instead of average predictor.
/// Uses unsafe unchecked access for performance.
#[inline]
fn compute_med_residual_row(current_row: &[u8], prev_row: &[u8], width: usize, out_row: &mut [u8]) {
    if width == 0 {
        return;
    }

    // SAFETY: bounds already verified by caller
    unsafe {
        // First pixel: use only top as predictor
        *out_row.get_unchecked_mut(0) = current_row
            .get_unchecked(0)
            .wrapping_sub(*prev_row.get_unchecked(0));

        // Remaining pixels: use MED predictor
        let mut left = *current_row.get_unchecked(0);
        for x in 1..width {
            let value = *current_row.get_unchecked(x);
            let top = *prev_row.get_unchecked(x);
            let top_left = *prev_row.get_unchecked(x - 1);
            let predictor = med_predictor(left, top, top_left);
            *out_row.get_unchecked_mut(x) = value.wrapping_sub(predictor);
            left = value;
        }
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

    // Process first row (horizontal delta only) - optimized version
    {
        let src_row = &src_image[..width];
        compute_first_row_deltas(src_row, &mut delta_buffer);
    }

    // Process remaining rows using pipelined approach like C++:
    // Encode previous row while computing current row
    for y in 1..height {
        // Encode the previously computed delta row
        encode_row(&delta_buffer, &mut writer);

        let row_offset = y * bytes_per_line;
        let prev_row_offset = (y - 1) * bytes_per_line;
        let src_row = &src_image[row_offset..row_offset + width];
        let prev_row = &src_image[prev_row_offset..prev_row_offset + width];

        // Compute deltas for current row
        compute_avg_predictor_deltas(src_row, prev_row, &mut delta_buffer);
    }

    // Encode the final row
    encode_row(&delta_buffer, &mut writer);

    Ok(writer.finish())
}

/// Compute delta values for the first row (horizontal differences).
/// Uses unsafe unchecked access for performance.
#[inline]
fn compute_first_row_deltas(src_row: &[u8], delta_buffer: &mut [u8]) {
    let width = src_row.len();

    // SAFETY: bounds already verified by caller
    unsafe {
        *delta_buffer.get_unchecked_mut(0) = *src_row.get_unchecked(0);

        // Process in chunks of 4 (matches C++ non-SIMD path)
        let mut x = 1;
        let mut c0 = *src_row.get_unchecked(0);

        while x + 3 < width {
            let c1 = *src_row.get_unchecked(x);
            let c2 = *src_row.get_unchecked(x + 1);
            let c3 = *src_row.get_unchecked(x + 2);
            let c4 = *src_row.get_unchecked(x + 3);
            *delta_buffer.get_unchecked_mut(x) = c1.wrapping_sub(c0);
            *delta_buffer.get_unchecked_mut(x + 1) = c2.wrapping_sub(c1);
            *delta_buffer.get_unchecked_mut(x + 2) = c3.wrapping_sub(c2);
            *delta_buffer.get_unchecked_mut(x + 3) = c4.wrapping_sub(c3);
            c0 = c4;
            x += 4;
        }

        // Handle remaining pixels
        while x < width {
            let c1 = *src_row.get_unchecked(x);
            *delta_buffer.get_unchecked_mut(x) = c1.wrapping_sub(c0);
            c0 = c1;
            x += 1;
        }
    }
}

/// Compute delta values using average predictor (left + top) / 2.
/// Uses unsafe unchecked access for performance.
#[inline]
fn compute_avg_predictor_deltas(src_row: &[u8], prev_row: &[u8], delta_buffer: &mut [u8]) {
    let width = src_row.len();

    // SAFETY: bounds already verified by caller
    unsafe {
        // First pixel uses only top predictor
        *delta_buffer.get_unchecked_mut(0) = src_row
            .get_unchecked(0)
            .wrapping_sub(*prev_row.get_unchecked(0));

        // Process remaining pixels - unrolled for better performance
        let mut x = 1;
        let mut left = *src_row.get_unchecked(0);

        while x + 3 < width {
            // Pixel x
            let top0 = *prev_row.get_unchecked(x);
            let val0 = *src_row.get_unchecked(x);
            let avg0 = ((left as u16 + top0 as u16) >> 1) as u8;
            *delta_buffer.get_unchecked_mut(x) = val0.wrapping_sub(avg0);

            // Pixel x+1
            let top1 = *prev_row.get_unchecked(x + 1);
            let val1 = *src_row.get_unchecked(x + 1);
            let avg1 = ((val0 as u16 + top1 as u16) >> 1) as u8;
            *delta_buffer.get_unchecked_mut(x + 1) = val1.wrapping_sub(avg1);

            // Pixel x+2
            let top2 = *prev_row.get_unchecked(x + 2);
            let val2 = *src_row.get_unchecked(x + 2);
            let avg2 = ((val1 as u16 + top2 as u16) >> 1) as u8;
            *delta_buffer.get_unchecked_mut(x + 2) = val2.wrapping_sub(avg2);

            // Pixel x+3
            let top3 = *prev_row.get_unchecked(x + 3);
            let val3 = *src_row.get_unchecked(x + 3);
            let avg3 = ((val2 as u16 + top3 as u16) >> 1) as u8;
            *delta_buffer.get_unchecked_mut(x + 3) = val3.wrapping_sub(avg3);

            left = val3;
            x += 4;
        }

        // Handle remaining pixels
        while x < width {
            let top = *prev_row.get_unchecked(x);
            let val = *src_row.get_unchecked(x);
            let avg = ((left as u16 + top as u16) >> 1) as u8;
            *delta_buffer.get_unchecked_mut(x) = val.wrapping_sub(avg);
            left = val;
            x += 1;
        }
    }
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
/// Matches C++ optimization: 8x unrolled loop processing 16 symbols per iteration.
#[inline]
fn encode_row(deltas: &[u8], writer: &mut BitWriter) {
    let width = deltas.len();
    let num_pairs = width / 2;

    // Cast to u16 pointer like C++ (safe: deltas is contiguous and we only read within bounds)
    let ptr = deltas.as_ptr() as *const u16;

    let mut i = 0;

    // Unroll by 8 (16 symbols per iteration, matches C++)
    while i + 8 <= num_pairs {
        // SAFETY: i + 8 <= num_pairs ensures we read at most num_pairs u16 values
        // which corresponds to 2*num_pairs bytes, all within the slice
        unsafe {
            let t0 = get_compress_table_entry(*ptr.add(i) as usize);
            writer.write_packed(t0);

            let t1 = get_compress_table_entry(*ptr.add(i + 1) as usize);
            writer.flush_if_needed();
            writer.write_packed(t1);

            let t2 = get_compress_table_entry(*ptr.add(i + 2) as usize);
            writer.flush_if_needed();
            writer.write_packed(t2);

            let t3 = get_compress_table_entry(*ptr.add(i + 3) as usize);
            writer.flush_if_needed();
            writer.write_packed(t3);

            let t4 = get_compress_table_entry(*ptr.add(i + 4) as usize);
            writer.flush_if_needed();
            writer.write_packed(t4);

            let t5 = get_compress_table_entry(*ptr.add(i + 5) as usize);
            writer.flush_if_needed();
            writer.write_packed(t5);

            let t6 = get_compress_table_entry(*ptr.add(i + 6) as usize);
            writer.flush_if_needed();
            writer.write_packed(t6);

            let t7 = get_compress_table_entry(*ptr.add(i + 7) as usize);
            writer.flush_if_needed();
            writer.write_packed(t7);
            writer.flush_if_needed();
        }

        i += 8;
    }

    // Handle remaining pairs
    while i < num_pairs {
        // SAFETY: i < num_pairs ensures we're within bounds
        let idx = unsafe { *ptr.add(i) as usize };
        let entry = get_compress_table_entry(idx);
        writer.write_packed(entry);
        writer.flush_if_needed();
        i += 1;
    }

    // Handle odd trailing symbol
    if (width & 1) != 0 {
        encode_single(deltas[width - 1], writer);
    }
}

/// Encode a single symbol at the end of a row.
///
/// Uses the same trick as C++: look up (sym, 0) pair and subtract 2 bits
/// to remove the trailing zero encoding.
#[inline]
fn encode_single(sym: u8, writer: &mut BitWriter) {
    // Look up symbol paired with 0
    let entry = get_compress_table_entry(sym as usize);
    // Subtract 2 bits (the encoding of the dummy zero)
    let adjusted = entry - 2;
    writer.write_packed(adjusted);
    writer.flush_if_needed();
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
