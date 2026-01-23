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
/// Uses SIMD when compiled with target features.
#[inline]
#[cfg(all(target_arch = "x86_64", target_feature = "ssse3"))]
fn compute_first_row_deltas(src_row: &[u8], delta_buffer: &mut [u8]) {
    unsafe {
        compute_first_row_deltas_ssse3(src_row, delta_buffer);
    }
}

#[inline]
#[cfg(not(all(target_arch = "x86_64", target_feature = "ssse3")))]
fn compute_first_row_deltas(src_row: &[u8], delta_buffer: &mut [u8]) {
    compute_first_row_deltas_scalar(src_row, delta_buffer);
}

/// Scalar fallback for first row delta computation.
#[inline]
fn compute_first_row_deltas_scalar(src_row: &[u8], delta_buffer: &mut [u8]) {
    let width = src_row.len();
    if width == 0 {
        return;
    }

    delta_buffer[0] = src_row[0];

    for i in 1..width {
        delta_buffer[i] = src_row[i].wrapping_sub(src_row[i - 1]);
    }
}

/// SSSE3 optimized first row delta computation.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
#[allow(dead_code)]
unsafe fn compute_first_row_deltas_ssse3(src_row: &[u8], delta_buffer: &mut [u8]) {
    use std::arch::x86_64::*;

    let width = src_row.len();
    if width == 0 {
        return;
    }

    *delta_buffer.get_unchecked_mut(0) = *src_row.get_unchecked(0);

    let mut x = 1;

    // Process 16 bytes at a time
    if width >= 17 {
        let mut prev_curr = _mm_set1_epi8(*src_row.get_unchecked(0) as i8);

        while x + 15 < width {
            let curr = _mm_loadu_si128(src_row.as_ptr().add(x) as *const __m128i);
            // Shift to get previous values: alignr(curr, prev, 15) gives [prev[15], curr[0..14]]
            let behind = _mm_alignr_epi8::<15>(curr, prev_curr);
            let delta = _mm_sub_epi8(curr, behind);
            _mm_storeu_si128(delta_buffer.as_mut_ptr().add(x) as *mut __m128i, delta);
            prev_curr = curr;
            x += 16;
        }
    }

    // Handle remaining pixels
    let mut c0 = if x > 1 {
        *src_row.get_unchecked(x - 1)
    } else {
        *src_row.get_unchecked(0)
    };

    while x < width {
        let c1 = *src_row.get_unchecked(x);
        *delta_buffer.get_unchecked_mut(x) = c1.wrapping_sub(c0);
        c0 = c1;
        x += 1;
    }
}

/// Compute delta values using average predictor (left + top) / 2.
/// Uses SIMD when compiled with target features.
#[inline]
#[cfg(all(target_arch = "x86_64", target_feature = "ssse3"))]
fn compute_avg_predictor_deltas(src_row: &[u8], prev_row: &[u8], delta_buffer: &mut [u8]) {
    unsafe {
        compute_avg_predictor_deltas_ssse3(src_row, prev_row, delta_buffer);
    }
}

#[inline]
#[cfg(not(all(target_arch = "x86_64", target_feature = "ssse3")))]
fn compute_avg_predictor_deltas(src_row: &[u8], prev_row: &[u8], delta_buffer: &mut [u8]) {
    compute_avg_predictor_deltas_scalar(src_row, prev_row, delta_buffer);
}

/// Scalar fallback for average predictor computation.
#[inline]
fn compute_avg_predictor_deltas_scalar(src_row: &[u8], prev_row: &[u8], delta_buffer: &mut [u8]) {
    let width = src_row.len();
    if width == 0 {
        return;
    }

    // First pixel uses only top predictor
    delta_buffer[0] = src_row[0].wrapping_sub(prev_row[0]);

    // Remaining pixels use average of left and top
    let mut left = src_row[0];
    for x in 1..width {
        let top = prev_row[x];
        let val = src_row[x];
        let avg = ((left as u16 + top as u16) >> 1) as u8;
        delta_buffer[x] = val.wrapping_sub(avg);
        left = val;
    }
}

/// SSSE3 optimized average predictor computation.
/// Processes 16 bytes at a time using SIMD instructions.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
#[allow(dead_code)]
unsafe fn compute_avg_predictor_deltas_ssse3(
    src_row: &[u8],
    prev_row: &[u8],
    delta_buffer: &mut [u8],
) {
    use std::arch::x86_64::*;

    let width = src_row.len();
    if width == 0 {
        return;
    }

    // First pixel uses only top predictor
    *delta_buffer.get_unchecked_mut(0) = src_row
        .get_unchecked(0)
        .wrapping_sub(*prev_row.get_unchecked(0));

    let mut x = 1;

    // Process 16 bytes at a time with SIMD
    if width >= 17 {
        // Initialize "previous current" vector for shifted alignment
        let mut prev_curr = _mm_set1_epi8(*src_row.get_unchecked(0) as i8);

        while x + 15 < width {
            // Load current row and top row (16 bytes each)
            let curr_vec = _mm_loadu_si128(src_row.as_ptr().add(x) as *const __m128i);
            let top_vec = _mm_loadu_si128(prev_row.as_ptr().add(x) as *const __m128i);

            // Create "left" vector by shifting: take last byte of prev_curr, first 15 of curr
            // alignr(a, b, n) = (a:b) >> (n*8), so alignr(curr, prev, 15) gives us shifted left
            let left_vec = _mm_alignr_epi8::<15>(curr_vec, prev_curr);

            // Compute average: (left + top) / 2
            // Use the same trick as C++: avg_epu8 does (a+b+1)/2, we need to compensate
            let c1 = _mm_set1_epi8(1);
            let avg = _mm_avg_epu8(left_vec, top_vec); // (left + top + 1) / 2
            let xor = _mm_xor_si128(left_vec, top_vec); // For rounding compensation
            let and = _mm_and_si128(xor, c1);
            let avg_correct = _mm_sub_epi8(avg, and); // Correct to floor division

            // Compute residual: curr - avg
            let residual = _mm_sub_epi8(curr_vec, avg_correct);

            // Store result
            _mm_storeu_si128(delta_buffer.as_mut_ptr().add(x) as *mut __m128i, residual);

            // Save for next iteration
            prev_curr = curr_vec;
            x += 16;
        }
    }

    // Handle remaining pixels with scalar code
    let mut left = if x > 1 {
        *src_row.get_unchecked(x - 1)
    } else {
        *src_row.get_unchecked(0)
    };

    while x < width {
        let top = *prev_row.get_unchecked(x);
        let val = *src_row.get_unchecked(x);
        let avg = ((left as u16 + top as u16) >> 1) as u8;
        *delta_buffer.get_unchecked_mut(x) = val.wrapping_sub(avg);
        left = val;
        x += 1;
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
