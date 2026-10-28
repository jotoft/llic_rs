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

use std::mem::MaybeUninit;

use super::tables::get_compress_table_entry;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod residuals_simd128;

/// MED (Median Edge Detector) predictor used in JPEG-LS and Dynamic mode.
///
/// Returns a prediction based on:
/// - If top_left >= max(left, top): min(left, top)
/// - If top_left <= min(left, top): max(left, top)
/// - Otherwise: left + top - top_left
#[inline(always)]
fn med_predictor(left: u8, top: u8, top_left: u8) -> u8 {
    // left + top - clamp(top_left, min, max) gives all three cases without
    // branches, which mispredict on noisy rows (the SSE code computes the same).
    let (min_ab, max_ab) = (left.min(top), left.max(top));
    left.wrapping_add(top)
        .wrapping_sub(top_left.max(min_ab).min(max_ab))
}

/// Code length of a residual pair `(s0, s1)` plus the lengths' sum in the low
/// six bits, from the pair table (codes are left-justified above bit 6).
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
#[inline(always)]
fn pair_entry(pair: &[u8]) -> u32 {
    get_compress_table_entry(usize::from(u16::from_le_bytes([pair[0], pair[1]])))
}

/// Table entry for a single trailing symbol: the pair `(sym, 0)` minus the
/// 2-bit code of the dummy zero (the length is at least 4, so this cannot
/// borrow from the code bits).
#[inline(always)]
fn single_entry(sym: u8) -> u32 {
    get_compress_table_entry(usize::from(sym)) - 2
}

/// Estimate how many bits a row of residuals takes to code.
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
#[inline]
fn estimate_row_bits(residuals: &[u8]) -> u64 {
    let pairs = residuals.chunks_exact(2);
    let tail = pairs
        .remainder()
        .first()
        .map_or(0, |&sym| single_entry(sym) & 0x3F);
    pairs
        .map(|pair| u64::from(pair_entry(pair) & 0x3F))
        .sum::<u64>()
        + u64::from(tail)
}

/// Compute MED residuals for a row (the first pixel predicts from the top).
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
#[inline]
fn compute_med_residual_row(current_row: &[u8], prev_row: &[u8], out_row: &mut [u8]) {
    let width = current_row.len();
    let (prev_row, out_row) = (&prev_row[..width], &mut out_row[..width]);
    out_row[0] = current_row[0].wrapping_sub(prev_row[0]);
    for x in 1..width {
        let predictor = med_predictor(current_row[x - 1], prev_row[x], prev_row[x - 1]);
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
    encode_vec(src_image, width, height, bytes_per_line, false)
}

/// Like [`compress`], writing into `dst`, which needs
/// [`max_compressed_size`]`(width, height, false)` bytes. Returns the bytes
/// written.
pub fn compress_into(
    src_image: &[u8],
    width: u32,
    height: u32,
    bytes_per_line: u32,
    dst: &mut [u8],
) -> crate::Result<usize> {
    encode(
        src_image,
        width,
        height,
        bytes_per_line,
        false,
        as_uninit(dst),
    )
}

/// Upper bound on the size of a [`compress`] (or, with `dynamic`,
/// [`compress_dynamic`]) stream: 13 bits per pixel, the predictor bitmap, and
/// 4 bytes of slack for the speculative word store of the branch-free flush.
pub fn max_compressed_size(width: u32, height: u32, dynamic: bool) -> usize {
    let (width, height) = (width as usize, height as usize);
    let predictor_bytes = if dynamic {
        align_to_4(height.div_ceil(8))
    } else {
        0
    };
    predictor_bytes + (13 * width * height).div_ceil(32) * 4 + 4
}

/// Views an initialised buffer as possibly uninitialised for the encoder.
fn as_uninit(bytes: &mut [u8]) -> &mut [MaybeUninit<u8>] {
    // SAFETY: same layout; the encoder only ever writes initialised bytes, so
    // `bytes` stays initialised.
    unsafe { &mut *(std::ptr::from_mut::<[u8]>(bytes) as *mut [MaybeUninit<u8>]) }
}

/// Encodes into a new vector, writing its capacity without zeroing it first.
fn encode_vec(
    src_image: &[u8],
    width: u32,
    height: u32,
    bytes_per_line: u32,
    dynamic: bool,
) -> crate::Result<Vec<u8>> {
    let mut out = Vec::with_capacity(max_compressed_size(width, height, dynamic));
    let size = encode(
        src_image,
        width,
        height,
        bytes_per_line,
        dynamic,
        out.spare_capacity_mut(),
    )?;
    // SAFETY: the encoder wrote every byte below `size`.
    unsafe { out.set_len(size) };
    Ok(out)
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
#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "ssse3"),
    all(target_arch = "wasm32", target_feature = "simd128")
)))]
fn compute_first_row_deltas(src_row: &[u8], delta_buffer: &mut [u8]) {
    compute_first_row_deltas_scalar(src_row, delta_buffer);
}

/// Scalar fallback for first row delta computation.
#[inline]
#[allow(dead_code)] // Used only when SSSE3 is not available
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
#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "ssse3"),
    all(target_arch = "wasm32", target_feature = "simd128")
)))]
fn compute_avg_predictor_deltas(src_row: &[u8], prev_row: &[u8], delta_buffer: &mut [u8]) {
    compute_avg_predictor_deltas_scalar(src_row, prev_row, delta_buffer);
}

/// Scalar fallback for average predictor computation.
#[inline]
#[allow(dead_code)] // Used only when SSSE3 is not available
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
    encode_vec(src_image, width, height, bytes_per_line, true)
}

/// Like [`compress_dynamic`], writing into `dst`, which needs
/// [`max_compressed_size`]`(width, height, true)` bytes. Returns the bytes
/// written.
pub fn compress_dynamic_into(
    src_image: &[u8],
    width: u32,
    height: u32,
    bytes_per_line: u32,
    dst: &mut [u8],
) -> crate::Result<usize> {
    encode(
        src_image,
        width,
        height,
        bytes_per_line,
        true,
        as_uninit(dst),
    )
}

/// Checks the arguments and runs the image encoder, on x86-64 the build with
/// BMI2 shifts and SSE residuals when the CPU has them (as the C++ reference
/// selects its BMI2 build at run time).
fn encode(
    src_image: &[u8],
    width: u32,
    height: u32,
    bytes_per_line: u32,
    dynamic: bool,
    out: &mut [MaybeUninit<u8>],
) -> crate::Result<usize> {
    if width == 0 || height == 0 || out.len() < max_compressed_size(width, height, dynamic) {
        return Err(crate::LlicError::InvalidArgument);
    }
    let (width, height, stride) = (width as usize, height as usize, bytes_per_line as usize);
    if stride < width || src_image.len() < (height - 1) * stride + width {
        return Err(crate::LlicError::InvalidArgument);
    }
    #[cfg(target_arch = "x86_64")]
    if simd_encoder_available() {
        // SAFETY: the CPU supports BMI2, SSSE3 and SSE4.1.
        return Ok(unsafe { encode_image_x86(src_image, width, height, stride, dynamic, out) });
    }
    Ok(encode_image::<false>(
        src_image, width, height, stride, dynamic, out,
    ))
}

/// Whether the CPU supports the x86-64 encoder build.
#[cfg(target_arch = "x86_64")]
fn simd_encoder_available() -> bool {
    #[cfg(test)]
    if tests::PORTABLE.with(std::cell::Cell::get) {
        return false;
    }
    is_x86_feature_detected!("bmi2")
        && is_x86_feature_detected!("ssse3")
        && is_x86_feature_detected!("sse4.1")
}

/// The image encoder built for BMI2 (shifts by a register become `SHLX`/`SHRX`,
/// one uop instead of three) with the SSE residual code inlined.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
unsafe fn encode_image_x86(
    src: &[u8],
    width: usize,
    height: usize,
    stride: usize,
    dynamic: bool,
    out: &mut [MaybeUninit<u8>],
) -> usize {
    encode_image::<true>(src, width, height, stride, dynamic, out)
}

/// Encodes an image into `out` (arguments checked by [`encode`]) and returns
/// the bytes written. `SIMD` selects the x86-64 SSE residual code, for callers
/// that enabled its target features.
#[inline(always)]
fn encode_image<const SIMD: bool>(
    src: &[u8],
    width: usize,
    height: usize,
    stride: usize,
    dynamic: bool,
    out: &mut [MaybeUninit<u8>],
) -> usize {
    let predictor_bytes = if dynamic {
        align_to_4(height.div_ceil(8))
    } else {
        0
    };
    let mut stream = Stream::new(out, predictor_bytes);
    let mut residuals = vec![0u8; width];
    let mut med_residuals = vec![0u8; if dynamic { width } else { 0 }];

    first_row_residuals::<SIMD>(&src[..width], &mut residuals);
    let mut use_med = false;
    let mut branch_free = true;
    for y in 1..height {
        // Code the previous row while this one is predicted.
        let row = if use_med { &med_residuals } else { &residuals };
        let row_bits = encode_row::<SIMD>(&mut stream, row, false, branch_free);
        // Above 5.5 bits per pair, flush timing is irregular enough that the
        // branch mispredicts more than storing every pair costs.
        branch_free = row_bits * 4 > 11 * width as u64;

        let current = &src[y * stride..y * stride + width];
        let prev = &src[(y - 1) * stride..(y - 1) * stride + width];
        if dynamic {
            let (avg_bits, med_bits) =
                dynamic_residual_rows::<SIMD>(current, prev, &mut residuals, &mut med_residuals);
            use_med = med_bits < avg_bits;
            if use_med {
                stream.set_predictor_bit(y);
            }
        } else {
            average_residuals::<SIMD>(current, prev, &mut residuals);
        }
    }
    let row = if use_med { &med_residuals } else { &residuals };
    encode_row::<SIMD>(&mut stream, row, true, branch_free);
    stream.finish()
}

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
#[inline(always)]
fn first_row_residuals<const SIMD: bool>(row: &[u8], out: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    if SIMD {
        // SAFETY: SIMD callers enabled SSSE3.
        return unsafe { compute_first_row_deltas_ssse3(row, out) };
    }
    compute_first_row_deltas(row, out);
}

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
#[inline(always)]
fn average_residuals<const SIMD: bool>(row: &[u8], prev: &[u8], out: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    if SIMD {
        // SAFETY: SIMD callers enabled SSSE3.
        return unsafe { compute_avg_predictor_deltas_ssse3(row, prev, out) };
    }
    compute_avg_predictor_deltas(row, prev, out);
}

/// WebAssembly with SIMD128: the residual rows are always vectorised.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
fn first_row_residuals<const SIMD: bool>(row: &[u8], out: &mut [u8]) {
    residuals_simd128::first_row(row, out);
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
fn average_residuals<const SIMD: bool>(row: &[u8], prev: &[u8], out: &mut [u8]) {
    residuals_simd128::average_row(row, prev, out);
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
fn dynamic_residual_rows<const SIMD: bool>(
    row: &[u8],
    prev: &[u8],
    avg_out: &mut [u8],
    med_out: &mut [u8],
) -> (u64, u64) {
    residuals_simd128::dynamic_rows(row, prev, avg_out, med_out)
}

/// Computes the average and MED residual rows and their coded sizes in bits.
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
#[inline(always)]
fn dynamic_residual_rows<const SIMD: bool>(
    row: &[u8],
    prev: &[u8],
    avg_out: &mut [u8],
    med_out: &mut [u8],
) -> (u64, u64) {
    #[cfg(target_arch = "x86_64")]
    if SIMD {
        // SAFETY: SIMD callers enabled SSSE3 and SSE4.1.
        return unsafe { dynamic_residual_rows_sse(row, prev, avg_out, med_out) };
    }
    compute_avg_predictor_deltas(row, prev, avg_out);
    compute_med_residual_row(row, prev, med_out);
    (estimate_row_bits(avg_out), estimate_row_bits(med_out))
}

/// Computes the average and MED residual rows and their coded sizes in one
/// pass, sixteen pixels at a time (the C++ `computeDynamicResidualRows`).
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "ssse3,sse4.1")]
unsafe fn dynamic_residual_rows_sse(
    row: &[u8],
    prev: &[u8],
    avg_out: &mut [u8],
    med_out: &mut [u8],
) -> (u64, u64) {
    use std::arch::x86_64::*;

    /// Code lengths of sixteen residuals, summed into two 64-bit lanes. Lengths
    /// depend on the signed magnitude: a table up to 16, 13 bits beyond.
    #[inline(always)]
    unsafe fn code_length_sums(residuals: __m128i) -> __m128i {
        let lengths = _mm_setr_epi8(2, 3, 3, 5, 5, 7, 7, 7, 7, 9, 9, 9, 9, 9, 9, 9);
        let magnitude = _mm_abs_epi8(residuals);
        let short_code = _mm_cmpeq_epi8(_mm_min_epu8(magnitude, _mm_set1_epi8(16)), magnitude);
        let length = _mm_add_epi8(
            _mm_shuffle_epi8(lengths, _mm_min_epu8(magnitude, _mm_set1_epi8(15))),
            _mm_andnot_si128(short_code, _mm_set1_epi8(4)),
        );
        _mm_sad_epu8(length, _mm_setzero_si128())
    }

    #[allow(clippy::cast_sign_loss)] // each lane of a row sum fits in 32 bits
    unsafe fn add_lanes(sums: __m128i) -> u64 {
        u64::from(_mm_cvtsi128_si32(sums) as u32)
            + u64::from(_mm_cvtsi128_si32(_mm_unpackhi_epi64(sums, sums)) as u32)
    }

    let width = row.len();
    assert!(prev.len() >= width && avg_out.len() >= width && med_out.len() >= width);
    let mut avg_sums = _mm_setzero_si128();
    let mut med_sums = _mm_setzero_si128();
    // With left equal to top, both predictors return top for the first pixel.
    #[allow(clippy::cast_possible_wrap)]
    let mut prev_current = _mm_set1_epi8(prev[0] as i8);
    let mut prev_top = prev_current;
    let one = _mm_set1_epi8(1);
    let mut x = 0;
    while x + 16 <= width {
        let current = _mm_loadu_si128(row.as_ptr().add(x).cast());
        let top = _mm_loadu_si128(prev.as_ptr().add(x).cast());
        let left = _mm_alignr_epi8::<15>(current, prev_current);
        let top_left = _mm_alignr_epi8::<15>(top, prev_top);
        prev_current = current;
        prev_top = top;
        // PAVGB rounds up; subtract the rounding to get the floor average.
        let average = _mm_sub_epi8(
            _mm_avg_epu8(left, top),
            _mm_and_si128(_mm_xor_si128(left, top), one),
        );
        let min_ab = _mm_min_epu8(left, top);
        let max_ab = _mm_max_epu8(left, top);
        let clamped = _mm_min_epu8(_mm_max_epu8(top_left, min_ab), max_ab);
        let med = _mm_sub_epi8(_mm_add_epi8(left, top), clamped);
        let avg_residuals = _mm_sub_epi8(current, average);
        let med_residuals = _mm_sub_epi8(current, med);
        _mm_storeu_si128(avg_out.as_mut_ptr().add(x).cast(), avg_residuals);
        _mm_storeu_si128(med_out.as_mut_ptr().add(x).cast(), med_residuals);
        avg_sums = _mm_add_epi64(avg_sums, code_length_sums(avg_residuals));
        med_sums = _mm_add_epi64(med_sums, code_length_sums(med_residuals));
        x += 16;
    }
    let mut avg_bits = add_lanes(avg_sums);
    let mut med_bits = add_lanes(med_sums);
    for x in x..width {
        let left = if x == 0 { prev[0] } else { row[x - 1] };
        let top_left = if x == 0 { prev[0] } else { prev[x - 1] };
        let average = ((u16::from(left) + u16::from(prev[x])) >> 1) as u8;
        let avg_residual = row[x].wrapping_sub(average);
        let med_residual = row[x].wrapping_sub(med_predictor(left, prev[x], top_left));
        avg_out[x] = avg_residual;
        med_out[x] = med_residual;
        avg_bits += u64::from(single_entry(avg_residual) & 0x3F);
        med_bits += u64::from(single_entry(med_residual) & 0x3F);
    }
    (avg_bits, med_bits)
}

/// Codes a row with [`Stream::encode_row`], kept out of line: inlined into the
/// image loop, the row loops lost registers (the output base was reloaded
/// for every store and table indices went through SSE registers), which cost
/// about 10%. `SIMD` callers get the BMI2 build.
#[inline(always)]
fn encode_row<const SIMD: bool>(
    stream: &mut Stream<'_>,
    row: &[u8],
    final_row: bool,
    branch_free: bool,
) -> u64 {
    #[cfg(target_arch = "x86_64")]
    if SIMD {
        // SAFETY: SIMD callers run on a CPU with BMI2.
        return unsafe { encode_row_bmi2(stream, row, final_row, branch_free) };
    }
    encode_row_portable(stream, row, final_row, branch_free)
}

#[inline(never)]
fn encode_row_portable(
    stream: &mut Stream<'_>,
    row: &[u8],
    final_row: bool,
    branch_free: bool,
) -> u64 {
    stream.encode_row(row, final_row, branch_free)
}

#[cfg(target_arch = "x86_64")]
#[inline(never)]
#[target_feature(enable = "bmi2")]
unsafe fn encode_row_bmi2(
    stream: &mut Stream<'_>,
    row: &[u8],
    final_row: bool,
    branch_free: bool,
) -> u64 {
    stream.encode_row(row, final_row, branch_free)
}

/// Bit stream output. Codes are packed MSB-first into a 64-bit container;
/// each full 32-bit word is stored as two 16-bit little-endian halves, high
/// half first (the C++ `storeWord`). The predictor bitmap, if any, goes in
/// front when the stream is finished.
struct Stream<'a> {
    /// Bytes `[predictor bitmap, pos)` are written; the rest is unwritten.
    out: &'a mut [MaybeUninit<u8>],
    pos: usize,
    bits: u64,
    num_bits: u32,
    predictor_bits: Vec<u8>,
}

impl<'a> Stream<'a> {
    /// A stream into `out` after a predictor bitmap of `predictor_bytes`.
    fn new(out: &'a mut [MaybeUninit<u8>], predictor_bytes: usize) -> Self {
        Self {
            out,
            pos: predictor_bytes,
            bits: 0,
            num_bits: 0,
            predictor_bits: vec![0; predictor_bytes],
        }
    }

    fn set_predictor_bit(&mut self, y: usize) {
        self.predictor_bits[y / 8] |= 1 << (y % 8);
    }

    /// Stores the container's top word at `pos` (without advancing).
    ///
    /// # Safety
    /// `pos + 4` must be within `out`.
    #[inline(always)]
    unsafe fn store_word(out: *mut MaybeUninit<u8>, pos: usize, bits: u64) {
        #[allow(clippy::cast_possible_truncation)] // the top word
        let word = ((bits >> 32) as u32).rotate_left(16);
        out.add(pos)
            .cast::<[u8; 4]>()
            .write_unaligned(word.to_le_bytes());
    }

    /// Codes a row of residuals in pairs (a trailing odd one alone) and
    /// returns the bits it took. With `branch_free`, words are flushed without
    /// a data-dependent branch: the top word is always stored and kept only
    /// when complete, so a following code must overwrite it or
    /// [`Stream::finish`] completes it; the final row leaves its last pair to
    /// the branching tail. The state stays in locals so it lives in registers,
    /// and both loops are unrolled by eight like the C++ `compressRow`.
    #[inline(always)]
    fn encode_row(&mut self, residuals: &[u8], final_row: bool, branch_free: bool) -> u64 {
        // At most 13 bits per residual on top of the pending bits, plus the
        // branch-free flush's speculative word. Rows before this one took at
        // most 13 bits per pixel, so this stays within `max_compressed_size`.
        let row_bytes = (self.num_bits as usize + 13 * residuals.len()).div_ceil(32) * 4 + 4;
        assert!(self.pos + row_bytes <= self.out.len());
        // A raw pointer keeps the output base in a register (a borrowed slice
        // was spilled and reloaded for every store).
        let out = self.out.as_mut_ptr();
        let (mut bits, mut num_bits, mut pos) = (self.bits, u64::from(self.num_bits), self.pos);
        let start = (pos, num_bits);
        let num_pairs = residuals.len() / 2;
        let tail = &residuals[2 * num_pairs..];
        // Reads pair `i` as one 16-bit load (`as_chunks` needs Rust 1.88).
        let entry = |i: usize| {
            debug_assert!(i < num_pairs);
            // SAFETY: every caller passes i < num_pairs, so bytes 2i and 2i + 1
            // are in `residuals`.
            let pair = unsafe {
                residuals
                    .as_ptr()
                    .add(2 * i)
                    .cast::<[u8; 2]>()
                    .read_unaligned()
            };
            get_compress_table_entry(usize::from(u16::from_le_bytes(pair)))
        };

        // Appends a table entry's code.
        macro_rules! put {
            ($entry:expr) => {{
                let entry: u32 = $entry;
                bits |= (u64::from(entry & 0xFFFF_FFC0) << 32) >> num_bits;
                num_bits += u64::from(entry & 0x3F);
            }};
        }
        // Stores a complete word, if there is one.
        macro_rules! flush {
            () => {
                if num_bits >= 32 {
                    num_bits -= 32;
                    // SAFETY: the capacity was checked above.
                    unsafe { Self::store_word(out, pos, bits) };
                    pos += 4;
                    bits <<= 32;
                }
            };
        }

        let mut i = 0;
        if branch_free {
            let unrolled = num_pairs.saturating_sub(usize::from(final_row)) / 8 * 8;
            while i < unrolled {
                for j in 0..8 {
                    let entry = entry(i + j);
                    bits |= (u64::from(entry & 0xFFFF_FFC0) << 32) >> (num_bits & 0x3F);
                    // At most 31 pending bits plus a 26-bit pair: bit 5 marks a
                    // complete word. Only the low six bits of `num_bits` count.
                    num_bits += u64::from(entry);
                    let full = num_bits & 32;
                    // SAFETY: the capacity was checked above.
                    unsafe { Self::store_word(out, pos, bits) };
                    pos += (full >> 3) as usize;
                    bits <<= full;
                    num_bits -= full;
                }
                i += 8;
            }
            num_bits &= 0x3F;
        } else {
            // Short codes flush rarely and predictably; load each next entry
            // before the preceding flush branch.
            while i + 8 <= num_pairs {
                let mut next = entry(i);
                for j in 1..8 {
                    put!(next);
                    next = entry(i + j);
                    flush!();
                }
                put!(next);
                flush!();
                i += 8;
            }
        }
        for i in i..num_pairs {
            put!(entry(i));
            flush!();
        }
        if let [sym] = tail {
            put!(single_entry(*sym));
            flush!();
        }
        #[allow(clippy::cast_possible_truncation)] // below 32
        {
            (self.bits, self.num_bits, self.pos) = (bits, num_bits as u32, pos);
        }
        ((pos - start.0) * 8) as u64 + num_bits - start.1
    }

    /// Stores the last, partial word and the predictor bitmap, and returns
    /// the bytes written.
    fn finish(mut self) -> usize {
        if self.num_bits > 0 {
            // SAFETY: the last row's capacity check included this word.
            unsafe { Self::store_word(self.out.as_mut_ptr(), self.pos, self.bits) };
            self.pos += 4;
        }
        for (slot, &byte) in self.out.iter_mut().zip(&self.predictor_bits) {
            slot.write(byte);
        }
        self.pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    thread_local! {
        /// Makes the encoder use its portable build on this thread.
        pub(super) static PORTABLE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    /// The BMI2/SSE build and the portable build must write identical streams
    /// (both flush strategies, odd widths, both predictors, narrow rows).
    #[test]
    fn simd_and_portable_encoders_match() {
        let mut state = 0x2468_ACE1_u32;
        for (width, height, stride) in [
            (1u32, 1u32, 1u32),
            (3, 5, 4),
            (17, 9, 17),
            (65, 33, 70),
            (300, 40, 300),
        ] {
            for amplitude in [2u32, 9, 70, 256] {
                let image: Vec<u8> = (0..stride * height)
                    .map(|i| {
                        state ^= state << 13;
                        state ^= state >> 17;
                        state ^= state << 5;
                        #[allow(clippy::cast_possible_truncation)]
                        let value = ((i % stride) * 2 + (i / stride) + state % amplitude) as u8;
                        value
                    })
                    .collect();
                for dynamic in [false, true] {
                    let run =
                        || encode_vec(&image, width, height, stride, dynamic).expect("encode");
                    let simd = run();
                    PORTABLE.with(|p| p.set(true));
                    let portable = run();
                    PORTABLE.with(|p| p.set(false));
                    assert_eq!(
                        simd, portable,
                        "{width}x{height} amplitude {amplitude} dynamic {dynamic}"
                    );
                    let mut decoded = vec![0u8; image.len()];
                    if dynamic {
                        decompress_dynamic(&simd, width, height, stride, &mut decoded)
                    } else {
                        decompress(&simd, width, height, stride, &mut decoded)
                    }
                    .expect("decode");
                    for y in 0..height as usize {
                        let row = y * stride as usize..y * stride as usize + width as usize;
                        assert_eq!(decoded[row.clone()], image[row]);
                    }
                }
            }
        }
    }
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
