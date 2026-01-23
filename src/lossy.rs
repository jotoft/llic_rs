//! Lossy compression and decompression for LLIC tile-based compression.
//!
//! This implements the 4x4 tile-based lossy compression algorithm used by LLIC
//! for quality levels VeryHigh (2), High (4), Medium (8), and Low (16).

use crate::{LlicError, Result};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Bucket LUT entry: (bits, bucket_size)
/// - bits: number of bits per pixel index (0, 1, 2, 3, 4, 5, 6, or 8)
/// - bucket_size: quantization step size
type BucketLutEntry = (u8, u8);

/// Precomputed reciprocals for fast division: x / d ≈ (x * RECIP[d]) >> 16
/// For bucket_size d in [1, 256], RECIP[d] = min(65536 / d, 65535)
/// Using u32 to avoid overflow in computation
static RECIPROCAL_TABLE: [u32; 257] = {
    let mut table = [0u32; 257];
    table[0] = 0; // unused, avoid div by zero
    let mut i = 1usize;
    while i <= 256 {
        let recip = 65536u32 / i as u32;
        // Cap at 65535 to fit in mulhi_epu16 range (though we use u32 here)
        table[i] = if recip > 65535 { 65535 } else { recip };
        i += 1;
    }
    table
};

/// Generate bucket lookup table for a given error limit (quality level).
///
/// For each possible dist value (0-255), determines the minimum number of
/// buckets needed such that bucket_size <= error_limit + 1.
fn generate_bucket_lut(error_limit: u8) -> [BucketLutEntry; 256] {
    let mut lut = [(0u8, 0u8); 256];
    let error_limit_plus1 = (error_limit as u32) + 1;

    for dist in 0..256u32 {
        let dist_plus1 = dist + 1;

        // Try with 1 bucket (0 bits)
        if dist_plus1 <= error_limit_plus1 {
            lut[dist as usize] = (0, dist_plus1 as u8);
            continue;
        }

        // Try with 2 buckets (1 bit)
        let bucket_size = dist_plus1.div_ceil(2);
        if bucket_size <= error_limit_plus1 {
            lut[dist as usize] = (1, bucket_size as u8);
            continue;
        }

        // Try with 4 buckets (2 bits)
        let bucket_size = dist_plus1.div_ceil(4);
        if bucket_size <= error_limit_plus1 {
            lut[dist as usize] = (2, bucket_size as u8);
            continue;
        }

        // Try with 8 buckets (3 bits)
        let bucket_size = dist_plus1.div_ceil(8);
        if bucket_size <= error_limit_plus1 {
            lut[dist as usize] = (3, bucket_size as u8);
            continue;
        }

        // Try with 16 buckets (4 bits)
        let bucket_size = dist_plus1.div_ceil(16);
        if bucket_size <= error_limit_plus1 {
            lut[dist as usize] = (4, bucket_size as u8);
            continue;
        }

        // Try with 32 buckets (5 bits)
        let bucket_size = dist_plus1.div_ceil(32);
        if bucket_size <= error_limit_plus1 {
            lut[dist as usize] = (5, bucket_size as u8);
            continue;
        }

        // Try with 64 buckets (6 bits)
        let bucket_size = dist_plus1.div_ceil(64);
        if bucket_size <= error_limit_plus1 {
            lut[dist as usize] = (6, bucket_size as u8);
            continue;
        }

        // Fall back to 256 buckets (8 bits)
        lut[dist as usize] = (8, 1);
    }

    lut
}

/// SSE4.1-optimized processing of a 4x4 tile matching C++ approach:
/// - Expands pixels to 32-bit integers
/// - Uses float reciprocal for division (very fast on modern CPUs)
/// - Outputs to 32-bit aligned result array for efficient packing
/// Returns (min_val, dist, bits)
#[cfg(all(target_arch = "x86_64", target_feature = "sse4.1"))]
#[inline(always)]
unsafe fn process_tile_sse41(
    src_data: &[u8],
    row0_start: usize,
    row1_start: usize,
    row2_start: usize,
    row3_start: usize,
    bucket_lut: &[BucketLutEntry; 256],
    result: &mut [u32; 16],
) -> (u8, u8, u8) {
    // Load 4 bytes per row and expand to 4x32-bit integers (matches C++ _MM_CVTEPU8_EPI32)
    let row0 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(
        *(src_data.as_ptr().add(row0_start) as *const i32),
    ));
    let row1 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(
        *(src_data.as_ptr().add(row1_start) as *const i32),
    ));
    let row2 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(
        *(src_data.as_ptr().add(row2_start) as *const i32),
    ));
    let row3 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(
        *(src_data.as_ptr().add(row3_start) as *const i32),
    ));

    // Find min/max using 32-bit comparisons
    let min01 = _mm_min_epi32(row0, row1);
    let min23 = _mm_min_epi32(row2, row3);
    let max01 = _mm_max_epi32(row0, row1);
    let max23 = _mm_max_epi32(row2, row3);

    let min4 = _mm_min_epi32(min01, min23);
    let max4 = _mm_max_epi32(max01, max23);

    // Horizontal reduction for min
    let min2 = _mm_min_epi32(min4, _mm_shuffle_epi32(min4, 0b00_00_11_10));
    let min1 = _mm_min_epi32(min2, _mm_shuffle_epi32(min2, 0b00_00_00_01));

    // Horizontal reduction for max
    let max2 = _mm_max_epi32(max4, _mm_shuffle_epi32(max4, 0b00_00_11_10));
    let max1 = _mm_max_epi32(max2, _mm_shuffle_epi32(max2, 0b00_00_00_01));

    let min_val = _mm_cvtsi128_si32(min1) as u8;
    let max_val = _mm_cvtsi128_si32(max1) as u8;
    let dist = max_val - min_val;

    let (bits, bucket_size) = bucket_lut[dist as usize];

    if bits > 0 {
        let v_min = _mm_set1_epi32(min_val as i32);

        // Subtract min from all pixels
        let res0 = _mm_sub_epi32(row0, v_min);
        let res1 = _mm_sub_epi32(row1, v_min);
        let res2 = _mm_sub_epi32(row2, v_min);
        let res3 = _mm_sub_epi32(row3, v_min);

        if bucket_size > 1 {
            // Use float reciprocal like C++: int32 -> float -> multiply -> truncate
            let v_rcp = _mm_set1_ps(1.0f32 / bucket_size as f32);

            let res0 = _mm_cvttps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(res0), v_rcp));
            let res1 = _mm_cvttps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(res1), v_rcp));
            let res2 = _mm_cvttps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(res2), v_rcp));
            let res3 = _mm_cvttps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(res3), v_rcp));

            _mm_storeu_si128(result.as_mut_ptr().add(0) as *mut __m128i, res0);
            _mm_storeu_si128(result.as_mut_ptr().add(4) as *mut __m128i, res1);
            _mm_storeu_si128(result.as_mut_ptr().add(8) as *mut __m128i, res2);
            _mm_storeu_si128(result.as_mut_ptr().add(12) as *mut __m128i, res3);
        } else {
            // bucket_size == 1, indices = diff directly
            _mm_storeu_si128(result.as_mut_ptr().add(0) as *mut __m128i, res0);
            _mm_storeu_si128(result.as_mut_ptr().add(4) as *mut __m128i, res1);
            _mm_storeu_si128(result.as_mut_ptr().add(8) as *mut __m128i, res2);
            _mm_storeu_si128(result.as_mut_ptr().add(12) as *mut __m128i, res3);
        }
    }

    (min_val, dist, bits)
}

/// SSE2-only fallback (for systems without SSE4.1)
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse2",
    not(target_feature = "sse4.1")
))]
#[inline(always)]
unsafe fn process_tile_sse2(
    src_data: &[u8],
    row0_start: usize,
    row1_start: usize,
    row2_start: usize,
    row3_start: usize,
    bucket_lut: &[BucketLutEntry; 256],
    indices: &mut [u8; 16],
) -> (u8, u8, u8) {
    // Load 4 bytes from each row into a single 128-bit register
    // We use u32 loads and combine them
    let r0 = _mm_cvtsi32_si128(*(src_data.as_ptr().add(row0_start) as *const i32));
    let r1 = _mm_cvtsi32_si128(*(src_data.as_ptr().add(row1_start) as *const i32));
    let r2 = _mm_cvtsi32_si128(*(src_data.as_ptr().add(row2_start) as *const i32));
    let r3 = _mm_cvtsi32_si128(*(src_data.as_ptr().add(row3_start) as *const i32));

    // Combine: [r0 | r1 | r2 | r3] in low 16 bytes
    let r01 = _mm_unpacklo_epi32(r0, r1);
    let r23 = _mm_unpacklo_epi32(r2, r3);
    let pixels = _mm_unpacklo_epi64(r01, r23);

    // Find min and max using horizontal operations
    // SSE2 approach: compare and reduce
    let mut min_vec = pixels;
    let mut max_vec = pixels;

    // Fold 16 -> 8
    let shifted = _mm_srli_si128(pixels, 8);
    min_vec = _mm_min_epu8(min_vec, shifted);
    max_vec = _mm_max_epu8(max_vec, shifted);

    // Fold 8 -> 4
    let shifted = _mm_srli_si128(min_vec, 4);
    min_vec = _mm_min_epu8(min_vec, shifted);
    let shifted = _mm_srli_si128(max_vec, 4);
    max_vec = _mm_max_epu8(max_vec, shifted);

    // Fold 4 -> 2
    let shifted = _mm_srli_si128(min_vec, 2);
    min_vec = _mm_min_epu8(min_vec, shifted);
    let shifted = _mm_srli_si128(max_vec, 2);
    max_vec = _mm_max_epu8(max_vec, shifted);

    // Fold 2 -> 1
    let shifted = _mm_srli_si128(min_vec, 1);
    min_vec = _mm_min_epu8(min_vec, shifted);
    let shifted = _mm_srli_si128(max_vec, 1);
    max_vec = _mm_max_epu8(max_vec, shifted);

    let min_val = _mm_cvtsi128_si32(min_vec) as u8;
    let max_val = _mm_cvtsi128_si32(max_vec) as u8;
    let dist = max_val - min_val;

    let (bits, bucket_size) = bucket_lut[dist as usize];

    if bits > 0 {
        let min_vec = _mm_set1_epi8(min_val as i8);

        // Subtract min from all pixels (saturating to handle underflow)
        let diff = _mm_subs_epu8(pixels, min_vec);

        if bucket_size == 1 {
            // For bucket_size = 1, indices = diff directly (no division needed)
            _mm_storeu_si128(indices.as_mut_ptr() as *mut __m128i, diff);
        } else {
            // Quantize using reciprocal multiplication: (pixel - min) / bucket_size
            // We unpack to 16-bit for the multiply
            let zero = _mm_setzero_si128();
            let diff_lo = _mm_unpacklo_epi8(diff, zero); // 8 x u16
            let diff_hi = _mm_unpackhi_epi8(diff, zero); // 8 x u16

            // Multiply by reciprocal and shift: (val * recip) >> 16
            let recip = RECIPROCAL_TABLE[bucket_size as usize];
            let recip_vec = _mm_set1_epi16(recip as i16);

            let idx_lo = _mm_mulhi_epu16(diff_lo, recip_vec);
            let idx_hi = _mm_mulhi_epu16(diff_hi, recip_vec);

            // Pack back to 8-bit
            let idx_packed = _mm_packus_epi16(idx_lo, idx_hi);

            // Store to indices array
            _mm_storeu_si128(indices.as_mut_ptr() as *mut __m128i, idx_packed);
        }
    }

    (min_val, dist, bits)
}

/// Scalar fallback for tile processing
#[allow(dead_code)]
#[inline(always)]
fn process_tile_scalar(
    src_data: &[u8],
    row0_start: usize,
    row1_start: usize,
    row2_start: usize,
    row3_start: usize,
    bucket_lut: &[BucketLutEntry; 256],
    indices: &mut [u8; 16],
) -> (u8, u8, u8) {
    // Find min and max
    let mut min_val = 255u8;
    let mut max_val = 0u8;

    for (i, &start) in [row0_start, row1_start, row2_start, row3_start]
        .iter()
        .enumerate()
    {
        for xx in 0..4 {
            let pixel = src_data[start + xx];
            min_val = min_val.min(pixel);
            max_val = max_val.max(pixel);
            indices[i * 4 + xx] = pixel; // Store for later quantization
        }
    }

    let dist = max_val - min_val;
    let (bits, bucket_size) = bucket_lut[dist as usize];

    if bits > 0 {
        if bucket_size == 1 {
            // For bucket_size = 1, indices = pixel - min directly
            for idx in indices.iter_mut() {
                *idx = idx.wrapping_sub(min_val);
            }
        } else {
            // Quantize using reciprocal multiplication
            let recip = RECIPROCAL_TABLE[bucket_size as usize];
            for idx in indices.iter_mut() {
                let diff = (*idx).wrapping_sub(min_val) as u32;
                *idx = ((diff * recip) >> 16) as u8;
            }
        }
    }

    (min_val, dist, bits)
}

/// Unpack 16 pixel indices from the compressed stream.
///
/// Returns the number of bytes consumed from the stream.
fn unpack_pixels(src: &[u8], bits: u8, result: &mut [u8; 16]) -> usize {
    match bits {
        0 => {
            // All pixels are the same (filled with min value)
            // No bytes consumed
            result.fill(0);
            0
        }
        1 => {
            // 2 bytes -> 16 pixels (1 bit each)
            // Specific column-major ordering from C++
            let byte0 = src[0];
            let byte1 = src[1];

            result[1] = byte0 & 0x1;
            result[5] = (byte0 >> 1) & 0x1;
            result[9] = (byte0 >> 2) & 0x1;
            result[13] = (byte0 >> 3) & 0x1;
            result[0] = (byte0 >> 4) & 0x1;
            result[4] = (byte0 >> 5) & 0x1;
            result[8] = (byte0 >> 6) & 0x1;
            result[12] = (byte0 >> 7) & 0x1;

            result[3] = byte1 & 0x1;
            result[7] = (byte1 >> 1) & 0x1;
            result[11] = (byte1 >> 2) & 0x1;
            result[15] = (byte1 >> 3) & 0x1;
            result[2] = (byte1 >> 4) & 0x1;
            result[6] = (byte1 >> 5) & 0x1;
            result[10] = (byte1 >> 6) & 0x1;
            result[14] = (byte1 >> 7) & 0x1;

            2
        }
        2 => {
            // 4 bytes -> 16 pixels (2 bits each)
            let byte0 = src[0];
            let byte1 = src[1];
            let byte2 = src[2];
            let byte3 = src[3];

            result[0] = byte0 & 0x3;
            result[1] = byte1 & 0x3;
            result[2] = byte2 & 0x3;
            result[3] = byte3 & 0x3;

            result[4] = (byte0 >> 2) & 0x3;
            result[5] = (byte1 >> 2) & 0x3;
            result[6] = (byte2 >> 2) & 0x3;
            result[7] = (byte3 >> 2) & 0x3;

            result[8] = (byte0 >> 4) & 0x3;
            result[9] = (byte1 >> 4) & 0x3;
            result[10] = (byte2 >> 4) & 0x3;
            result[11] = (byte3 >> 4) & 0x3;

            result[12] = (byte0 >> 6) & 0x3;
            result[13] = (byte1 >> 6) & 0x3;
            result[14] = (byte2 >> 6) & 0x3;
            result[15] = (byte3 >> 6) & 0x3;

            4
        }
        3 => {
            // 6 bytes -> 16 pixels (3 bits each)
            let word0 = (src[0] as u32) | ((src[1] as u32) << 8) | ((src[2] as u32) << 16);
            let word1 = (src[3] as u32) | ((src[4] as u32) << 8) | ((src[5] as u32) << 16);

            result[0] = (word0 & 0x7) as u8;
            result[4] = ((word0 >> 3) & 0x7) as u8;
            result[8] = ((word0 >> 6) & 0x7) as u8;
            result[12] = ((word0 >> 9) & 0x7) as u8;
            result[2] = ((word0 >> 12) & 0x7) as u8;
            result[6] = ((word0 >> 15) & 0x7) as u8;
            result[10] = ((word0 >> 18) & 0x7) as u8;
            result[14] = ((word0 >> 21) & 0x7) as u8;

            result[1] = (word1 & 0x7) as u8;
            result[5] = ((word1 >> 3) & 0x7) as u8;
            result[9] = ((word1 >> 6) & 0x7) as u8;
            result[13] = ((word1 >> 9) & 0x7) as u8;
            result[3] = ((word1 >> 12) & 0x7) as u8;
            result[7] = ((word1 >> 15) & 0x7) as u8;
            result[11] = ((word1 >> 18) & 0x7) as u8;
            result[15] = ((word1 >> 21) & 0x7) as u8;

            6
        }
        4 => {
            // 8 bytes -> 16 pixels (4 bits each, nibbles)
            result[0] = src[0] & 0xf;
            result[8] = (src[0] >> 4) & 0xf;
            result[1] = src[1] & 0xf;
            result[9] = (src[1] >> 4) & 0xf;
            result[2] = src[2] & 0xf;
            result[10] = (src[2] >> 4) & 0xf;
            result[3] = src[3] & 0xf;
            result[11] = (src[3] >> 4) & 0xf;
            result[4] = src[4] & 0xf;
            result[12] = (src[4] >> 4) & 0xf;
            result[5] = src[5] & 0xf;
            result[13] = (src[5] >> 4) & 0xf;
            result[6] = src[6] & 0xf;
            result[14] = (src[6] >> 4) & 0xf;
            result[7] = src[7] & 0xf;
            result[15] = (src[7] >> 4) & 0xf;

            8
        }
        5 => {
            // 10 bytes -> 16 pixels (5 bits each)
            let word0 = (src[0] as u32)
                | ((src[1] as u32) << 8)
                | ((src[2] as u32) << 16)
                | ((src[3] as u32) << 24);
            let word1 = (src[4] as u32)
                | ((src[5] as u32) << 8)
                | ((src[6] as u32) << 16)
                | ((src[7] as u32) << 24);
            let word2 = (src[8] as u32) | ((src[9] as u32) << 8);

            result[0] = (word0 & 0x1f) as u8;
            result[4] = ((word0 >> 5) & 0x1f) as u8;
            result[8] = ((word0 >> 10) & 0x1f) as u8;
            result[12] = ((word0 >> 15) & 0x1f) as u8;
            result[1] = ((word0 >> 20) & 0x1f) as u8;
            result[5] = ((word0 >> 25) & 0x1f) as u8;
            result[9] = (((word0 >> 30) & 0x03) | ((word1 << 2) & 0x1c)) as u8;
            result[13] = ((word1 >> 3) & 0x1f) as u8;
            result[2] = ((word1 >> 8) & 0x1f) as u8;
            result[6] = ((word1 >> 13) & 0x1f) as u8;
            result[10] = ((word1 >> 18) & 0x1f) as u8;
            result[14] = ((word1 >> 23) & 0x1f) as u8;
            result[3] = (((word1 >> 28) & 0x0f) | ((word2 << 4) & 0x10)) as u8;
            result[7] = ((word2 >> 1) & 0x1f) as u8;
            result[11] = ((word2 >> 6) & 0x1f) as u8;
            result[15] = ((word2 >> 11) & 0x1f) as u8;

            10
        }
        6 => {
            // 12 bytes -> 16 pixels (6 bits each)
            let word0 = (src[0] as u32)
                | ((src[1] as u32) << 8)
                | ((src[2] as u32) << 16)
                | ((src[3] as u32) << 24);
            let word1 = (src[4] as u32)
                | ((src[5] as u32) << 8)
                | ((src[6] as u32) << 16)
                | ((src[7] as u32) << 24);
            let word2 = (src[8] as u32)
                | ((src[9] as u32) << 8)
                | ((src[10] as u32) << 16)
                | ((src[11] as u32) << 24);

            result[0] = (word0 & 0x3f) as u8;
            result[4] = ((word0 >> 6) & 0x3f) as u8;
            result[8] = ((word0 >> 12) & 0x3f) as u8;
            result[12] = ((word0 >> 18) & 0x3f) as u8;
            result[1] = ((word0 >> 24) & 0x3f) as u8;
            result[5] = (((word0 >> 30) & 0x03) | ((word1 << 2) & 0x3c)) as u8;
            result[9] = ((word1 >> 4) & 0x3f) as u8;
            result[13] = ((word1 >> 10) & 0x3f) as u8;
            result[2] = ((word1 >> 16) & 0x3f) as u8;
            result[6] = ((word1 >> 22) & 0x3f) as u8;
            result[10] = (((word1 >> 28) & 0x0f) | ((word2 << 4) & 0x30)) as u8;
            result[14] = ((word2 >> 2) & 0x3f) as u8;
            result[3] = ((word2 >> 8) & 0x3f) as u8;
            result[7] = ((word2 >> 14) & 0x3f) as u8;
            result[11] = ((word2 >> 20) & 0x3f) as u8;
            result[15] = ((word2 >> 26) & 0x3f) as u8;

            12
        }
        8 => {
            // 16 bytes -> 16 pixels (8 bits each, direct copy)
            result.copy_from_slice(&src[..16]);
            16
        }
        _ => {
            // Invalid bits value
            result.fill(0);
            0
        }
    }
}

/// SSSE3-optimized pack for 4-bit indices (nibble packing).
/// Packs 16 indices into 8 bytes: dst[i] = (indices[i+8] << 4) | indices[i]
#[cfg(all(target_arch = "x86_64", target_feature = "ssse3"))]
#[inline(always)]
unsafe fn pack_pixels_4bit_ssse3(indices: &[u8; 16], dst: &mut [u8]) {
    // Load all 16 indices
    let v = _mm_loadu_si128(indices.as_ptr() as *const __m128i);

    // Shuffle to interleave: [i0,i8, i1,i9, i2,i10, i3,i11, i4,i12, i5,i13, i6,i14, i7,i15]
    let shuffle = _mm_setr_epi8(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
    let interleaved = _mm_shuffle_epi8(v, shuffle);

    // Now we need to combine adjacent pairs: (lo, hi) -> (hi << 4) | lo
    // Split into even (low nibbles) and odd (high nibbles) bytes
    let lo_mask = _mm_set1_epi16(0x00FF_u16 as i16);
    let lo = _mm_and_si128(interleaved, lo_mask);
    let hi = _mm_srli_epi16(interleaved, 8);
    let hi_shifted = _mm_slli_epi16(hi, 4);

    // Combine and pack to 8 bytes
    let combined = _mm_or_si128(lo, hi_shifted);

    // Pack 16-bit values to 8-bit (only low byte of each 16-bit lane)
    let pack_shuffle = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1);
    let packed = _mm_shuffle_epi8(combined, pack_shuffle);

    // Store 8 bytes
    _mm_storel_epi64(dst.as_mut_ptr() as *mut __m128i, packed);
}

/// SSE4.1-optimized pack for 32-bit result arrays (matches C++ addToCompressedStream).
/// Much faster than scalar when indices are already in 32-bit lanes.
#[cfg(all(target_arch = "x86_64", target_feature = "sse4.1"))]
#[inline(always)]
unsafe fn pack_pixels_u32(result: &[u32; 16], bits: u8, dst: &mut [u8]) -> usize {
    match bits {
        0 => 0,
        1 => {
            // Load 4 rows from 32-bit array
            let row0 = _mm_loadu_si128(result.as_ptr().add(0) as *const __m128i);
            let row1 = _mm_slli_epi32(_mm_loadu_si128(result.as_ptr().add(4) as *const __m128i), 1);
            let row2 = _mm_slli_epi32(_mm_loadu_si128(result.as_ptr().add(8) as *const __m128i), 2);
            let row3 = _mm_slli_epi32(
                _mm_loadu_si128(result.as_ptr().add(12) as *const __m128i),
                3,
            );
            let row0123 = _mm_or_si128(_mm_or_si128(row3, row2), _mm_or_si128(row1, row0));
            let row0123 = _mm_packs_epi32(row0123, row0123);
            let row0123 = _mm_packus_epi16(row0123, row0123);
            let row0123 = _mm_or_si128(
                _mm_and_si128(
                    _mm_slli_epi16(row0123, 4),
                    _mm_set1_epi32(0x00ff00ffu32 as i32),
                ),
                _mm_srli_epi16(row0123, 8),
            );
            let row0123 = _mm_or_si128(_mm_srli_si128(row0123, 1), row0123);
            *(dst.as_mut_ptr() as *mut u16) = _mm_cvtsi128_si32(row0123) as u16;
            2
        }
        2 => {
            let row0 = _mm_loadu_si128(result.as_ptr().add(0) as *const __m128i);
            let row1 = _mm_slli_epi32(_mm_loadu_si128(result.as_ptr().add(4) as *const __m128i), 2);
            let row2 = _mm_slli_epi32(_mm_loadu_si128(result.as_ptr().add(8) as *const __m128i), 4);
            let row3 = _mm_slli_epi32(
                _mm_loadu_si128(result.as_ptr().add(12) as *const __m128i),
                6,
            );
            let row0123 = _mm_or_si128(_mm_or_si128(row3, row2), _mm_or_si128(row1, row0));
            let row0123 = _mm_packs_epi32(row0123, row0123);
            let row0123 = _mm_packus_epi16(row0123, row0123);
            *(dst.as_mut_ptr() as *mut u32) = _mm_cvtsi128_si32(row0123) as u32;
            4
        }
        3 => {
            // 3-bit packing with SSSE3 shuffle
            #[repr(align(16))]
            struct Aligned([u8; 16]);
            static MASK: Aligned = Aligned([
                0, 1, 2, 4, 5, 6, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
            ]);

            let row0 = _mm_loadu_si128(result.as_ptr().add(0) as *const __m128i);
            let row1 = _mm_slli_epi32(_mm_loadu_si128(result.as_ptr().add(4) as *const __m128i), 3);
            let row2 = _mm_slli_epi32(_mm_loadu_si128(result.as_ptr().add(8) as *const __m128i), 6);
            let row3 = _mm_slli_epi32(
                _mm_loadu_si128(result.as_ptr().add(12) as *const __m128i),
                9,
            );
            let mut row0123 = _mm_or_si128(_mm_or_si128(row0, row1), _mm_or_si128(row2, row3));
            row0123 = _mm_or_si128(
                row0123,
                _mm_shuffle_epi32(_mm_slli_epi32(row0123, 12), 0b00_00_11_10),
            );
            row0123 = _mm_shuffle_epi8(row0123, _mm_load_si128(MASK.0.as_ptr() as *const __m128i));
            *(dst.as_mut_ptr() as *mut u32) = _mm_cvtsi128_si32(row0123) as u32;
            *(dst.as_mut_ptr().add(4) as *mut u16) = _mm_extract_epi16(row0123, 2) as u16;
            6
        }
        4 => {
            let row0 = _mm_loadu_si128(result.as_ptr().add(0) as *const __m128i);
            let row1 = _mm_loadu_si128(result.as_ptr().add(4) as *const __m128i);
            let row2 = _mm_slli_epi32(_mm_loadu_si128(result.as_ptr().add(8) as *const __m128i), 4);
            let row3 = _mm_slli_epi32(
                _mm_loadu_si128(result.as_ptr().add(12) as *const __m128i),
                4,
            );
            let row02 = _mm_or_si128(row0, row2);
            let row13 = _mm_or_si128(row1, row3);
            let row0213 = _mm_packs_epi32(row02, row13);
            let row0213 = _mm_packus_epi16(row0213, row0213);
            *(dst.as_mut_ptr() as *mut u32) = _mm_extract_epi32(row0213, 0) as u32;
            *(dst.as_mut_ptr().add(4) as *mut u32) = _mm_extract_epi32(row0213, 1) as u32;
            8
        }
        5 => {
            // 5-bit packing with SSSE3 shuffle
            #[repr(align(16))]
            struct Aligned([u8; 16]);
            static MASK1: Aligned = Aligned([
                0, 1, 2, 0x80, 0x80, 8, 9, 10, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
            ]);
            static MASK2: Aligned = Aligned([
                0x80, 0x80, 4, 5, 6, 0x80, 0x80, 12, 13, 14, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
            ]);

            let row0 = _mm_loadu_si128(result.as_ptr().add(0) as *const __m128i);
            let row1 = _mm_slli_epi32(_mm_loadu_si128(result.as_ptr().add(4) as *const __m128i), 5);
            let row2 = _mm_slli_epi32(
                _mm_loadu_si128(result.as_ptr().add(8) as *const __m128i),
                10,
            );
            let row3 = _mm_slli_epi32(
                _mm_loadu_si128(result.as_ptr().add(12) as *const __m128i),
                15,
            );
            let row0123 = _mm_or_si128(_mm_or_si128(row0, row1), _mm_or_si128(row2, row3));
            let row0123a =
                _mm_shuffle_epi8(row0123, _mm_load_si128(MASK1.0.as_ptr() as *const __m128i));
            let row0123b = _mm_shuffle_epi8(
                _mm_slli_epi32(row0123, 4),
                _mm_load_si128(MASK2.0.as_ptr() as *const __m128i),
            );
            let row0123 = _mm_or_si128(row0123a, row0123b);
            _mm_storel_epi64(dst.as_mut_ptr() as *mut __m128i, row0123);
            *(dst.as_mut_ptr().add(8) as *mut u16) = _mm_extract_epi16(row0123, 4) as u16;
            10
        }
        6 => {
            // 6-bit packing with SSSE3 shuffle
            #[repr(align(16))]
            struct Aligned([u8; 16]);
            static MASK1: Aligned = Aligned([
                0, 1, 2, 0x80, 8, 9, 10, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
            ]);
            static MASK2: Aligned = Aligned([
                0x80, 0x80, 4, 5, 0x80, 0x80, 12, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
                0x80,
            ]);

            let row0 = _mm_loadu_si128(result.as_ptr().add(0) as *const __m128i);
            let row1 = _mm_slli_epi32(_mm_loadu_si128(result.as_ptr().add(4) as *const __m128i), 6);
            let row2 = _mm_slli_epi32(
                _mm_loadu_si128(result.as_ptr().add(8) as *const __m128i),
                12,
            );
            let row3 = _mm_slli_epi32(
                _mm_loadu_si128(result.as_ptr().add(12) as *const __m128i),
                18,
            );
            let row0123 = _mm_or_si128(_mm_or_si128(row0, row1), _mm_or_si128(row2, row3));
            let row0123a =
                _mm_shuffle_epi8(row0123, _mm_load_si128(MASK1.0.as_ptr() as *const __m128i));
            let row0123b = _mm_shuffle_epi8(
                _mm_slli_epi32(row0123, 8),
                _mm_load_si128(MASK2.0.as_ptr() as *const __m128i),
            );
            let row0123 = _mm_or_si128(row0123a, row0123b);
            _mm_storel_epi64(dst.as_mut_ptr() as *mut __m128i, row0123);
            *(dst.as_mut_ptr().add(8) as *mut u32) = _mm_extract_epi32(row0123, 2) as u32;
            12
        }
        8 => {
            // 8-bit: just pack down to bytes
            let row0 = _mm_loadu_si128(result.as_ptr().add(0) as *const __m128i);
            let row1 = _mm_loadu_si128(result.as_ptr().add(4) as *const __m128i);
            let row2 = _mm_loadu_si128(result.as_ptr().add(8) as *const __m128i);
            let row3 = _mm_loadu_si128(result.as_ptr().add(12) as *const __m128i);
            let row01 = _mm_packs_epi32(row0, row1);
            let row23 = _mm_packs_epi32(row2, row3);
            let packed = _mm_packus_epi16(row01, row23);
            _mm_storeu_si128(dst.as_mut_ptr() as *mut __m128i, packed);
            16
        }
        _ => 0,
    }
}

/// Pack 16 pixel indices into the compressed stream.
///
/// This is the inverse of `unpack_pixels`. Returns the number of bytes written.
#[inline(always)]
fn pack_pixels(indices: &[u8; 16], bits: u8, dst: &mut [u8]) -> usize {
    match bits {
        0 => {
            // All pixels are the same - no data needed
            0
        }
        1 => {
            // 16 pixels -> 2 bytes (1 bit each)
            // Matches C++ addToCompressedStream scalar path
            dst[0] = (indices[12] << 7)
                | (indices[8] << 6)
                | (indices[4] << 5)
                | (indices[0] << 4)
                | (indices[13] << 3)
                | (indices[9] << 2)
                | (indices[5] << 1)
                | indices[1];
            dst[1] = (indices[14] << 7)
                | (indices[10] << 6)
                | (indices[6] << 5)
                | (indices[2] << 4)
                | (indices[15] << 3)
                | (indices[11] << 2)
                | (indices[7] << 1)
                | indices[3];
            2
        }
        2 => {
            // 16 pixels -> 4 bytes (2 bits each)
            dst[0] = (indices[12] << 6) | (indices[8] << 4) | (indices[4] << 2) | indices[0];
            dst[1] = (indices[13] << 6) | (indices[9] << 4) | (indices[5] << 2) | indices[1];
            dst[2] = (indices[14] << 6) | (indices[10] << 4) | (indices[6] << 2) | indices[2];
            dst[3] = (indices[15] << 6) | (indices[11] << 4) | (indices[7] << 2) | indices[3];
            4
        }
        3 => {
            // 16 pixels -> 6 bytes (3 bits each)
            let word0 = (indices[0] as u32)
                | ((indices[4] as u32) << 3)
                | ((indices[8] as u32) << 6)
                | ((indices[12] as u32) << 9)
                | ((indices[2] as u32) << 12)
                | ((indices[6] as u32) << 15)
                | ((indices[10] as u32) << 18)
                | ((indices[14] as u32) << 21);
            let word1 = (indices[1] as u32)
                | ((indices[5] as u32) << 3)
                | ((indices[9] as u32) << 6)
                | ((indices[13] as u32) << 9)
                | ((indices[3] as u32) << 12)
                | ((indices[7] as u32) << 15)
                | ((indices[11] as u32) << 18)
                | ((indices[15] as u32) << 21);
            dst[0] = word0 as u8;
            dst[1] = (word0 >> 8) as u8;
            dst[2] = (word0 >> 16) as u8;
            dst[3] = word1 as u8;
            dst[4] = (word1 >> 8) as u8;
            dst[5] = (word1 >> 16) as u8;
            6
        }
        4 => {
            // 16 pixels -> 8 bytes (4 bits each, nibbles)
            #[cfg(all(target_arch = "x86_64", target_feature = "ssse3"))]
            {
                unsafe { pack_pixels_4bit_ssse3(indices, dst) };
                return 8;
            }
            #[cfg(not(all(target_arch = "x86_64", target_feature = "ssse3")))]
            {
                dst[0] = (indices[8] << 4) | indices[0];
                dst[1] = (indices[9] << 4) | indices[1];
                dst[2] = (indices[10] << 4) | indices[2];
                dst[3] = (indices[11] << 4) | indices[3];
                dst[4] = (indices[12] << 4) | indices[4];
                dst[5] = (indices[13] << 4) | indices[5];
                dst[6] = (indices[14] << 4) | indices[6];
                dst[7] = (indices[15] << 4) | indices[7];
                8
            }
        }
        5 => {
            // 16 pixels -> 10 bytes (5 bits each)
            let word0 = (indices[0] as u32)
                | ((indices[4] as u32) << 5)
                | ((indices[8] as u32) << 10)
                | ((indices[12] as u32) << 15)
                | ((indices[1] as u32) << 20)
                | ((indices[5] as u32) << 25)
                | ((indices[9] as u32) << 30);
            let word1 = ((indices[9] as u32) >> 2)
                | ((indices[13] as u32) << 3)
                | ((indices[2] as u32) << 8)
                | ((indices[6] as u32) << 13)
                | ((indices[10] as u32) << 18)
                | ((indices[14] as u32) << 23)
                | ((indices[3] as u32) << 28);
            let word2 = ((indices[3] as u32) >> 4)
                | ((indices[7] as u32) << 1)
                | ((indices[11] as u32) << 6)
                | ((indices[15] as u32) << 11);
            dst[0] = word0 as u8;
            dst[1] = (word0 >> 8) as u8;
            dst[2] = (word0 >> 16) as u8;
            dst[3] = (word0 >> 24) as u8;
            dst[4] = word1 as u8;
            dst[5] = (word1 >> 8) as u8;
            dst[6] = (word1 >> 16) as u8;
            dst[7] = (word1 >> 24) as u8;
            dst[8] = word2 as u8;
            dst[9] = (word2 >> 8) as u8;
            10
        }
        6 => {
            // 16 pixels -> 12 bytes (6 bits each)
            let word0 = (indices[0] as u32)
                | ((indices[4] as u32) << 6)
                | ((indices[8] as u32) << 12)
                | ((indices[12] as u32) << 18)
                | ((indices[1] as u32) << 24)
                | ((indices[5] as u32) << 30);
            let word1 = ((indices[5] as u32) >> 2)
                | ((indices[9] as u32) << 4)
                | ((indices[13] as u32) << 10)
                | ((indices[2] as u32) << 16)
                | ((indices[6] as u32) << 22)
                | ((indices[10] as u32) << 28);
            let word2 = ((indices[10] as u32) >> 4)
                | ((indices[14] as u32) << 2)
                | ((indices[3] as u32) << 8)
                | ((indices[7] as u32) << 14)
                | ((indices[11] as u32) << 20)
                | ((indices[15] as u32) << 26);
            dst[0] = word0 as u8;
            dst[1] = (word0 >> 8) as u8;
            dst[2] = (word0 >> 16) as u8;
            dst[3] = (word0 >> 24) as u8;
            dst[4] = word1 as u8;
            dst[5] = (word1 >> 8) as u8;
            dst[6] = (word1 >> 16) as u8;
            dst[7] = (word1 >> 24) as u8;
            dst[8] = word2 as u8;
            dst[9] = (word2 >> 8) as u8;
            dst[10] = (word2 >> 16) as u8;
            dst[11] = (word2 >> 24) as u8;
            12
        }
        8 => {
            // 16 pixels -> 16 bytes (8 bits each, direct copy)
            dst[..16].copy_from_slice(indices);
            16
        }
        _ => 0,
    }
}

/// Compress a single block of grayscale image data using tile-based lossy compression.
///
/// This produces the same format as the C++ encoder's `doCompressTileBased` function.
///
/// # Arguments
/// * `src_data` - Source grayscale image data
/// * `width` - Image width in pixels (must be multiple of 4)
/// * `rows` - Number of rows to compress (must be multiple of 4)
/// * `bytes_per_line` - Stride of the source image
/// * `error_limit` - Quality level (2, 4, 8, or 16)
/// * `compress_header` - Whether to attempt header compression
/// * `use_dynamic_predictor` - Whether to use dynamic predictor for header compression
///
/// # Returns
/// Compressed data as a Vec<u8>
pub fn compress_tile_block(
    src_data: &[u8],
    width: u32,
    rows: u32,
    bytes_per_line: u32,
    error_limit: u8,
    compress_header: bool,
    use_dynamic_predictor: bool,
) -> Result<Vec<u8>> {
    if !width.is_multiple_of(4) || !rows.is_multiple_of(4) {
        return Err(LlicError::ImageDimensions);
    }

    let num_tiles = ((width / 4) * (rows / 4)) as usize;

    // Generate bucket LUT for this error_limit
    let bucket_lut = generate_bucket_lut(error_limit);

    // Allocate output buffer for uncompressed header first
    // Format: 1 byte header + num_tiles min values + num_tiles dist values + pixel data
    // Worst case pixel data: 16 bytes per tile (8-bit mode)
    let max_size = 1 + num_tiles * 2 + num_tiles * 16;
    let mut temp_output = vec![0u8; max_size];

    // First byte: flags (bit 7 = compressed header) + error_limit
    // Set bit 7 to 0 initially (uncompressed header)
    temp_output[0] = error_limit & 0x7f;

    // Split temp_output into non-overlapping mutable slices
    let (header_and_streams, pixel_stream) = temp_output[1..].split_at_mut(num_tiles * 2);
    let (min_stream, dist_stream) = header_and_streams.split_at_mut(num_tiles);
    let mut pixel_pos = 0usize;

    // Process each 4x4 block - use SSE4.1 optimized path if available
    #[cfg(all(target_arch = "x86_64", target_feature = "sse4.1"))]
    {
        // 32-bit aligned result buffer (matches C++ approach for efficient SIMD packing)
        let mut result = [0u32; 16];

        for y in (0..rows).step_by(4) {
            for x in (0..width).step_by(4) {
                let row0_start = (y * bytes_per_line + x) as usize;
                let row1_start = ((y + 1) * bytes_per_line + x) as usize;
                let row2_start = ((y + 2) * bytes_per_line + x) as usize;
                let row3_start = ((y + 3) * bytes_per_line + x) as usize;

                // SAFETY: We've verified dimensions are multiples of 4
                let (min_val, dist, bits) = unsafe {
                    process_tile_sse41(
                        src_data,
                        row0_start,
                        row1_start,
                        row2_start,
                        row3_start,
                        &bucket_lut,
                        &mut result,
                    )
                };

                let block_idx = ((y * width) >> 4) + (x >> 2);
                let block_idx = block_idx as usize;

                min_stream[block_idx] = min_val;
                dist_stream[block_idx] = dist;

                if bits > 0 {
                    let bytes_written =
                        unsafe { pack_pixels_u32(&result, bits, &mut pixel_stream[pixel_pos..]) };
                    pixel_pos += bytes_written;
                }
            }
        }
    }

    // SSE2-only fallback (no SSE4.1)
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "sse2",
        not(target_feature = "sse4.1")
    ))]
    {
        let mut indices = [0u8; 16];
        for y in (0..rows).step_by(4) {
            for x in (0..width).step_by(4) {
                let row0_start = (y * bytes_per_line + x) as usize;
                let row1_start = ((y + 1) * bytes_per_line + x) as usize;
                let row2_start = ((y + 2) * bytes_per_line + x) as usize;
                let row3_start = ((y + 3) * bytes_per_line + x) as usize;

                // SAFETY: We've verified dimensions are multiples of 4
                let (min_val, dist, bits) = unsafe {
                    process_tile_sse2(
                        src_data,
                        row0_start,
                        row1_start,
                        row2_start,
                        row3_start,
                        &bucket_lut,
                        &mut indices,
                    )
                };

                let block_idx = ((y * width) >> 4) + (x >> 2);
                let block_idx = block_idx as usize;

                min_stream[block_idx] = min_val;
                dist_stream[block_idx] = dist;

                if bits > 0 {
                    let bytes_written = pack_pixels(&indices, bits, &mut pixel_stream[pixel_pos..]);
                    pixel_pos += bytes_written;
                }
            }
        }
    }

    // Scalar fallback (no SIMD)
    #[cfg(not(all(target_arch = "x86_64", target_feature = "sse2")))]
    {
        let mut indices = [0u8; 16];
        for y in (0..rows).step_by(4) {
            for x in (0..width).step_by(4) {
                let row0_start = (y * bytes_per_line + x) as usize;
                let row1_start = ((y + 1) * bytes_per_line + x) as usize;
                let row2_start = ((y + 2) * bytes_per_line + x) as usize;
                let row3_start = ((y + 3) * bytes_per_line + x) as usize;

                let (min_val, dist, bits) = process_tile_scalar(
                    src_data,
                    row0_start,
                    row1_start,
                    row2_start,
                    row3_start,
                    &bucket_lut,
                    &mut indices,
                );

                let block_idx = ((y * width) >> 4) + (x >> 2);
                let block_idx = block_idx as usize;

                min_stream[block_idx] = min_val;
                dist_stream[block_idx] = dist;

                if bits > 0 {
                    let bytes_written = pack_pixels(&indices, bits, &mut pixel_stream[pixel_pos..]);
                    pixel_pos += bytes_written;
                }
            }
        }
    }

    // Calculate uncompressed size
    let uncompressed_size = 1 + num_tiles * 2 + pixel_pos;

    // Try header compression if requested
    if compress_header {
        // Arrange header as: [min_values][dist_values] in a single buffer
        let header_width = width / 4;
        let header_height = (rows / 4) * 2;
        let mut header_buffer = vec![0u8; num_tiles * 2];

        // Copy min and dist streams from their slices (which are views into temp_output)
        header_buffer[..num_tiles].copy_from_slice(min_stream);
        header_buffer[num_tiles..].copy_from_slice(dist_stream);

        // Compress header using entropy coder (dynamic predictor if requested)
        let compressed_header = if use_dynamic_predictor {
            crate::entropy_coder::compress_dynamic(
                &header_buffer,
                header_width,
                header_height,
                header_width,
            )?
        } else {
            crate::entropy_coder::compress(
                &header_buffer,
                header_width,
                header_height,
                header_width,
            )?
        };

        // Check if compression actually saves space (need at least 5 bytes overhead)
        let compressed_size_with_header = 1 + 4 + compressed_header.len() + pixel_pos;

        if compressed_size_with_header < uncompressed_size {
            // Use compressed header
            // Format: [flags|error_limit:1][header_size:4][compressed_header][pixel_data]
            let mut output = vec![0u8; compressed_size_with_header];
            output[0] = 0x80 | (error_limit & 0x7f); // Set bit 7 for compressed header

            // Write header size (little-endian u32)
            let header_size = compressed_header.len() as u32;
            output[1..5].copy_from_slice(&header_size.to_le_bytes());

            // Copy compressed header
            output[5..5 + compressed_header.len()].copy_from_slice(&compressed_header);

            // Copy pixel data
            output[5 + compressed_header.len()..].copy_from_slice(&pixel_stream[..pixel_pos]);

            return Ok(output);
        }
    }

    // Use uncompressed header (or compression didn't help)
    temp_output.truncate(uncompressed_size);
    Ok(temp_output)
}

/// Decompress a single block of tile-based lossy compressed data.
///
/// This handles one "thread block" of compressed data as produced by the C++ encoder.
/// The error_limit comes from the main header's quality byte, not from per-block data.
pub fn decompress_tile_block(
    src_data: &[u8],
    width: u32,
    rows: u32,
    bytes_per_line: u32,
    dst_graymap: &mut [u8],
) -> Result<()> {
    // Default: no dynamic predictor for headers
    decompress_tile_block_impl(src_data, width, rows, bytes_per_line, false, dst_graymap)
}

/// Decompress a single block of tile-based lossy compressed data with dynamic predictor option.
pub fn decompress_tile_block_dynamic(
    src_data: &[u8],
    width: u32,
    rows: u32,
    bytes_per_line: u32,
    use_dynamic_predictor: bool,
    dst_graymap: &mut [u8],
) -> Result<()> {
    decompress_tile_block_impl(
        src_data,
        width,
        rows,
        bytes_per_line,
        use_dynamic_predictor,
        dst_graymap,
    )
}

/// Decompress a tile-based block implementation.
///
/// The block format from C++:
/// - Byte 0: flags/quality (bit 7 = compressed header, bits 0-6 = error_limit)
/// - If uncompressed header (flags & 0x80 == 0):
///   - Bytes 1..: min[num_tiles], dist[num_tiles], pixels[...]
/// - If compressed header (flags & 0x80 == 0x80):
///   - Bytes 1-4: header_size (u32)
///   - Bytes 5..: compressed_header[header_size], pixels[...]
fn decompress_tile_block_impl(
    src_data: &[u8],
    width: u32,
    rows: u32,
    bytes_per_line: u32,
    use_dynamic_predictor: bool,
    dst_graymap: &mut [u8],
) -> Result<()> {
    if src_data.is_empty() {
        return Err(LlicError::InvalidData);
    }

    let num_tiles = ((width / 4) * (rows / 4)) as usize;

    // First byte contains flags and error_limit
    let flags = src_data[0];
    let compressed_header = (flags & 0x80) != 0;
    let error_limit = flags & 0x7f; // Extract error_limit from lower 7 bits

    // Generate bucket LUT for the actual error_limit from the block
    let bucket_lut = generate_bucket_lut(error_limit);

    if compressed_header {
        // Compressed header - decompress min/dist streams using entropy coder
        // Format: [flags:1][header_size:4][compressed_header:header_size][pixel_data:...]
        if src_data.len() < 5 {
            return Err(LlicError::InvalidData);
        }

        let header_size =
            u32::from_le_bytes([src_data[1], src_data[2], src_data[3], src_data[4]]) as usize;

        if src_data.len() < 5 + header_size {
            return Err(LlicError::InvalidData);
        }

        // The header is entropy-coded as a (width/4) × (rows/4 * 2) "image"
        // First half = min values, second half = dist values
        let header_width = width / 4;
        let header_height = (rows / 4) * 2;
        let header_decompressed_size = (header_width * header_height) as usize;

        // Decompress the header (using dynamic predictor if requested)
        let mut header_buffer = vec![0u8; header_decompressed_size];
        if use_dynamic_predictor {
            crate::entropy_coder::decompress_dynamic(
                &src_data[5..5 + header_size],
                header_width,
                header_height,
                header_width,
                &mut header_buffer,
            )?;
        } else {
            crate::entropy_coder::decompress(
                &src_data[5..5 + header_size],
                header_width,
                header_height,
                header_width,
                &mut header_buffer,
            )?;
        }

        // Split into min and dist streams
        let half = header_decompressed_size / 2;
        let min_stream = &header_buffer[..half];
        let dist_stream = &header_buffer[half..];
        let pixel_stream_start = 5 + header_size;

        decompress_tiles(
            min_stream,
            dist_stream,
            &src_data[pixel_stream_start..],
            width,
            rows,
            bytes_per_line,
            &bucket_lut,
            dst_graymap,
        )
    } else {
        // Uncompressed header: min and dist streams start at byte 1
        if src_data.len() < 1 + num_tiles * 2 {
            return Err(LlicError::InvalidData);
        }

        let min_stream = &src_data[1..1 + num_tiles];
        let dist_stream = &src_data[1 + num_tiles..1 + num_tiles * 2];
        let pixel_stream_start = 1 + num_tiles * 2;

        decompress_tiles(
            min_stream,
            dist_stream,
            &src_data[pixel_stream_start..],
            width,
            rows,
            bytes_per_line,
            &bucket_lut,
            dst_graymap,
        )
    }
}

/// Helper for decompressing with a pre-decompressed header buffer.
/// Reserved for future compressed header support.
#[allow(dead_code)]
fn decompress_tile_block_with_header(
    header_buffer: &[u8],
    pixel_stream: &[u8],
    width: u32,
    rows: u32,
    bytes_per_line: u32,
    bucket_lut: &[BucketLutEntry; 256],
    dst_graymap: &mut [u8],
) -> Result<()> {
    let num_blocks = ((width / 4) * (rows / 4)) as usize;

    let min_stream = &header_buffer[..num_blocks];
    let dist_stream = &header_buffer[num_blocks..num_blocks * 2];

    decompress_tiles(
        min_stream,
        dist_stream,
        pixel_stream,
        width,
        rows,
        bytes_per_line,
        bucket_lut,
        dst_graymap,
    )
}

/// Extract tile metadata (min, dist, bits) from compressed tile-based data.
///
/// Returns a vector of (min, dist, bits) tuples for each 4x4 tile.
/// This is useful for visualization and debugging.
///
/// # Arguments
/// * `src_data` - Compressed tile block data (starts after format header)
/// * `width` - Image width in pixels (must be multiple of 4)
/// * `height` - Image height in pixels (must be multiple of 4)
///
/// # Returns
/// Vec of (min, dist, bits) for each tile in row-major order
pub fn extract_tile_metadata(
    src_data: &[u8],
    width: u32,
    height: u32,
    use_dynamic_predictor: bool,
) -> Result<Vec<(u8, u8, u8)>> {
    if src_data.is_empty() {
        return Err(LlicError::InvalidData);
    }

    let num_tiles = ((width / 4) * (height / 4)) as usize;

    // First byte contains flags and error_limit
    let flags = src_data[0];
    let compressed_header = (flags & 0x80) != 0;
    let error_limit = flags & 0x7f;

    // Generate bucket LUT for this error_limit
    let bucket_lut = generate_bucket_lut(error_limit);

    let (min_stream, dist_stream): (Vec<u8>, Vec<u8>) = if compressed_header {
        // Compressed header - decompress min/dist streams using entropy coder
        if src_data.len() < 5 {
            return Err(LlicError::InvalidData);
        }

        let header_size =
            u32::from_le_bytes([src_data[1], src_data[2], src_data[3], src_data[4]]) as usize;

        if src_data.len() < 5 + header_size {
            return Err(LlicError::InvalidData);
        }

        // The header is entropy-coded as a (width/4) × (height/4 * 2) "image"
        let header_width = width / 4;
        let header_height = (height / 4) * 2;
        let header_decompressed_size = (header_width * header_height) as usize;

        // Decompress the header (using dynamic predictor if requested)
        let mut header_buffer = vec![0u8; header_decompressed_size];
        if use_dynamic_predictor {
            crate::entropy_coder::decompress_dynamic(
                &src_data[5..5 + header_size],
                header_width,
                header_height,
                header_width,
                &mut header_buffer,
            )?;
        } else {
            crate::entropy_coder::decompress(
                &src_data[5..5 + header_size],
                header_width,
                header_height,
                header_width,
                &mut header_buffer,
            )?;
        }

        // Split into min and dist streams
        let half = header_decompressed_size / 2;
        (
            header_buffer[..half].to_vec(),
            header_buffer[half..].to_vec(),
        )
    } else {
        // Uncompressed header: min and dist streams start at byte 1
        if src_data.len() < 1 + num_tiles * 2 {
            return Err(LlicError::InvalidData);
        }

        (
            src_data[1..1 + num_tiles].to_vec(),
            src_data[1 + num_tiles..1 + num_tiles * 2].to_vec(),
        )
    };

    // Build result vector
    let mut result = Vec::with_capacity(num_tiles);
    for i in 0..num_tiles {
        let min_val = min_stream[i];
        let dist = dist_stream[i];
        let (bits, _bucket_size) = bucket_lut[dist as usize];
        result.push((min_val, dist, bits));
    }

    Ok(result)
}

/// Core tile decompression loop.
#[allow(clippy::too_many_arguments)]
fn decompress_tiles(
    min_stream: &[u8],
    dist_stream: &[u8],
    pixel_stream: &[u8],
    width: u32,
    rows: u32,
    bytes_per_line: u32,
    bucket_lut: &[BucketLutEntry; 256],
    dst_graymap: &mut [u8],
) -> Result<()> {
    let mut pixel_pos = 0usize;
    let mut pixels = [0u8; 16];

    for y in (0..rows).step_by(4) {
        for x in (0..width).step_by(4) {
            let block_idx = ((y * width) >> 4) + (x >> 2);
            let block_idx = block_idx as usize;

            let min_val = min_stream[block_idx];
            let dist = dist_stream[block_idx];

            // Look up bits and bucket_size from LUT
            let (bits, bucket_size) = bucket_lut[dist as usize];

            // Add back light lost during quantization
            let min_adjusted = min_val.saturating_add(bucket_size >> 1);

            if bits == 0 {
                // All pixels in this block are the same value
                for yy in 0..4 {
                    let row_start = ((y + yy) * bytes_per_line + x) as usize;
                    for xx in 0..4 {
                        dst_graymap[row_start + xx] = min_adjusted;
                    }
                }
            } else {
                // Unpack pixel indices
                let bytes_consumed = unpack_pixels(&pixel_stream[pixel_pos..], bits, &mut pixels);
                pixel_pos += bytes_consumed;

                // Reconstruct pixel values
                for yy in 0..4u32 {
                    let row_start = ((y + yy) * bytes_per_line + x) as usize;
                    for xx in 0..4u32 {
                        let idx = (yy * 4 + xx) as usize;
                        let pixel_val = if bucket_size == 1 {
                            min_adjusted.saturating_add(pixels[idx])
                        } else {
                            let scaled = (pixels[idx] as u16) * (bucket_size as u16);
                            (scaled as u8).saturating_add(min_adjusted)
                        };
                        dst_graymap[row_start + xx as usize] = pixel_val;
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bucket_lut_error_limit_2() {
        let lut = generate_bucket_lut(2);

        // dist=0: 1 bucket (0 bits), bucket_size=1
        assert_eq!(lut[0], (0, 1));

        // dist=2: 1 bucket (0 bits), bucket_size=3 (3 <= 3)
        assert_eq!(lut[2], (0, 3));

        // dist=3: 2 buckets (1 bit), bucket_size=2 (ceil(4/2)=2 <= 3)
        assert_eq!(lut[3], (1, 2));

        // dist=5: 2 buckets (1 bit), bucket_size=3 (ceil(6/2)=3 <= 3)
        assert_eq!(lut[5], (1, 3));

        // dist=6: 4 buckets (2 bits), bucket_size=2 (ceil(7/4)=2 <= 3)
        assert_eq!(lut[6], (2, 2));
    }

    #[test]
    fn test_bucket_lut_error_limit_16() {
        let lut = generate_bucket_lut(16);

        // dist=16: 1 bucket (0 bits), bucket_size=17 (17 <= 17)
        assert_eq!(lut[16], (0, 17));

        // dist=17: 2 buckets (1 bit), bucket_size=9 (ceil(18/2)=9 <= 17)
        assert_eq!(lut[17], (1, 9));

        // dist=255: should need fewer bits than error_limit=2
        let (bits, bucket_size) = lut[255];
        assert!(bucket_size <= 17);
        assert!(bits <= 8);
    }

    #[test]
    fn test_unpack_pixels_8bit() {
        let src: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let mut result = [0u8; 16];
        let consumed = unpack_pixels(&src, 8, &mut result);

        assert_eq!(consumed, 16);
        assert_eq!(result, src);
    }

    #[test]
    fn test_unpack_pixels_4bit() {
        // Each byte holds 2 pixels (low nibble at index N, high nibble at index N+8)
        let src: [u8; 8] = [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE];
        let mut result = [0u8; 16];
        let consumed = unpack_pixels(&src, 4, &mut result);

        assert_eq!(consumed, 8);
        // Low nibbles go to indices 0-7
        assert_eq!(result[0], 0x0);
        assert_eq!(result[1], 0x2);
        assert_eq!(result[2], 0x4);
        assert_eq!(result[3], 0x6);
        assert_eq!(result[4], 0x8);
        assert_eq!(result[5], 0xA);
        assert_eq!(result[6], 0xC);
        assert_eq!(result[7], 0xE);
        // High nibbles go to indices 8-15
        assert_eq!(result[8], 0x1);
        assert_eq!(result[9], 0x3);
        assert_eq!(result[10], 0x5);
        assert_eq!(result[11], 0x7);
        assert_eq!(result[12], 0x9);
        assert_eq!(result[13], 0xB);
        assert_eq!(result[14], 0xD);
        assert_eq!(result[15], 0xF);
    }

    /// Test that pack_pixels and unpack_pixels are inverses of each other
    #[test]
    fn test_pack_unpack_roundtrip() {
        // Test all bit depths with various patterns
        for bits in [1u8, 2, 3, 4, 5, 6, 8] {
            let max_val = if bits == 8 { 255u8 } else { (1u8 << bits) - 1 };

            // Test with sequential values
            let indices: [u8; 16] =
                std::array::from_fn(|i| (i as u8) % (max_val.saturating_add(1)));

            // Pack
            let mut packed = [0u8; 16];
            let bytes_written = pack_pixels(&indices, bits, &mut packed);

            // Unpack
            let mut unpacked = [0u8; 16];
            let bytes_read = unpack_pixels(&packed, bits, &mut unpacked);

            assert_eq!(
                bytes_written, bytes_read,
                "Bytes mismatch for {} bits",
                bits
            );
            assert_eq!(
                indices, unpacked,
                "Roundtrip failed for {} bits: {:?} != {:?}",
                bits, indices, unpacked
            );
        }
    }

    /// Test pack/unpack with all zeros
    #[test]
    fn test_pack_unpack_zeros() {
        for bits in [1u8, 2, 3, 4, 5, 6, 8] {
            let indices = [0u8; 16];

            let mut packed = [0u8; 16];
            let bytes_written = pack_pixels(&indices, bits, &mut packed);

            let mut unpacked = [0u8; 16];
            unpack_pixels(&packed, bits, &mut unpacked);

            assert_eq!(indices, unpacked, "Zero roundtrip failed for {} bits", bits);
            assert!(bytes_written > 0 || bits == 0);
        }
    }

    /// Test pack/unpack with max values
    #[test]
    fn test_pack_unpack_max_values() {
        for bits in [1u8, 2, 3, 4, 5, 6, 8] {
            let max_val = if bits == 8 { 255u8 } else { (1u8 << bits) - 1 };
            let indices = [max_val; 16];

            let mut packed = [0u8; 16];
            pack_pixels(&indices, bits, &mut packed);

            let mut unpacked = [0u8; 16];
            unpack_pixels(&packed, bits, &mut unpacked);

            assert_eq!(
                indices, unpacked,
                "Max value roundtrip failed for {} bits",
                bits
            );
        }
    }

    /// Test compression and decompression round-trip
    #[test]
    fn test_compress_decompress_roundtrip() {
        // Create a simple 8x8 gradient image
        let width = 8u32;
        let height = 8u32;
        let mut image: Vec<u8> = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                image.push(((x + y * 16) % 256) as u8);
            }
        }

        // Test with different quality levels
        for error_limit in [2u8, 4, 8, 16] {
            let compressed =
                compress_tile_block(&image, width, height, width, error_limit, false, false)
                    .expect("Compression failed");

            // Verify header
            assert_eq!(
                compressed[0] & 0x7f,
                error_limit,
                "Error limit mismatch in header"
            );
            assert_eq!(
                compressed[0] & 0x80,
                0,
                "Compressed header flag should be 0"
            );

            // Decompress
            let mut decompressed = vec![0u8; (width * height) as usize];
            decompress_tile_block(&compressed, width, height, width, &mut decompressed)
                .expect("Decompression failed");

            // Verify pixel error is within bounds
            for i in 0..image.len() {
                let diff = (image[i] as i32 - decompressed[i] as i32).abs();
                assert!(
                    diff <= error_limit as i32,
                    "Pixel {} error {} exceeds limit {} (original={}, decompressed={})",
                    i,
                    diff,
                    error_limit,
                    image[i],
                    decompressed[i]
                );
            }
        }
    }

    /// Test compression with uniform blocks (dist=0)
    #[test]
    fn test_compress_uniform_block() {
        let width = 4u32;
        let height = 4u32;
        let image = vec![128u8; (width * height) as usize];

        let compressed = compress_tile_block(&image, width, height, width, 16, false, false)
            .expect("Compression failed");

        // For a uniform block, we should have minimal pixel data
        // Header (1 byte) + min stream (1 byte) + dist stream (1 byte) + no pixel data
        assert_eq!(
            compressed.len(),
            3,
            "Uniform block should compress to 3 bytes"
        );

        // Verify dist is 0
        assert_eq!(compressed[2], 0, "Dist should be 0 for uniform block");

        // Decompress and verify
        let mut decompressed = vec![0u8; (width * height) as usize];
        decompress_tile_block(&compressed, width, height, width, &mut decompressed)
            .expect("Decompression failed");

        // All pixels should be 128 (or very close due to bucket_size adjustment)
        for (i, &pixel) in decompressed.iter().enumerate() {
            let diff = (128i32 - pixel as i32).abs();
            assert!(diff <= 1, "Pixel {} should be ~128, got {}", i, pixel);
        }
    }

    /// Test compression with full range block (dist=255)
    #[test]
    fn test_compress_full_range_block() {
        let width = 4u32;
        let height = 4u32;
        let mut image = vec![0u8; (width * height) as usize];
        image[0] = 0;
        image[15] = 255;
        // Fill rest with gradient
        for (i, pixel) in image.iter_mut().enumerate().take(15).skip(1) {
            *pixel = (i * 17) as u8;
        }

        for error_limit in [2u8, 4, 8, 16] {
            let compressed =
                compress_tile_block(&image, width, height, width, error_limit, false, false)
                    .expect("Compression failed");

            let mut decompressed = vec![0u8; (width * height) as usize];
            decompress_tile_block(&compressed, width, height, width, &mut decompressed)
                .expect("Decompression failed");

            // Verify all pixels are within error bounds
            for i in 0..image.len() {
                let diff = (image[i] as i32 - decompressed[i] as i32).abs();
                assert!(
                    diff <= error_limit as i32 + 1, // +1 for rounding
                    "Pixel {} error {} exceeds limit {} (original={}, decompressed={})",
                    i,
                    diff,
                    error_limit,
                    image[i],
                    decompressed[i]
                );
            }
        }
    }
}
