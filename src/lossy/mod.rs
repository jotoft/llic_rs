//! Lossy decompression for LLIC tile-based compression.
//!
//! This implements the 4x4 tile-based lossy compression algorithm used by LLIC
//! for quality levels VeryHigh (2), High (4), Medium (8), and Low (16).

use crate::{LlicError, Result};

/// Bucket LUT entry: (bits, bucket_size)
/// - bits: number of bits per pixel index (0, 1, 2, 3, 4, 5, 6, or 8)
/// - bucket_size: quantization step size
type BucketLutEntry = (u8, u8);

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
        let bucket_size = (dist_plus1 + 1) / 2; // ceil division
        if bucket_size <= error_limit_plus1 {
            lut[dist as usize] = (1, bucket_size as u8);
            continue;
        }

        // Try with 4 buckets (2 bits)
        let bucket_size = (dist_plus1 + 3) / 4;
        if bucket_size <= error_limit_plus1 {
            lut[dist as usize] = (2, bucket_size as u8);
            continue;
        }

        // Try with 8 buckets (3 bits)
        let bucket_size = (dist_plus1 + 7) / 8;
        if bucket_size <= error_limit_plus1 {
            lut[dist as usize] = (3, bucket_size as u8);
            continue;
        }

        // Try with 16 buckets (4 bits)
        let bucket_size = (dist_plus1 + 15) / 16;
        if bucket_size <= error_limit_plus1 {
            lut[dist as usize] = (4, bucket_size as u8);
            continue;
        }

        // Try with 32 buckets (5 bits)
        let bucket_size = (dist_plus1 + 31) / 32;
        if bucket_size <= error_limit_plus1 {
            lut[dist as usize] = (5, bucket_size as u8);
            continue;
        }

        // Try with 64 buckets (6 bits)
        let bucket_size = (dist_plus1 + 63) / 64;
        if bucket_size <= error_limit_plus1 {
            lut[dist as usize] = (6, bucket_size as u8);
            continue;
        }

        // Fall back to 256 buckets (8 bits)
        lut[dist as usize] = (8, 1);
    }

    lut
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

            result[1] = (byte0 >> 0) & 0x1;
            result[5] = (byte0 >> 1) & 0x1;
            result[9] = (byte0 >> 2) & 0x1;
            result[13] = (byte0 >> 3) & 0x1;
            result[0] = (byte0 >> 4) & 0x1;
            result[4] = (byte0 >> 5) & 0x1;
            result[8] = (byte0 >> 6) & 0x1;
            result[12] = (byte0 >> 7) & 0x1;

            result[3] = (byte1 >> 0) & 0x1;
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

            result[0] = (byte0 >> 0) & 0x3;
            result[1] = (byte1 >> 0) & 0x3;
            result[2] = (byte2 >> 0) & 0x3;
            result[3] = (byte3 >> 0) & 0x3;

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

            result[0] = ((word0 >> 0) & 0x7) as u8;
            result[4] = ((word0 >> 3) & 0x7) as u8;
            result[8] = ((word0 >> 6) & 0x7) as u8;
            result[12] = ((word0 >> 9) & 0x7) as u8;
            result[2] = ((word0 >> 12) & 0x7) as u8;
            result[6] = ((word0 >> 15) & 0x7) as u8;
            result[10] = ((word0 >> 18) & 0x7) as u8;
            result[14] = ((word0 >> 21) & 0x7) as u8;

            result[1] = ((word1 >> 0) & 0x7) as u8;
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
            result[0] = (src[0] >> 0) & 0xf;
            result[8] = (src[0] >> 4) & 0xf;
            result[1] = (src[1] >> 0) & 0xf;
            result[9] = (src[1] >> 4) & 0xf;
            result[2] = (src[2] >> 0) & 0xf;
            result[10] = (src[2] >> 4) & 0xf;
            result[3] = (src[3] >> 0) & 0xf;
            result[11] = (src[3] >> 4) & 0xf;
            result[4] = (src[4] >> 0) & 0xf;
            result[12] = (src[4] >> 4) & 0xf;
            result[5] = (src[5] >> 0) & 0xf;
            result[13] = (src[5] >> 4) & 0xf;
            result[6] = (src[6] >> 0) & 0xf;
            result[14] = (src[6] >> 4) & 0xf;
            result[7] = (src[7] >> 0) & 0xf;
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

            result[0] = ((word0 >> 0) & 0x1f) as u8;
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

            result[0] = ((word0 >> 0) & 0x3f) as u8;
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
    // For now, use a fixed error_limit based on typical quality levels
    // The actual error_limit should come from the main header
    // We'll detect it from the block structure
    decompress_tile_block_with_error_limit(src_data, width, rows, bytes_per_line, 16, dst_graymap)
}

/// Decompress a tile-based block.
///
/// The block format from C++:
/// - Byte 0: flags/quality (bit 7 = compressed header, bits 0-6 = error_limit)
/// - If uncompressed header (flags & 0x80 == 0):
///   - Bytes 1..: min[num_tiles], dist[num_tiles], pixels[...]
/// - If compressed header (flags & 0x80 == 0x80):
///   - Bytes 1-4: header_size (u32)
///   - Bytes 5..: compressed_header[header_size], pixels[...]
pub fn decompress_tile_block_with_error_limit(
    src_data: &[u8],
    width: u32,
    rows: u32,
    bytes_per_line: u32,
    _error_limit_hint: u8, // Ignored - we read it from block data
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
        // Compressed header - need to decompress min/dist streams
        // For now, return error - we'll implement this if needed
        return Err(LlicError::UnsupportedFormat);
    }

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

/// Core tile decompression loop.
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
}
