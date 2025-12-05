//! Lossless compression using the u8v1 entropy coder.
//!
//! This is the inverse of u8v1_decompress. It takes raw pixel data and produces
//! a compressed bitstream that can be decoded by the decompressor.
//!
//! The encoding process:
//! 1. Apply delta encoding with prediction (same predictor as decoder)
//! 2. Encode delta values using the compression table (pairs of symbols)
//! 3. Pack bits into the output stream

use super::bit_writer::BitWriter;
use super::tables::get_compress_table_entry;

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
    use crate::entropy_coder::decompress;

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
}
