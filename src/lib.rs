use thiserror::Error;

pub const API_VERSION_MAJOR: u32 = 3;
pub const API_VERSION_MINOR: u32 = 0;

/// Format version for compressed streams
pub const FORMAT_VERSION: u8 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quality {
    /// Lossless compression (tile-based, error_limit=0)
    Lossless = 0,
    /// Near-lossless, error limit ±2
    VeryHigh = 2,
    /// High quality, error limit ±4
    High = 4,
    /// Medium quality, error limit ±8
    Medium = 8,
    /// Low quality, error limit ±16
    Low = 16,
    /// Very low quality, error limit ±32
    VeryLow = 32,
    /// Legacy entropy-coded lossless (for backward compatibility)
    LosslessEntropy = 64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Default = 0,
    Fast = 1,
    Dynamic = 2,
}

#[derive(Error, Debug)]
pub enum LlicError {
    #[error("Out of memory")]
    OutOfMemory,
    #[error("Invalid image dimensions (must be multiple of 4)")]
    ImageDimensions,
    #[error("Unsupported format")]
    UnsupportedFormat,
    #[error("Invalid argument")]
    InvalidArgument,
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid compressed data")]
    InvalidData,
    #[error("File I/O error")]
    FileIO,
}

pub type Result<T> = std::result::Result<T, LlicError>;

pub struct LlicContext {
    width: u32,
    height: u32,
    bytes_per_line: u32,
    #[allow(dead_code)] // Will be used for multi-threaded compression
    num_threads: usize,
}

impl LlicContext {
    pub fn new(
        width: u32,
        height: u32,
        bytes_per_line: u32,
        num_threads: Option<usize>,
    ) -> Result<Self> {
        if !width.is_multiple_of(4) || !height.is_multiple_of(4) {
            return Err(LlicError::ImageDimensions);
        }

        if width == 0 || height == 0 {
            return Err(LlicError::InvalidArgument);
        }

        #[cfg(feature = "std")]
        let num_threads = num_threads.unwrap_or_else(num_cpus::get);
        #[cfg(not(feature = "std"))]
        let num_threads = num_threads.unwrap_or(1);

        Ok(Self {
            width,
            height,
            bytes_per_line,
            num_threads,
        })
    }

    pub fn compressed_buffer_size(&self) -> usize {
        ((self.width as usize * self.height as usize * 5) / 4) + 16384
    }

    pub fn decompress_gray8(
        &self,
        src_data: &[u8],
        dst_graymap: &mut [u8],
    ) -> Result<(Quality, Mode)> {
        if src_data.len() < 4 {
            return Err(LlicError::InvalidData);
        }

        // Parse header
        let version = src_data[0];
        let num_blocks = src_data[1];

        // Support format versions 3 and 4
        if version != 3 && version != 4 {
            return Err(LlicError::UnsupportedFormat);
        }

        // Parse format-specific header fields
        let (tile_based, mode, block_sizes_start) = if version >= 4 {
            // Format v4: [version][num_blocks][tile_based][mode][block_sizes...]
            let tile_based = src_data[2] != 0;
            let mode = match src_data[3] {
                0 => Mode::Default,
                1 => Mode::Fast,
                2 => Mode::Dynamic,
                _ => Mode::Default,
            };
            (tile_based, mode, 4usize)
        } else {
            // Format v3: [version][num_blocks][quality_or_block_size_byte...]
            // If byte 2 is 0, it's entropy-coded lossless (tile_based = false)
            // If byte 2 is non-0, it's tile-based (byte 2 is first byte of block sizes)
            let is_entropy_lossless = src_data[2] == 0;
            let tile_based = !is_entropy_lossless;
            let block_sizes_start = if is_entropy_lossless { 3 } else { 2 };
            (tile_based, Mode::Default, block_sizes_start)
        };

        // Read block size table (always little-endian)
        let mut pos = block_sizes_start;
        let mut block_sizes = Vec::with_capacity(num_blocks as usize);
        for _ in 0..num_blocks {
            if pos + 4 > src_data.len() {
                return Err(LlicError::InvalidData);
            }
            let size = u32::from_le_bytes([
                src_data[pos],
                src_data[pos + 1],
                src_data[pos + 2],
                src_data[pos + 3],
            ]);
            block_sizes.push(size);
            pos += 4;
        }

        // Determine quality from block data
        let quality = if tile_based {
            // For tile-based, quality (error_limit) is in the first byte of first block data
            if pos >= src_data.len() {
                return Err(LlicError::InvalidData);
            }
            let first_block_byte = src_data[pos];
            let error_limit = first_block_byte & 0x7f;

            match error_limit {
                0 => Quality::Lossless,
                2 => Quality::VeryHigh,
                4 => Quality::High,
                8 => Quality::Medium,
                16 => Quality::Low,
                32 => Quality::VeryLow,
                _ => Quality::Lossless, // Default fallback for unknown
            }
        } else {
            // Entropy-coded is always lossless
            Quality::LosslessEntropy
        };

        // For formats with multiple blocks, image is divided among threads
        // First, check if we have many small blocks of similar size
        let non_zero_blocks: Vec<_> = block_sizes
            .iter()
            .enumerate()
            .filter(|(_, &size)| size > 0)
            .collect();

        // Choose decompression algorithm based on tile_based flag
        type DecompressFn = fn(&[u8], u32, u32, u32, &mut [u8]) -> Result<()>;
        let decompress_block: DecompressFn = if tile_based {
            |data, w, h, bpl, dst| lossy::decompress_tile_block(data, w, h, bpl, dst)
        } else {
            |data, w, h, bpl, dst| entropy_coder::decompress(data, w, h, bpl, dst)
        };

        if non_zero_blocks.len() > 1 {
            // Multiple blocks with data - image is divided among threads
            // Each thread handles a horizontal stripe of the image

            // Calculate rows per block using the same logic as the C++ code
            let base_block_size = (self.height as usize / num_blocks as usize) / 4 * 4;
            let last_block_size =
                self.height as usize - base_block_size * (num_blocks as usize - 1);

            // Calculate block positions
            let mut block_positions = Vec::new();
            let mut current_pos = pos;
            for size in block_sizes.iter().take(num_blocks as usize) {
                block_positions.push(current_pos);
                current_pos += *size as usize;
            }

            // Decompress each block into its corresponding rows
            let mut row_offset = 0;
            for block_idx in 0..num_blocks as usize {
                let block_size = block_sizes[block_idx];
                if block_size == 0 {
                    continue;
                }

                let block_rows = if block_idx == num_blocks as usize - 1 {
                    last_block_size
                } else {
                    base_block_size
                };

                if block_rows == 0 {
                    continue;
                }

                let block_pos = block_positions[block_idx];
                let block_data = &src_data[block_pos..block_pos + block_size as usize];

                // Decompress this block into a temporary buffer
                let mut block_output = vec![0u8; self.width as usize * block_rows];

                match decompress_block(
                    block_data,
                    self.width,
                    block_rows as u32,
                    self.bytes_per_line,
                    &mut block_output,
                ) {
                    Ok(_) => {
                        // Copy the decompressed rows to the output buffer
                        let src_start = 0;
                        let dst_start = row_offset * self.bytes_per_line as usize;
                        let copy_len = block_rows * self.bytes_per_line as usize;

                        dst_graymap[dst_start..dst_start + copy_len]
                            .copy_from_slice(&block_output[src_start..src_start + copy_len]);

                        row_offset += block_rows;
                    }
                    Err(_e) => {
                        return Err(LlicError::InvalidData);
                    }
                }
            }

            if row_offset != self.height as usize {
                return Err(LlicError::InvalidData);
            }
        } else if let Some((_block_idx, &block_size)) = non_zero_blocks.first() {
            // Single block with data - it should contain the entire image
            let block_data = &src_data[pos..pos + block_size as usize];

            decompress_block(
                block_data,
                self.width,
                self.height,
                self.bytes_per_line,
                dst_graymap,
            )?;
        }

        Ok((quality, mode))
    }

    /// Compress grayscale image data.
    ///
    /// # Arguments
    /// * `src_graymap` - Source pixel data (row-major, 8-bit grayscale)
    /// * `quality` - Compression quality level
    /// * `mode` - Compression mode (Default or Fast)
    /// * `dst_data` - Output buffer for compressed data
    ///
    /// # Returns
    /// Number of bytes written to dst_data, or error.
    ///
    /// # Format
    /// Uses format v4: `[version=4][num_blocks=1][tile_based][mode][block_size:u32 LE][data]`
    /// - `tile_based=1` for all qualities except LosslessEntropy
    /// - `tile_based=0` for LosslessEntropy (legacy entropy-coded path)
    pub fn compress_gray8(
        &self,
        src_graymap: &[u8],
        quality: Quality,
        mode: Mode,
        dst_data: &mut [u8],
    ) -> Result<usize> {
        // Verify source buffer size
        let expected_size = self.height as usize * self.bytes_per_line as usize;
        if src_graymap.len() < expected_size {
            return Err(LlicError::InvalidArgument);
        }

        // Determine if using entropy-coded path (legacy) or tile-based path
        let use_entropy = quality == Quality::LosslessEntropy;

        // Get error_limit for tile-based compression
        let error_limit = match quality {
            Quality::Lossless | Quality::LosslessEntropy => 0,
            Quality::VeryHigh => 2,
            Quality::High => 4,
            Quality::Medium => 8,
            Quality::Low => 16,
            Quality::VeryLow => 32,
        };

        if use_entropy {
            // Legacy entropy-coded lossless path
            let compressed =
                entropy_coder::compress(src_graymap, self.width, self.height, self.bytes_per_line)?;

            // Build LLIC v4 format header for entropy-coded
            // Format: [version=4][num_blocks=1][tile_based=0][mode][block_size:u32 LE][data]
            let header_size = 4 + 4; // 4 header bytes + 1 block size
            let total_size = header_size + compressed.len();

            if dst_data.len() < total_size {
                return Err(LlicError::InvalidArgument);
            }

            // Write header
            dst_data[0] = FORMAT_VERSION;
            dst_data[1] = 1; // num_blocks
            dst_data[2] = 0; // tile_based = false
            dst_data[3] = mode as u8;

            // Block size (little-endian u32)
            let block_size = compressed.len() as u32;
            dst_data[4..8].copy_from_slice(&block_size.to_le_bytes());

            // Copy compressed data
            dst_data[8..total_size].copy_from_slice(&compressed);

            Ok(total_size)
        } else {
            // Tile-based compression (all qualities including lossless)
            // Determine if we should compress headers based on mode
            // Fast mode: no header compression
            // Default/Dynamic: compress headers
            let compress_header = mode != Mode::Fast;

            let compressed = lossy::compress_tile_block(
                src_graymap,
                self.width,
                self.height,
                self.bytes_per_line,
                error_limit,
                compress_header,
            )?;

            // Build LLIC v4 format header for tile-based
            // Format: [version=4][num_blocks=1][tile_based=1][mode][block_size:u32 LE][data]
            let header_size = 4 + 4; // 4 header bytes + 1 block size
            let total_size = header_size + compressed.len();

            if dst_data.len() < total_size {
                return Err(LlicError::InvalidArgument);
            }

            // Write header
            dst_data[0] = FORMAT_VERSION;
            dst_data[1] = 1; // num_blocks
            dst_data[2] = 1; // tile_based = true
            dst_data[3] = mode as u8;

            // Block size (little-endian u32)
            let block_size = compressed.len() as u32;
            dst_data[4..8].copy_from_slice(&block_size.to_le_bytes());

            // Copy compressed data
            dst_data[8..total_size].copy_from_slice(&compressed);

            Ok(total_size)
        }
    }
}

pub mod entropy_coder;
#[cfg(feature = "cpp-reference")]
pub mod ffi;
pub mod lossy;
#[cfg(feature = "std")]
pub mod pgm;
#[cfg(feature = "wasm")]
pub mod wasm;

// Export aliases for convenience
pub use Mode as CompressionMode;
pub use Quality as CompressionQuality;

/// Compute prediction residual for grayscale image data.
///
/// Uses the same predictor as the lossless compression:
/// - First pixel: stored as-is (no prediction)
/// - First row: horizontal prediction (pixel - left)
/// - First column: vertical prediction (pixel - top)
/// - Other pixels: average prediction (pixel - (left + top) / 2)
///
/// Returns residuals mapped to 0-255 where 128 = zero residual.
pub fn compute_prediction_residual(data: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut residual = Vec::with_capacity(width * height);

    for y in 0..height {
        for x in 0..width {
            let pixel = data[y * width + x] as i16;
            let predicted = if y == 0 {
                if x == 0 {
                    0
                } else {
                    data[y * width + x - 1] as i16
                }
            } else if x == 0 {
                data[(y - 1) * width + x] as i16
            } else {
                let left = data[y * width + x - 1] as i16;
                let top = data[(y - 1) * width + x] as i16;
                (left + top) >> 1
            };
            let delta = pixel - predicted;
            // Map signed delta to unsigned: 128 = zero, <128 = negative, >128 = positive
            residual.push((delta + 128).clamp(0, 255) as u8);
        }
    }

    residual
}

/// LLSC file format magic header.
///
/// The file container format follows the original C++ implementation:
/// ```text
/// LLSC\n
/// <width> <height>\n
/// <compressed_size>\n
/// <binary_compressed_data>
/// ```
pub const LLSC_MAGIC: &[u8] = b"LLSC";

// File I/O convenience functions (only available with std feature)
#[cfg(feature = "std")]
pub fn decode_file(input_path: &str, output_path: &str) -> Result<()> {
    use std::fs::File;
    use std::io::Read;

    // Read the LLIC file
    let mut file = File::open(input_path).map_err(|_| LlicError::FileIO)?;
    let mut llic_data = Vec::new();
    file.read_to_end(&mut llic_data)
        .map_err(|_| LlicError::FileIO)?;

    // Parse the simple container format to extract dimensions and compressed data
    let (width, height, compressed_data) = parse_llic_container(&llic_data)?;

    // Create context and decompress
    let context = LlicContext::new(width, height, width, None)?;
    let mut image_data = vec![0u8; (width * height) as usize];
    context.decompress_gray8(&compressed_data, &mut image_data)?;

    // Save as PGM
    pgm::write(output_path, width, height, &image_data)?;

    Ok(())
}

#[cfg(feature = "std")]
pub fn encode_file(
    _input_path: &str,
    _output_path: &str,
    _quality: CompressionQuality,
    _mode: CompressionMode,
) -> Result<()> {
    todo!("Encoding not yet implemented")
}

/// Write compressed data to LLSC file format.
///
/// Format follows the original C++ implementation:
/// ```text
/// LLSC\n
/// <width> <height>\n
/// <compressed_size>\n
/// <binary_compressed_data>
/// ```
#[cfg(feature = "std")]
pub fn write_llic_file(path: &str, width: u32, height: u32, compressed_data: &[u8]) -> Result<()> {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(path).map_err(|_| LlicError::FileIO)?;

    // Write LLSC magic header
    writeln!(file, "LLSC").map_err(|_| LlicError::FileIO)?;
    writeln!(file, "{} {}", width, height).map_err(|_| LlicError::FileIO)?;
    writeln!(file, "{}", compressed_data.len()).map_err(|_| LlicError::FileIO)?;
    file.write_all(compressed_data)
        .map_err(|_| LlicError::FileIO)?;

    Ok(())
}

/// Read compressed data from LLSC file format.
///
/// Supports both formats:
/// - New format with "LLSC" magic header (from C++ llic_compress)
/// - Legacy format without magic header (width height on first line)
#[cfg(feature = "std")]
pub fn read_llic_file(path: &str) -> Result<(u32, u32, Vec<u8>)> {
    use std::fs::File;
    use std::io::Read;

    let mut file = File::open(path).map_err(|_| LlicError::FileIO)?;
    let mut data = Vec::new();
    file.read_to_end(&mut data).map_err(|_| LlicError::FileIO)?;

    parse_llic_container(&data)
}

#[cfg(feature = "std")]
fn parse_llic_container(data: &[u8]) -> Result<(u32, u32, Vec<u8>)> {
    use std::str;

    // Find first newline
    let first_newline = data
        .iter()
        .position(|&b| b == b'\n')
        .ok_or(LlicError::UnsupportedFormat)?;

    let first_line =
        str::from_utf8(&data[..first_newline]).map_err(|_| LlicError::UnsupportedFormat)?;

    // Check if this is the new format with LLSC magic header
    let (width, height, size_line_start) = if first_line == "LLSC" {
        // New format: LLSC\n<width> <height>\n<size>\n<data>
        let second_start = first_newline + 1;
        let second_newline = data[second_start..]
            .iter()
            .position(|&b| b == b'\n')
            .ok_or(LlicError::UnsupportedFormat)?
            + second_start;

        let dims_line = str::from_utf8(&data[second_start..second_newline])
            .map_err(|_| LlicError::UnsupportedFormat)?;

        let parts: Vec<&str> = dims_line.split_whitespace().collect();
        if parts.len() != 2 {
            return Err(LlicError::UnsupportedFormat);
        }

        let width: u32 = parts[0].parse().map_err(|_| LlicError::UnsupportedFormat)?;
        let height: u32 = parts[1].parse().map_err(|_| LlicError::UnsupportedFormat)?;

        (width, height, second_newline + 1)
    } else {
        // Legacy format: <width> <height>\n<size>\n<data>
        let parts: Vec<&str> = first_line.split_whitespace().collect();
        if parts.len() != 2 {
            return Err(LlicError::UnsupportedFormat);
        }

        let width: u32 = parts[0].parse().map_err(|_| LlicError::UnsupportedFormat)?;
        let height: u32 = parts[1].parse().map_err(|_| LlicError::UnsupportedFormat)?;

        (width, height, first_newline + 1)
    };

    // Parse compressed size
    let size_newline = data[size_line_start..]
        .iter()
        .position(|&b| b == b'\n')
        .ok_or(LlicError::UnsupportedFormat)?
        + size_line_start;

    let size_line = str::from_utf8(&data[size_line_start..size_newline])
        .map_err(|_| LlicError::UnsupportedFormat)?;

    let compressed_size: usize = size_line
        .trim()
        .parse()
        .map_err(|_| LlicError::UnsupportedFormat)?;

    // Extract compressed data
    let data_start = size_newline + 1;
    if data_start + compressed_size > data.len() {
        return Err(LlicError::UnsupportedFormat);
    }

    let compressed_data = data[data_start..data_start + compressed_size].to_vec();

    Ok((width, height, compressed_data))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_decompression_v3_entropy() {
        // Test with a known format v3 entropy-coded compressed file
        // This is a 4x4 all-zeros image compressed with old llic
        let compressed_data = vec![
            0x03, 0x01, 0x00, // Header: version=3, blocks=1, quality=0 (entropy-coded)
            0x04, 0x00, 0x00, 0x00, // Block size: 4 bytes
            0x00, 0x00, 0x00, 0x00, // Compressed data: 4 bytes of zeros
        ];

        let context = LlicContext::new(4, 4, 4, Some(1)).unwrap();
        let mut output = vec![0u8; 16];

        let (quality, mode) = context
            .decompress_gray8(&compressed_data, &mut output)
            .unwrap();

        // Format v3 entropy-coded returns LosslessEntropy
        assert_eq!(quality, Quality::LosslessEntropy);
        assert_eq!(mode, Mode::Default);

        // All pixels should be 0
        assert!(output.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_gradient_decompression_v3() {
        // Test decompression of format v3 files (created with old C++ llic)
        let compressed_path = "test_data/gradient_4x4_q0.llic";
        if std::path::Path::new(compressed_path).exists() {
            let file_content = std::fs::read(compressed_path).unwrap();

            // Skip the text header "4 4\n79\n"
            let header_end = file_content
                .windows(1)
                .enumerate()
                .filter(|(_, w)| w[0] == b'\n')
                .nth(1)
                .map(|(i, _)| i + 1)
                .unwrap();

            let compressed_data = &file_content[header_end..];

            let context = LlicContext::new(4, 4, 4, Some(1)).unwrap();
            let mut output = vec![0u8; 16];

            let (quality, mode) = context
                .decompress_gray8(compressed_data, &mut output)
                .unwrap();

            // Format v3 lossless returns LosslessEntropy
            assert_eq!(quality, Quality::LosslessEntropy);
            assert_eq!(mode, Mode::Default);

            // Check if we get the expected gradient
            let expected: Vec<u8> = (0..16).collect();
            if output != expected {
                println!("Expected: {:?}", expected);
                println!("Got:      {:?}", output);
                panic!("Gradient decompression failed!");
            }
        }
    }

    #[test]
    fn test_zeros_decompression_v3() {
        // Test decompressing format v3 all-zeros image
        let compressed_path = "test_data/zeros_4x4_q0.llic";
        if std::path::Path::new(compressed_path).exists() {
            let file_content = std::fs::read(compressed_path).unwrap();

            // Skip the text header
            let header_end = file_content
                .windows(1)
                .enumerate()
                .filter(|(_, w)| w[0] == b'\n')
                .nth(1)
                .map(|(i, _)| i + 1)
                .unwrap();

            let compressed_data = &file_content[header_end..];

            let context = LlicContext::new(4, 4, 4, Some(1)).unwrap();
            let mut output = vec![0u8; 16];

            let (quality, mode) = context
                .decompress_gray8(compressed_data, &mut output)
                .unwrap();

            // Format v3 lossless returns LosslessEntropy
            assert_eq!(quality, Quality::LosslessEntropy);
            assert_eq!(mode, Mode::Default);

            // All pixels should be 0
            assert!(
                output.iter().all(|&x| x == 0),
                "Expected all zeros, got: {:?}",
                output
            );
        }
    }

    #[test]
    fn test_roundtrip_tile_based_lossless() {
        // Test round-trip with new tile-based lossless (format v4)
        let original: Vec<u8> = (0..16).collect(); // 4x4 gradient

        let context = LlicContext::new(4, 4, 4, Some(1)).unwrap();
        let mut compressed = vec![0u8; context.compressed_buffer_size()];

        // Compress with tile-based lossless
        let compressed_size = context
            .compress_gray8(&original, Quality::Lossless, Mode::Default, &mut compressed)
            .unwrap();
        compressed.truncate(compressed_size);

        // Check format v4 header
        assert_eq!(compressed[0], 4); // version
        assert_eq!(compressed[2], 1); // tile_based = true

        // Decompress
        let mut output = vec![0u8; 16];
        let (quality, mode) = context.decompress_gray8(&compressed, &mut output).unwrap();

        assert_eq!(quality, Quality::Lossless);
        assert_eq!(mode, Mode::Default);
        assert_eq!(output, original);
    }

    #[test]
    fn test_roundtrip_entropy_lossless() {
        // Test round-trip with legacy entropy-coded lossless (format v4 with tile_based=0)
        let original: Vec<u8> = (0..16).collect(); // 4x4 gradient

        let context = LlicContext::new(4, 4, 4, Some(1)).unwrap();
        let mut compressed = vec![0u8; context.compressed_buffer_size()];

        // Compress with entropy-coded lossless (legacy)
        let compressed_size = context
            .compress_gray8(
                &original,
                Quality::LosslessEntropy,
                Mode::Default,
                &mut compressed,
            )
            .unwrap();
        compressed.truncate(compressed_size);

        // Check format v4 header
        assert_eq!(compressed[0], 4); // version
        assert_eq!(compressed[2], 0); // tile_based = false

        // Decompress
        let mut output = vec![0u8; 16];
        let (quality, mode) = context.decompress_gray8(&compressed, &mut output).unwrap();

        assert_eq!(quality, Quality::LosslessEntropy);
        assert_eq!(mode, Mode::Default);
        assert_eq!(output, original);
    }

    #[test]
    fn test_all_modes_roundtrip() {
        // Test all three modes with tile-based lossless
        let original: Vec<u8> = (0..64).collect(); // 8x8 gradient

        let context = LlicContext::new(8, 8, 8, Some(1)).unwrap();

        for mode in [Mode::Default, Mode::Fast, Mode::Dynamic] {
            let mut compressed = vec![0u8; context.compressed_buffer_size()];

            let compressed_size = context
                .compress_gray8(&original, Quality::Lossless, mode, &mut compressed)
                .unwrap();
            compressed.truncate(compressed_size);

            // Check format v4 header
            assert_eq!(compressed[0], 4, "version should be 4 for mode {:?}", mode);
            assert_eq!(
                compressed[2], 1,
                "tile_based should be 1 for mode {:?}",
                mode
            );
            assert_eq!(compressed[3], mode as u8, "mode byte mismatch");

            // Decompress
            let mut output = vec![0u8; 64];
            let (quality, decoded_mode) =
                context.decompress_gray8(&compressed, &mut output).unwrap();

            assert_eq!(
                quality,
                Quality::Lossless,
                "quality mismatch for mode {:?}",
                mode
            );
            assert_eq!(decoded_mode, mode, "mode mismatch");
            assert_eq!(output, original, "pixel data mismatch for mode {:?}", mode);
        }
    }

    #[test]
    fn test_mode_compression_differences() {
        // Test that Fast mode produces larger output than Default/Dynamic (for some images)
        // and that Default/Dynamic compress headers
        let mut image = vec![0u8; 64]; // 8x8
                                       // Create a pattern with some structure (compressible headers)
        for (i, pixel) in image.iter_mut().enumerate().take(64) {
            *pixel = ((i / 4) * 17) as u8;
        }

        let context = LlicContext::new(8, 8, 8, Some(1)).unwrap();

        let mut compressed_default = vec![0u8; context.compressed_buffer_size()];
        let mut compressed_fast = vec![0u8; context.compressed_buffer_size()];

        let size_default = context
            .compress_gray8(
                &image,
                Quality::Lossless,
                Mode::Default,
                &mut compressed_default,
            )
            .unwrap();

        let size_fast = context
            .compress_gray8(&image, Quality::Lossless, Mode::Fast, &mut compressed_fast)
            .unwrap();

        // Fast mode should be >= Default mode (no header compression)
        // The difference may be small or zero for very simple images
        assert!(
            size_fast >= size_default,
            "Fast mode should produce >= size than Default (fast={}, default={})",
            size_fast,
            size_default
        );
    }
}
