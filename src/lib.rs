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

/// Container format version for compressed streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormatVersion {
    /// Legacy format without a mode byte:
    /// `[3][num_blocks][block-size table or 0x00 entropy marker]...`
    V3 = 3,
    /// Current format: `[4][num_blocks][tile_based][mode][block-size table]...`
    V4 = 4,
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

        // Support format versions 1, 3 and 4.
        //
        // Experimental: v1 has the same on-disk layout as v3 entropy-coded
        // lossless at the header level (version, num_blocks, type_byte, then
        // a u32-LE block-sizes table), so we accept it via the v3 path and
        // let the per-block entropy decoder run as-is. If the per-block
        // payload turns out to have changed between v1 and v3 the decoder
        // will fail with a regular entropy_coder error rather than
        // UnsupportedFormat — we want to know either way.
        if version != 1 && version != 3 && version != 4 {
            return Err(LlicError::UnsupportedFormat);
        }

        // Parse format-specific header fields
        let (tile_based, mut mode, block_sizes_start) = if version >= 4 {
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
            // Format v1 / v3: [version][num_blocks][quality_or_block_size_byte...]
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

            // Legacy streams have no mode byte; tile-based encodes the mode in
            // bit 7 of the first payload byte (set = compressed tile headers =
            // Default, clear = Fast), matching the C++ reference decoder.
            if version < 4 {
                mode = if first_block_byte & 0x80 != 0 {
                    Mode::Default
                } else {
                    Mode::Fast
                };
            }

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

        // Helper closure to decompress a block based on mode and tile_based flag
        let decompress_block =
            |data: &[u8], w: u32, h: u32, bpl: u32, dst: &mut [u8]| -> Result<()> {
                if tile_based {
                    if mode == Mode::Dynamic {
                        lossy::decompress_tile_block_dynamic(data, w, h, bpl, true, dst)
                    } else {
                        lossy::decompress_tile_block(data, w, h, bpl, dst)
                    }
                } else if mode == Mode::Dynamic {
                    entropy_coder::decompress_dynamic(data, w, h, bpl, dst)
                } else {
                    entropy_coder::decompress(data, w, h, bpl, dst)
                }
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

    /// Compress grayscale image data using the current format version (v4).
    ///
    /// Equivalent to calling [`compress_gray8_versioned`](Self::compress_gray8_versioned)
    /// with [`FormatVersion::V4`].
    pub fn compress_gray8(
        &self,
        src_graymap: &[u8],
        quality: Quality,
        mode: Mode,
        dst_data: &mut [u8],
    ) -> Result<usize> {
        self.compress_gray8_versioned(src_graymap, quality, mode, FormatVersion::V4, dst_data)
    }

    /// Compress grayscale image data with an explicit format version.
    ///
    /// # Arguments
    /// * `src_graymap` - Source pixel data (row-major, 8-bit grayscale)
    /// * `quality` - Compression quality level
    /// * `mode` - Compression mode (Default, Fast, Dynamic)
    /// * `format_version` - LLIC container format version
    /// * `dst_data` - Output buffer for compressed data
    ///
    /// # Returns
    /// Number of bytes written to dst_data, or error.
    ///
    /// # Format
    /// - v4: `[version=4][num_blocks=1][tile_based][mode][block_size:u32 LE][data]`
    /// - v3 entropy-coded: `[version=3][num_blocks=1][0x00][block_size:u32 LE][data]`
    /// - v3 tile-based: `[version=3][num_blocks=1][block_size:u32 LE][data]`
    ///
    /// In v3 the entropy-vs-tile discriminator is "byte 2 == 0", so for v3
    /// tile-based the least-significant byte of `block_size` must be non-zero.
    /// When the payload length lands on a multiple of 256, one pad byte is
    /// appended (and counted in `block_size`) to keep the LSB non-zero;
    /// decoders never read it.
    ///
    /// v3 has no mode byte. For tile-based streams the mode is carried in
    /// bit 7 of the first payload byte, so `Default` and `Fast` round-trip;
    /// for entropy-coded streams `Fast` is identical to `Default` and decodes
    /// as `Default`. `Dynamic` is not representable in v3 and returns
    /// `UnsupportedFormat`.
    pub fn compress_gray8_versioned(
        &self,
        src_graymap: &[u8],
        quality: Quality,
        mode: Mode,
        format_version: FormatVersion,
        dst_data: &mut [u8],
    ) -> Result<usize> {
        if format_version == FormatVersion::V3 && mode == Mode::Dynamic {
            return Err(LlicError::UnsupportedFormat);
        }

        let expected_size = self.height as usize * self.bytes_per_line as usize;
        if src_graymap.len() < expected_size {
            return Err(LlicError::InvalidArgument);
        }

        let use_entropy = quality == Quality::LosslessEntropy;
        let error_limit = match quality {
            Quality::Lossless | Quality::LosslessEntropy => 0,
            Quality::VeryHigh => 2,
            Quality::High => 4,
            Quality::Medium => 8,
            Quality::Low => 16,
            Quality::VeryLow => 32,
        };

        // Run the chosen algorithm — block payload is identical across versions.
        let compressed: Vec<u8> = if use_entropy {
            if mode == Mode::Dynamic {
                entropy_coder::compress_dynamic(
                    src_graymap,
                    self.width,
                    self.height,
                    self.bytes_per_line,
                )?
            } else {
                entropy_coder::compress(src_graymap, self.width, self.height, self.bytes_per_line)?
            }
        } else {
            let compress_header = mode != Mode::Fast;
            let use_dynamic_predictor = mode == Mode::Dynamic;
            lossy::compress_tile_block(
                src_graymap,
                self.width,
                self.height,
                self.bytes_per_line,
                error_limit,
                compress_header,
                use_dynamic_predictor,
            )?
        };

        match format_version {
            FormatVersion::V4 => write_v4_block(&compressed, !use_entropy, mode, dst_data),
            FormatVersion::V3 => write_v3_block(&compressed, use_entropy, dst_data),
        }
    }
}

/// Write a single-block v4 container: `[4][1][tile_based][mode][block_size:u32 LE][payload]`.
fn write_v4_block(
    payload: &[u8],
    tile_based: bool,
    mode: Mode,
    dst_data: &mut [u8],
) -> Result<usize> {
    let total = 8 + payload.len();
    if dst_data.len() < total {
        return Err(LlicError::InvalidArgument);
    }
    dst_data[0] = FormatVersion::V4 as u8;
    dst_data[1] = 1;
    dst_data[2] = tile_based as u8;
    dst_data[3] = mode as u8;
    dst_data[4..8].copy_from_slice(&(payload.len() as u32).to_le_bytes());
    dst_data[8..total].copy_from_slice(payload);
    Ok(total)
}

/// Write a single-block legacy v3 container.
///
/// Entropy-coded: `[3][1][0x00][block_size:u32 LE][payload]`.
/// Tile-based: `[3][1][block_size:u32 LE][payload]`. The decoder treats
/// byte 2 == 0 as entropy-coded, so the LSB of block_size must be non-zero:
/// when the payload length is a multiple of 256, one zero pad byte is appended
/// and counted in block_size. Decoders use block sizes only to locate the next
/// block and read tile streams by need, so the pad byte is never read (both in
/// this crate and the C++ reference). The current tile encoder always produces
/// odd payload lengths (the bit writer flushes 16-bit words and pixel data is
/// an even byte count per tile), so the padding is defensive.
fn write_v3_block(payload: &[u8], entropy: bool, dst_data: &mut [u8]) -> Result<usize> {
    if entropy {
        let total = 7 + payload.len();
        if dst_data.len() < total {
            return Err(LlicError::InvalidArgument);
        }
        dst_data[0] = FormatVersion::V3 as u8;
        dst_data[1] = 1;
        dst_data[2] = 0;
        dst_data[3..7].copy_from_slice(&(payload.len() as u32).to_le_bytes());
        dst_data[7..total].copy_from_slice(payload);
        Ok(total)
    } else {
        let padding = usize::from(payload.len().is_multiple_of(256));
        let block_size = (payload.len() + padding) as u32;
        let total = 6 + payload.len() + padding;
        if dst_data.len() < total {
            return Err(LlicError::InvalidArgument);
        }
        dst_data[0] = FormatVersion::V3 as u8;
        dst_data[1] = 1;
        dst_data[2..6].copy_from_slice(&block_size.to_le_bytes());
        dst_data[6..6 + payload.len()].copy_from_slice(payload);
        if padding == 1 {
            dst_data[total - 1] = 0;
        }
        Ok(total)
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

    #[test]
    fn test_dynamic_mode_with_edges() {
        // Test dynamic mode with an image that has sharp edges
        // where MED predictor should outperform average predictor
        let width = 64u32;
        let height = 64u32;
        let mut image = vec![0u8; (width * height) as usize];

        // Create vertical stripes - MED predictor should help here
        for y in 0..height {
            for x in 0..width {
                image[(y * width + x) as usize] = if x % 8 < 4 { 0 } else { 255 };
            }
        }

        let context = LlicContext::new(width, height, width, Some(1)).unwrap();

        // Test with all quality levels (the discriminant is the error limit)
        for quality in [
            Quality::Lossless,
            Quality::VeryHigh,
            Quality::High,
            Quality::Medium,
            Quality::Low,
        ] {
            let mut compressed = vec![0u8; context.compressed_buffer_size()];

            let compressed_size = context
                .compress_gray8(&image, quality, Mode::Dynamic, &mut compressed)
                .unwrap();
            compressed.truncate(compressed_size);

            // Decompress
            let mut output = vec![0u8; (width * height) as usize];
            let (decoded_quality, decoded_mode) =
                context.decompress_gray8(&compressed, &mut output).unwrap();

            assert_eq!(decoded_quality, quality);
            assert_eq!(decoded_mode, Mode::Dynamic);

            // Verify pixels are within error bounds (discriminant is the error limit)
            let error_limit = quality as u8;
            for i in 0..image.len() {
                let diff = (image[i] as i32 - output[i] as i32).abs();
                assert!(
                    diff <= error_limit as i32 + 1, // +1 for rounding
                    "Pixel {} error {} exceeds limit {} for quality {:?}",
                    i,
                    diff,
                    error_limit,
                    quality
                );
            }
        }
    }

    fn gradient_image(width: u32, height: u32) -> Vec<u8> {
        let mut img = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                img.push(((x * 7 + y * 13) & 0xFF) as u8);
            }
        }
        img
    }

    /// Wrap-free ramp (requires width + height <= 128): tile min/dist values
    /// are smooth, so tile-header compression engages, unlike with
    /// `gradient_image` whose `& 0xFF` wraparound makes them noisy.
    fn smooth_image(width: u32, height: u32) -> Vec<u8> {
        assert!(width + height <= 128);
        let mut img = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                img.push((x * 2 + y * 2) as u8);
            }
        }
        img
    }

    #[test]
    fn v3_entropy_roundtrip() {
        let width = 32;
        let height = 16;
        let src = gradient_image(width, height);
        let ctx = LlicContext::new(width, height, width, Some(1)).unwrap();
        let mut buf = vec![0u8; ctx.compressed_buffer_size()];
        let n = ctx
            .compress_gray8_versioned(
                &src,
                Quality::LosslessEntropy,
                Mode::Default,
                FormatVersion::V3,
                &mut buf,
            )
            .expect("v3 entropy compress");
        assert_eq!(buf[0], 3, "version byte");
        assert_eq!(buf[1], 1, "num_blocks");
        assert_eq!(buf[2], 0, "type byte for entropy-coded");

        let mut decoded = vec![0u8; (width * height) as usize];
        let (q, m) = ctx
            .decompress_gray8(&buf[..n], &mut decoded)
            .expect("v3 entropy decompress");
        assert_eq!(q, Quality::LosslessEntropy);
        assert_eq!(m, Mode::Default);
        assert_eq!(decoded, src);
    }

    #[test]
    fn v3_tile_lossless_roundtrip() {
        // Smooth enough that tile-header compression pays off: legacy streams
        // recover the mode from bit 7 of the first payload byte (compressed
        // header = Default), so an image whose header stays uncompressed
        // would faithfully decode as Fast — same as the C++ reference.
        let width = 64;
        let height = 64;
        let src = smooth_image(width, height);
        let ctx = LlicContext::new(width, height, width, Some(1)).unwrap();
        let mut buf = vec![0u8; ctx.compressed_buffer_size()];
        let n = ctx
            .compress_gray8_versioned(
                &src,
                Quality::Lossless,
                Mode::Default,
                FormatVersion::V3,
                &mut buf,
            )
            .expect("v3 tile-based compress");
        assert_eq!(buf[0], 3, "version byte");
        assert_eq!(buf[1], 1, "num_blocks");
        assert_ne!(buf[2], 0, "byte 2 must be non-zero (LSB of block_size)");

        let mut decoded = vec![0u8; (width * height) as usize];
        let (q, m) = ctx
            .decompress_gray8(&buf[..n], &mut decoded)
            .expect("v3 tile-based decompress");
        assert_eq!(q, Quality::Lossless);
        assert_eq!(m, Mode::Default);
        assert_eq!(decoded, src);
    }

    #[test]
    fn v3_tile_fast_roundtrip() {
        let width = 32;
        let height = 16;
        let src = gradient_image(width, height);
        let ctx = LlicContext::new(width, height, width, Some(1)).unwrap();
        let mut buf = vec![0u8; ctx.compressed_buffer_size()];
        let n = ctx
            .compress_gray8_versioned(
                &src,
                Quality::Lossless,
                Mode::Fast,
                FormatVersion::V3,
                &mut buf,
            )
            .expect("v3 fast tile-based compress");
        assert_eq!(buf[0], 3, "version byte");

        let mut decoded = vec![0u8; (width * height) as usize];
        let (q, m) = ctx
            .decompress_gray8(&buf[..n], &mut decoded)
            .expect("v3 fast tile-based decompress");
        assert_eq!(q, Quality::Lossless);
        assert_eq!(
            m,
            Mode::Fast,
            "mode recovered from bit 7 of first payload byte"
        );
        assert_eq!(decoded, src);
    }

    #[test]
    fn v3_rejects_dynamic() {
        let width = 16;
        let height = 16;
        let src = gradient_image(width, height);
        let ctx = LlicContext::new(width, height, width, Some(1)).unwrap();
        let mut buf = vec![0u8; ctx.compressed_buffer_size()];
        assert!(matches!(
            ctx.compress_gray8_versioned(
                &src,
                Quality::Lossless,
                Mode::Dynamic,
                FormatVersion::V3,
                &mut buf
            ),
            Err(LlicError::UnsupportedFormat)
        ));
        assert!(matches!(
            ctx.compress_gray8_versioned(
                &src,
                Quality::LosslessEntropy,
                Mode::Dynamic,
                FormatVersion::V3,
                &mut buf
            ),
            Err(LlicError::UnsupportedFormat)
        ));
    }

    #[test]
    fn write_v3_block_pads_when_payload_length_lsb_is_zero() {
        // The tile encoder currently always produces odd payload lengths, so
        // exercise the defensive pad path at the block-writer level.
        let mut buf = vec![0xAAu8; 600];

        // 256-byte payload: LSB of block_size would be 0, so one pad byte is
        // appended and counted.
        let payload = vec![7u8; 256];
        let n = write_v3_block(&payload, false, &mut buf).unwrap();
        assert_eq!(n, 6 + 257);
        let block_size = u32::from_le_bytes([buf[2], buf[3], buf[4], buf[5]]);
        assert_eq!(block_size, 257, "block size counts the pad byte");
        assert_ne!(buf[2], 0, "byte 2 must be non-zero (LSB of block_size)");
        assert_eq!(buf[n - 1], 0, "pad byte");
        assert_eq!(&buf[6..6 + 256], &payload[..], "payload unchanged");

        // 255-byte payload: no padding needed.
        let payload = vec![7u8; 255];
        let n = write_v3_block(&payload, false, &mut buf).unwrap();
        assert_eq!(n, 6 + 255);
        let block_size = u32::from_le_bytes([buf[2], buf[3], buf[4], buf[5]]);
        assert_eq!(block_size, 255);
    }

    #[test]
    fn v3_tile_decoder_ignores_pad_byte() {
        // Decoders locate blocks by offset and read tile streams by need, so a
        // padded stream (block_size + 1, trailing zero byte) must decode
        // identically — this is the premise the pad path relies on.
        let width = 64;
        let height = 64;
        let src = smooth_image(width, height);
        let ctx = LlicContext::new(width, height, width, Some(1)).unwrap();
        let mut buf = vec![0u8; ctx.compressed_buffer_size()];
        let n = ctx
            .compress_gray8_versioned(
                &src,
                Quality::Lossless,
                Mode::Default,
                FormatVersion::V3,
                &mut buf,
            )
            .expect("v3 tile-based compress");

        let mut padded = buf[..n].to_vec();
        padded.push(0);
        let block_size = u32::from_le_bytes([padded[2], padded[3], padded[4], padded[5]]) + 1;
        padded[2..6].copy_from_slice(&block_size.to_le_bytes());

        let mut decoded = vec![0u8; (width * height) as usize];
        let (q, m) = ctx
            .decompress_gray8(&padded, &mut decoded)
            .expect("padded v3 tile-based decompress");
        assert_eq!(q, Quality::Lossless);
        assert_eq!(m, Mode::Default);
        assert_eq!(decoded, src);
    }

    #[test]
    fn compress_gray8_still_defaults_to_v4() {
        let width = 16;
        let height = 16;
        let src = gradient_image(width, height);
        let ctx = LlicContext::new(width, height, width, Some(1)).unwrap();
        let mut buf = vec![0u8; ctx.compressed_buffer_size()];
        let n = ctx
            .compress_gray8(&src, Quality::LosslessEntropy, Mode::Default, &mut buf)
            .expect("v4 compress");
        assert_eq!(buf[0], 4, "compress_gray8 must keep emitting v4");
        // Sanity round-trip
        let mut decoded = vec![0u8; (width * height) as usize];
        ctx.decompress_gray8(&buf[..n], &mut decoded).unwrap();
        assert_eq!(decoded, src);
    }
}
