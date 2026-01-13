use thiserror::Error;

pub const API_VERSION_MAJOR: u32 = 2;
pub const API_VERSION_MINOR: u32 = 0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quality {
    Lossless = 0,
    VeryHigh = 2,
    High = 4,
    Medium = 8,
    Low = 16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Default,
    Fast,
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
    num_threads: usize,
}

impl LlicContext {
    pub fn new(width: u32, height: u32, bytes_per_line: u32, num_threads: Option<usize>) -> Result<Self> {
        if width % 4 != 0 || height % 4 != 0 {
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
    
    pub fn decompress_gray8(&self, src_data: &[u8], dst_graymap: &mut [u8]) -> Result<(Quality, Mode)> {
        if src_data.len() < 3 {
            return Err(LlicError::InvalidData);
        }
        
        // Parse header
        let version = src_data[0];
        let num_blocks = src_data[1];
        let quality_mode = src_data[2];
        
        if version != 3 {
            return Err(LlicError::UnsupportedFormat);
        }
        
        // Determine if lossy or lossless based on byte 2
        // If byte 2 is 0, it's lossless mode (quality_mode = 0, block sizes start at byte 3)
        // If byte 2 is non-0, it's lossy mode (byte 2 is first byte of block sizes)
        let is_lossless = quality_mode == 0;

        // Read block size table (always little-endian)
        // For lossless: starts at byte 3 (after version, num_blocks, quality_mode=0)
        // For lossy: starts at byte 2 (byte 2 is first byte of first block size)
        let mut pos = if is_lossless { 3 } else { 2 };
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

        // Determine quality and mode
        let (quality, mode) = if is_lossless {
            (Quality::Lossless, Mode::Default)
        } else {
            // For lossy, quality is in the first byte of first block data (bits 0-6 = error_limit)
            // pos now points to start of block data
            if pos >= src_data.len() {
                return Err(LlicError::InvalidData);
            }
            let first_block_byte = src_data[pos];
            let error_limit = first_block_byte & 0x7f;
            let compressed_header = (first_block_byte & 0x80) != 0;

            let quality = match error_limit {
                2 => Quality::VeryHigh,
                4 => Quality::High,
                8 => Quality::Medium,
                16 => Quality::Low,
                _ => Quality::VeryHigh, // Default fallback
            };
            let mode = if compressed_header { Mode::Default } else { Mode::Fast };
            (quality, mode)
        };
        
        // For v3 format with multiple blocks, we need to determine if:
        // 1. Each block contains a portion of the image (rows divided among threads)
        // 2. Only certain blocks contain the full image (excess threads)

        // First, check if we have many small blocks of similar size
        let non_zero_blocks: Vec<_> = block_sizes.iter()
            .enumerate()
            .filter(|(_, &size)| size > 0)
            .collect();

        // Choose decompression algorithm based on mode
        let decompress_block: fn(&[u8], u32, u32, u32, &mut [u8]) -> Result<()> =
            if is_lossless {
                |data, w, h, bpl, dst| entropy_coder::decompress(data, w, h, bpl, dst)
            } else {
                |data, w, h, bpl, dst| lossy::decompress_tile_block(data, w, h, bpl, dst)
            };

        if non_zero_blocks.len() > 1 {
            // Multiple blocks with data - image is divided among threads
            // Each thread handles a horizontal stripe of the image

            // Calculate rows per block using the same logic as the C++ code
            let base_block_size = (self.height as usize / num_blocks as usize) / 4 * 4;
            let last_block_size = self.height as usize - base_block_size * (num_blocks as usize - 1);

            // Calculate block positions
            let mut block_positions = Vec::new();
            let mut current_pos = pos;
            for i in 0..num_blocks as usize {
                block_positions.push(current_pos);
                current_pos += block_sizes[i] as usize;
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
    /// * `quality` - Compression quality (only Lossless currently supported)
    /// * `mode` - Compression mode
    /// * `dst_data` - Output buffer for compressed data
    ///
    /// # Returns
    /// Number of bytes written to dst_data, or error.
    pub fn compress_gray8(&self, src_graymap: &[u8], quality: Quality, _mode: Mode, dst_data: &mut [u8]) -> Result<usize> {
        if quality != Quality::Lossless {
            return Err(LlicError::UnsupportedFormat);
        }

        // Verify source buffer size
        let expected_size = self.height as usize * self.bytes_per_line as usize;
        if src_graymap.len() < expected_size {
            return Err(LlicError::InvalidArgument);
        }

        // Compress using entropy coder
        let compressed = entropy_coder::compress(
            src_graymap,
            self.width,
            self.height,
            self.bytes_per_line,
        )?;

        // Build LLIC v3 format header
        // Format: [version=3][num_blocks=1][quality=0][block_size:u32 LE][compressed_data]
        let header_size = 3 + 4; // version + num_blocks + quality + 1 block size
        let total_size = header_size + compressed.len();

        if dst_data.len() < total_size {
            return Err(LlicError::InvalidArgument);
        }

        // Write header
        dst_data[0] = 3; // version
        dst_data[1] = 1; // num_blocks
        dst_data[2] = quality as u8;

        // Block size (little-endian u32)
        let block_size = compressed.len() as u32;
        dst_data[3..7].copy_from_slice(&block_size.to_le_bytes());

        // Copy compressed data
        dst_data[7..total_size].copy_from_slice(&compressed);

        Ok(total_size)
    }
}

#[cfg(feature = "std")]
pub mod pgm;
pub mod entropy_coder;
#[cfg(feature = "cpp-reference")]
pub mod ffi;
pub mod lossy;
#[cfg(feature = "wasm")]
pub mod wasm;

// Export aliases for convenience
pub use Quality as CompressionQuality;
pub use Mode as CompressionMode;

// File I/O convenience functions (only available with std feature)
#[cfg(feature = "std")]
pub fn decode_file(input_path: &str, output_path: &str) -> Result<()> {
    use std::fs::File;
    use std::io::Read;

    // Read the LLIC file
    let mut file = File::open(input_path).map_err(|_| LlicError::FileIO)?;
    let mut llic_data = Vec::new();
    file.read_to_end(&mut llic_data).map_err(|_| LlicError::FileIO)?;

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
pub fn encode_file(_input_path: &str, _output_path: &str, _quality: CompressionQuality, _mode: CompressionMode) -> Result<()> {
    todo!("Encoding not yet implemented")
}

#[cfg(feature = "std")]
fn parse_llic_container(data: &[u8]) -> Result<(u32, u32, Vec<u8>)> {
    use std::str;

    // Find first newline
    let first_newline = data.iter().position(|&b| b == b'\n')
        .ok_or(LlicError::UnsupportedFormat)?;

    let header_line = str::from_utf8(&data[..first_newline])
        .map_err(|_| LlicError::UnsupportedFormat)?;

    let parts: Vec<&str> = header_line.split_whitespace().collect();
    if parts.len() != 2 {
        return Err(LlicError::UnsupportedFormat);
    }

    let width: u32 = parts[0].parse().map_err(|_| LlicError::UnsupportedFormat)?;
    let height: u32 = parts[1].parse().map_err(|_| LlicError::UnsupportedFormat)?;

    // Find second newline
    let second_start = first_newline + 1;
    let second_newline = data[second_start..].iter().position(|&b| b == b'\n')
        .ok_or(LlicError::UnsupportedFormat)? + second_start;

    let size_line = str::from_utf8(&data[second_start..second_newline])
        .map_err(|_| LlicError::UnsupportedFormat)?;

    let compressed_size: usize = size_line.trim().parse()
        .map_err(|_| LlicError::UnsupportedFormat)?;

    // Extract compressed data
    let data_start = second_newline + 1;
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
    fn test_simple_decompression() {
        // Test with a known compressed file
        // This is a 4x4 all-zeros image compressed with llic
        let compressed_data = vec![
            0x03, 0x01, 0x00, // Header: version=3, blocks=1, quality=0
            0x04, 0x00, 0x00, 0x00, // Block size: 4 bytes
            0x00, 0x00, 0x00, 0x00, // Compressed data: 4 bytes of zeros
        ];
        
        let context = LlicContext::new(4, 4, 4, Some(1)).unwrap();
        let mut output = vec![0u8; 16];
        
        let (quality, mode) = context.decompress_gray8(&compressed_data, &mut output).unwrap();
        
        assert_eq!(quality, Quality::Lossless);
        assert_eq!(mode, Mode::Default);
        
        // All pixels should be 0
        assert!(output.iter().all(|&x| x == 0));
    }
    
    #[test]
    fn test_gradient_decompression() {
        // Create a test for gradient pattern
        // First, let's manually create what we expect the compressed data to be
        // For a 4x4 gradient (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)
        // First row deltas: 0, 1, 1, 1
        // Subsequent rows use predictor
        
        // For now, let's use the actual compressed file we have
        let compressed_path = "test_data/gradient_4x4_q0.llic";
        if std::path::Path::new(compressed_path).exists() {
            let file_content = std::fs::read(compressed_path).unwrap();
            
            // Skip the text header "4 4\n79\n"
            let header_end = file_content.windows(1)
                .enumerate()
                .filter(|(_, w)| w[0] == b'\n')
                .nth(1)
                .map(|(i, _)| i + 1)
                .unwrap();
            
            let compressed_data = &file_content[header_end..];
            
            let context = LlicContext::new(4, 4, 4, Some(1)).unwrap();
            let mut output = vec![0u8; 16];
            
            let (quality, mode) = context.decompress_gray8(compressed_data, &mut output).unwrap();
            
            assert_eq!(quality, Quality::Lossless);
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
    fn test_zeros_decompression() {
        // Test decompressing the all-zeros image
        let compressed_path = "test_data/zeros_4x4_q0.llic";
        if std::path::Path::new(compressed_path).exists() {
            let file_content = std::fs::read(compressed_path).unwrap();
            
            // Skip the text header
            let header_end = file_content.windows(1)
                .enumerate()
                .filter(|(_, w)| w[0] == b'\n')
                .nth(1)
                .map(|(i, _)| i + 1)
                .unwrap();
            
            let compressed_data = &file_content[header_end..];
            
            let context = LlicContext::new(4, 4, 4, Some(1)).unwrap();
            let mut output = vec![0u8; 16];
            
            let (quality, mode) = context.decompress_gray8(compressed_data, &mut output).unwrap();
            
            assert_eq!(quality, Quality::Lossless);
            assert_eq!(mode, Mode::Default);
            
            // All pixels should be 0
            assert!(output.iter().all(|&x| x == 0), "Expected all zeros, got: {:?}", output);
        }
    }
}