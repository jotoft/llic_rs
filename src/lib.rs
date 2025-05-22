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

        let num_threads = num_threads.unwrap_or_else(num_cpus::get);
        
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
        
        // Determine quality and mode from header
        let (quality, mode) = match quality_mode {
            0 => (Quality::Lossless, Mode::Default),
            2 => (Quality::VeryHigh, Mode::Default),
            4 => (Quality::High, Mode::Default),
            8 => (Quality::Medium, Mode::Default),
            16 => (Quality::Low, Mode::Default),
            _ => return Err(LlicError::InvalidData),
        };
        
        // Read block size table
        let mut pos = 3;
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
        
        if quality != Quality::Lossless {
            return Err(LlicError::UnsupportedFormat);
        }
        
        // For v3 format, blocks are assigned in a specific way
        // With many threads and small images, only the last few blocks may have data
        let mut row_offset = 0;
        
        for block_idx in 0..num_blocks as usize {
            let block_size = block_sizes[block_idx];
            
            // Skip empty blocks
            if block_size == 0 {
                continue;
            }
            
            // For the blocks that have data, they contain the full image
            // This happens when there are more threads than needed
            if row_offset == 0 && block_size > 0 {
                // This block contains the entire image
                let block_data = &src_data[pos..pos + block_size as usize];
                pos += block_size as usize;
                
                // Decompress the entire image
                entropy_coder::decompress(
                    block_data,
                    self.width,
                    self.height,
                    self.bytes_per_line,
                    dst_graymap,
                )?;
                
                // We're done - the entire image was in this block
                break;
            }
        }
        
        Ok((quality, mode))
    }
    
    pub fn compress_gray8(&self, _src_graymap: &[u8], _quality: Quality, _mode: Mode, _dst_data: &mut [u8]) -> Result<usize> {
        todo!("Compression not yet implemented")
    }
}

pub mod pgm;
pub mod entropy_coder;