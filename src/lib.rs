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
        todo!("Decompression not yet implemented")
    }
    
    pub fn compress_gray8(&self, src_graymap: &[u8], quality: Quality, mode: Mode, dst_data: &mut [u8]) -> Result<usize> {
        todo!("Compression not yet implemented")
    }
}

pub mod pgm;
pub mod entropy_coder;