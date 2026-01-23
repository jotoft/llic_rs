//! FFI bindings to the C++ llic library for benchmarking comparisons.

use std::ffi::c_char;
use std::os::raw::c_int;

/// Compression quality levels (matches C++ llic_quality_t)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlicQuality {
    Lossless = 0,
    VeryHigh = 2,
    High = 4,
    Medium = 8,
    Low = 16,
}

/// Compression mode (matches C++ llic_mode_t)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlicMode {
    Default = 0,
    Fast = 1,
    Dynamic = 2,
}

/// Error codes (matches C++ llic_error_t)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlicError {
    None = 1,
    OutOfMemory = 2,
    ImageDimensions = 3,
    UnsupportedFormat = 4,
    InvalidArgument = 5,
}

/// Opaque context handle
#[repr(C)]
pub struct LlicContext {
    _private: [u8; 0],
}

extern "C" {
    pub fn llic_init(
        ctx: *mut *mut LlicContext,
        width: u32,
        height: u32,
        bytes_per_line: u32,
        num_threads: i32,
    ) -> LlicError;

    pub fn llic_destroy(ctx: *mut *mut LlicContext);

    pub fn llic_getLastError(ctx: *mut LlicContext) -> LlicError;

    pub fn llic_getErrorString(error: LlicError) -> *const c_char;

    pub fn llic_compressedBufferSize(ctx: *mut LlicContext) -> usize;

    pub fn llic_compressGray8(
        ctx: *mut LlicContext,
        src_graymap: *const u8,
        quality: LlicQuality,
        mode: LlicMode,
        dst_data: *mut u8,
    ) -> usize;

    pub fn llic_decompressGray8(
        ctx: *mut LlicContext,
        src_data: *const u8,
        dst_graymap: *mut u8,
        quality: *mut LlicQuality,
        mode: *mut LlicMode,
    ) -> usize;
}

/// Safe wrapper around the C++ llic context
pub struct CppLlicContext {
    ctx: *mut LlicContext,
    width: u32,
    height: u32,
}

impl CppLlicContext {
    /// Create a new C++ LLIC context
    pub fn new(width: u32, height: u32, num_threads: i32) -> Result<Self, LlicError> {
        let mut ctx: *mut LlicContext = std::ptr::null_mut();
        let result = unsafe { llic_init(&mut ctx, width, height, width, num_threads) };

        if result != LlicError::None {
            return Err(result);
        }

        Ok(Self { ctx, width, height })
    }

    /// Get the maximum compressed buffer size
    pub fn compressed_buffer_size(&self) -> usize {
        unsafe { llic_compressedBufferSize(self.ctx) }
    }

    /// Compress grayscale image data
    pub fn compress(
        &mut self,
        src: &[u8],
        quality: LlicQuality,
        mode: LlicMode,
    ) -> Result<Vec<u8>, LlicError> {
        let expected_size = (self.width * self.height) as usize;
        if src.len() != expected_size {
            return Err(LlicError::InvalidArgument);
        }

        let max_size = self.compressed_buffer_size();
        let mut dst = vec![0u8; max_size];

        let compressed_size =
            unsafe { llic_compressGray8(self.ctx, src.as_ptr(), quality, mode, dst.as_mut_ptr()) };

        if compressed_size == 0 {
            return Err(unsafe { llic_getLastError(self.ctx) });
        }

        dst.truncate(compressed_size);
        Ok(dst)
    }

    /// Decompress LLIC data
    pub fn decompress(
        &mut self,
        src: &[u8],
    ) -> Result<(Vec<u8>, LlicQuality, LlicMode), LlicError> {
        let output_size = (self.width * self.height) as usize;
        let mut dst = vec![0u8; output_size];
        let mut quality = LlicQuality::Lossless;
        let mut mode = LlicMode::Default;

        let decompressed_size = unsafe {
            llic_decompressGray8(
                self.ctx,
                src.as_ptr(),
                dst.as_mut_ptr(),
                &mut quality,
                &mut mode,
            )
        };

        if decompressed_size == 0 {
            return Err(unsafe { llic_getLastError(self.ctx) });
        }

        Ok((dst, quality, mode))
    }
}

impl Drop for CppLlicContext {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe {
                llic_destroy(&mut self.ctx);
            }
        }
    }
}

// Safety: The C++ context is thread-safe when using separate contexts per thread
unsafe impl Send for CppLlicContext {}
