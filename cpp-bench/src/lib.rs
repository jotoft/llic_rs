//! FFI bindings to the C++ llic library (4.x C API) for benchmarking comparisons.

use std::ffi::c_char;

/// Compression quality levels (matches C++ llic_quality_t)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlicQuality {
    Lossless = 0,
    VeryHigh = 2,
    High = 4,
    Medium = 8,
    Low = 16,
    VeryLow = 32,
    LosslessEntropy = 64,
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
    CorruptData = 6,
    EntropyDisabled = 7,
    HelperDimensions = 8,
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

    /// Version of the C++ source the library was built from (API 4.1).
    pub fn llic_version() -> *const c_char;

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
        src_size: usize,
        dst_graymap: *mut u8,
        dst_size: usize,
        quality: *mut LlicQuality,
        mode: *mut LlicMode,
    ) -> usize;
}

/// Version of the linked C++ library, such as "4.1".
pub fn cpp_version() -> String {
    // SAFETY: llic_version returns a static NUL-terminated string.
    unsafe { std::ffi::CStr::from_ptr(llic_version()) }
        .to_string_lossy()
        .into_owned()
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
                src.len(),
                dst.as_mut_ptr(),
                dst.len(),
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

/// The C++ color helpers (`llic_helpers.h`), run without a helper context:
/// single-threaded and unstaged, which gives the same bytes as with one.
pub mod helpers {
    use super::LlicError;

    #[repr(C)]
    struct ChannelBuffer {
        data: *mut u8,
        stride: isize,
    }

    #[repr(C)]
    struct ConstChannelBuffer {
        data: *const u8,
        stride: isize,
    }

    type Split = unsafe extern "C" fn(
        *mut std::ffi::c_void,
        *const u8,
        isize,
        u32,
        u32,
        *mut ChannelBuffer,
    ) -> LlicError;
    type Merge = unsafe extern "C" fn(
        *mut std::ffi::c_void,
        *const ConstChannelBuffer,
        u32,
        u32,
        *mut u8,
        isize,
    ) -> LlicError;

    extern "C" {
        fn llic_deinterleaveRGB(
            helpers: *mut std::ffi::c_void,
            src: *const u8,
            src_stride: isize,
            width: u32,
            height: u32,
            channels: *mut ChannelBuffer,
        ) -> LlicError;
        fn llic_deinterleaveRGBA(
            helpers: *mut std::ffi::c_void,
            src: *const u8,
            src_stride: isize,
            width: u32,
            height: u32,
            channels: *mut ChannelBuffer,
        ) -> LlicError;
        fn llic_deinterleaveRGBToYCoCgR(
            helpers: *mut std::ffi::c_void,
            src: *const u8,
            src_stride: isize,
            width: u32,
            height: u32,
            channels: *mut ChannelBuffer,
        ) -> LlicError;
        fn llic_deinterleaveRGBToYCoCg(
            helpers: *mut std::ffi::c_void,
            src: *const u8,
            src_stride: isize,
            width: u32,
            height: u32,
            channels: *mut ChannelBuffer,
        ) -> LlicError;
        fn llic_interleaveRGB(
            helpers: *mut std::ffi::c_void,
            channels: *const ConstChannelBuffer,
            width: u32,
            height: u32,
            dst: *mut u8,
            dst_stride: isize,
        ) -> LlicError;
        fn llic_interleaveRGBA(
            helpers: *mut std::ffi::c_void,
            channels: *const ConstChannelBuffer,
            width: u32,
            height: u32,
            dst: *mut u8,
            dst_stride: isize,
        ) -> LlicError;
        fn llic_interleaveYCoCgRToRGB(
            helpers: *mut std::ffi::c_void,
            channels: *const ConstChannelBuffer,
            width: u32,
            height: u32,
            dst: *mut u8,
            dst_stride: isize,
        ) -> LlicError;
        fn llic_interleaveYCoCgToRGB(
            helpers: *mut std::ffi::c_void,
            channels: *const ConstChannelBuffer,
            width: u32,
            height: u32,
            dst: *mut u8,
            dst_stride: isize,
        ) -> LlicError;
    }

    /// A color helper pair: interleaved pixels to planes and back.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ColorHelper {
        Rgb,
        Rgba,
        YCoCgR,
        YCoCg,
    }

    impl ColorHelper {
        /// Channels per pixel, and planes.
        pub fn channels(self) -> usize {
            match self {
                Self::Rgba => 4,
                Self::Rgb | Self::YCoCgR | Self::YCoCg => 3,
            }
        }

        fn functions(self) -> (Split, Merge) {
            match self {
                Self::Rgb => (llic_deinterleaveRGB, llic_interleaveRGB),
                Self::Rgba => (llic_deinterleaveRGBA, llic_interleaveRGBA),
                Self::YCoCgR => (llic_deinterleaveRGBToYCoCgR, llic_interleaveYCoCgRToRGB),
                Self::YCoCg => (llic_deinterleaveRGBToYCoCg, llic_interleaveYCoCgToRGB),
            }
        }

        /// Splits packed pixels into packed planes.
        pub fn split(self, src: &[u8], width: u32, height: u32) -> Result<Vec<Vec<u8>>, LlicError> {
            let size = width as usize * height as usize;
            let mut planes = vec![vec![0u8; size]; self.channels()];
            self.split_into(src, width, height, &mut planes)?;
            Ok(planes)
        }

        /// Splits packed pixels into existing packed planes.
        pub fn split_into(
            self,
            src: &[u8],
            width: u32,
            height: u32,
            planes: &mut [Vec<u8>],
        ) -> Result<(), LlicError> {
            let size = width as usize * height as usize;
            assert_eq!(src.len(), size * self.channels());
            assert_eq!(planes.len(), self.channels());
            assert!(planes.iter().all(|plane| plane.len() == size));
            let mut buffers: Vec<ChannelBuffer> = planes
                .iter_mut()
                .map(|plane| ChannelBuffer {
                    data: plane.as_mut_ptr(),
                    stride: 0,
                })
                .collect();
            let (split, _) = self.functions();
            // SAFETY: the source and planes hold width x height packed pixels.
            let result = unsafe {
                split(
                    std::ptr::null_mut(),
                    src.as_ptr(),
                    0,
                    width,
                    height,
                    buffers.as_mut_ptr(),
                )
            };
            match result {
                LlicError::None => Ok(()),
                error => Err(error),
            }
        }

        /// Merges packed planes into packed pixels.
        pub fn merge(
            self,
            planes: &[Vec<u8>],
            width: u32,
            height: u32,
        ) -> Result<Vec<u8>, LlicError> {
            let mut dst = vec![0u8; width as usize * height as usize * self.channels()];
            self.merge_into(planes, width, height, &mut dst)?;
            Ok(dst)
        }

        /// Merges packed planes into an existing buffer of packed pixels.
        pub fn merge_into(
            self,
            planes: &[Vec<u8>],
            width: u32,
            height: u32,
            dst: &mut [u8],
        ) -> Result<(), LlicError> {
            let size = width as usize * height as usize;
            assert_eq!(planes.len(), self.channels());
            assert!(planes.iter().all(|plane| plane.len() == size));
            assert_eq!(dst.len(), size * self.channels());
            let buffers: Vec<ConstChannelBuffer> = planes
                .iter()
                .map(|plane| ConstChannelBuffer {
                    data: plane.as_ptr(),
                    stride: 0,
                })
                .collect();
            let (_, merge) = self.functions();
            // SAFETY: the planes and destination hold width x height packed pixels.
            let result = unsafe {
                merge(
                    std::ptr::null_mut(),
                    buffers.as_ptr(),
                    width,
                    height,
                    dst.as_mut_ptr(),
                    0,
                )
            };
            match result {
                LlicError::None => Ok(()),
                error => Err(error),
            }
        }
    }
}
