//! Helpers that turn color images into the 8-bit planes the codec compresses,
//! and back after decompression: a port of the C++ reference's
//! `llic_helpers.h` (API 4.0).
//!
//! The codec compresses one 8-bit plane at a time. To compress a color image,
//! split it into planes with one of the `deinterleave_*` functions, compress
//! each plane with an [`LlicContext`](crate::LlicContext) of the plane's size,
//! and after decompressing merge the planes with the matching `interleave_*`
//! function. [`plane_sizes`] gives the planes of each layout. The outputs equal
//! the C++ helpers' byte for byte.
//!
//! | Planes compressed | Split / merge | Why |
//! |---|---|---|
//! | All lossless | [`deinterleave_rgb_to_ycocg_r`] / [`interleave_ycocg_r_to_rgb`] | Reversible: every color comes back exactly. |
//! | Any lossy | [`deinterleave_rgb_to_ycocg`] / [`interleave_ycocg_to_rgb`] | The inverse saturates, so coding errors stay small. |
//! | Either, keeping RGB | [`deinterleave_rgb`] / [`interleave_rgb`] | No transform; for synthetic images. |
//!
//! YCoCg moves the brightness shared by R, G and B into one plane, Y, and
//! leaves two mostly flat color-difference planes, Co and Cg, that compress
//! better than R, G and B. Never compress YCoCg-R planes lossily: it works
//! modulo 256, so an error near 0 or 255 wraps to the other end of the range.
//!
//! ```
//! use llic::helpers::{self, Rows, RowsMut};
//!
//! let (width, height) = (16, 4);
//! let rgb: Vec<u8> = (0..width * height * 3).map(|i| i as u8).collect();
//! let mut planes = vec![vec![0u8; width * height]; 3];
//! let [y, co, cg] = planes.get_disjoint_mut([0, 1, 2]).unwrap();
//! helpers::deinterleave_rgb_to_ycocg_r(
//!     Rows::packed(&rgb),
//!     width as u32,
//!     height as u32,
//!     [RowsMut::packed(y), RowsMut::packed(co), RowsMut::packed(cg)],
//! )?;
//!
//! // ... compress and decompress each plane ...
//!
//! let mut restored = vec![0u8; rgb.len()];
//! helpers::interleave_ycocg_r_to_rgb(
//!     [Rows::packed(&planes[0]), Rows::packed(&planes[1]), Rows::packed(&planes[2])],
//!     width as u32,
//!     height as u32,
//!     RowsMut::packed(&mut restored),
//! )?;
//! assert_eq!(restored, rgb);
//! # Ok::<(), llic::helpers::BufferError>(())
//! ```

mod color;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod color_simd128;
#[cfg(target_arch = "x86_64")]
mod color_sse;

use std::fmt;
use std::ops::Deref;

use thiserror::Error;

/// How an image is split into planes; see [`plane_sizes`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlaneLayout {
    /// Three full-size planes: R, G and B, or Y, Co and Cg.
    Rgb,
    /// Four full-size planes: R, G, B and A.
    Rgba,
    /// Bayer 2x2 mosaic: R and B at width/2 by height/2, G at width/2 by
    /// height. Width and height must be even.
    Bayer2x2,
    /// RGB-IR 4x4 mosaic: R and B at width/4 by height/2, G at width/2 by
    /// height, IR at width/2 by height/2. Width and height must be multiples
    /// of 4.
    RgbIr4x4,
    /// High and low planes of samples deeper than 8 bits, both full size.
    HighLow,
    /// YCoCg 4:2:0: Y at width by height, Co and Cg at width/2 by height/2.
    /// Width and height must be even.
    YCoCg420,
}

/// The width or height does not fit a layout's pixel pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("{width}x{height} does not fit the {layout:?} layout")]
pub struct DimensionsError {
    pub layout: PlaneLayout,
    pub width: u32,
    pub height: u32,
}

/// A buffer passed to a split or merge helper does not fit the image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum BufferError {
    #[error("{buffer} has {len} bytes but needs {needed}")]
    TooSmall {
        buffer: Buffer,
        len: usize,
        needed: usize,
    },
    #[error("{buffer} stride {stride} is shorter than a row of {row_bytes} bytes")]
    StrideTooShort {
        buffer: Buffer,
        stride: usize,
        row_bytes: usize,
    },
    #[error("image size overflows usize")]
    TooLarge,
}

/// Which buffer a [`BufferError`] is about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Buffer {
    /// The interleaved image a split reads.
    Source,
    /// The interleaved image a merge writes.
    Destination,
    /// Plane `n`, in channel order.
    Plane(usize),
}

impl fmt::Display for Buffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Source => f.write_str("source"),
            Self::Destination => f.write_str("destination"),
            Self::Plane(n) => write!(f, "plane {n}"),
        }
    }
}

/// Size of one plane in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PlaneSize {
    pub width: u32,
    pub height: u32,
}

/// The planes of a layout, in channel order; dereferences to a slice of two to
/// four sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaneSizes {
    sizes: [PlaneSize; 4],
    count: usize,
}

impl Deref for PlaneSizes {
    type Target = [PlaneSize];

    fn deref(&self) -> &[PlaneSize] {
        &self.sizes[..self.count]
    }
}

/// The number and sizes of the planes a layout splits a `width` x `height`
/// image into, in channel order (R, G, B, A or IR; Y, Co, Cg; high, low).
///
/// The codec needs every plane to be a non-zero multiple of 4 in both
/// directions; this checks only what the layout needs.
///
/// # Errors
/// [`DimensionsError`] when the width or height does not fit the layout's
/// pixel pattern.
pub fn plane_sizes(
    layout: PlaneLayout,
    width: u32,
    height: u32,
) -> Result<PlaneSizes, DimensionsError> {
    let size = |width, height| PlaneSize { width, height };
    let full = size(width, height);
    let even = width.is_multiple_of(2) && height.is_multiple_of(2);
    let (sizes, count) = match layout {
        PlaneLayout::Rgb => ([full; 4], 3),
        PlaneLayout::Rgba => ([full; 4], 4),
        PlaneLayout::HighLow => ([full; 4], 2),
        PlaneLayout::Bayer2x2 if even => {
            let quarter = size(width / 2, height / 2);
            (
                [
                    quarter,
                    size(width / 2, height),
                    quarter,
                    PlaneSize::default(),
                ],
                3,
            )
        }
        PlaneLayout::RgbIr4x4 if width.is_multiple_of(4) && height.is_multiple_of(4) => {
            let color = size(width / 4, height / 2);
            let ir = size(width / 2, height / 2);
            ([color, size(width / 2, height), color, ir], 4)
        }
        PlaneLayout::YCoCg420 if even => {
            let chroma = size(width / 2, height / 2);
            ([full, chroma, chroma, PlaneSize::default()], 3)
        }
        PlaneLayout::Bayer2x2 | PlaneLayout::RgbIr4x4 | PlaneLayout::YCoCg420 => {
            return Err(DimensionsError {
                layout,
                width,
                height,
            })
        }
    };
    Ok(PlaneSizes { sizes, count })
}

/// Rows of bytes to read: an interleaved image or one plane. Row `y` starts at
/// byte `y * stride`.
#[derive(Debug, Clone, Copy)]
pub struct Rows<'a> {
    data: &'a [u8],
    /// Bytes from one row to the next; 0 means packed rows.
    stride: usize,
}

impl<'a> Rows<'a> {
    /// Rows packed back to back, as many bytes apart as a row has.
    pub fn packed(data: &'a [u8]) -> Self {
        Self { data, stride: 0 }
    }

    /// Rows `stride` bytes apart, for padded buffers. The stride must be at
    /// least the bytes of one row.
    pub fn with_stride(data: &'a [u8], stride: usize) -> Self {
        Self { data, stride }
    }
}

/// Rows of bytes to write: an interleaved image or one plane. Row `y` starts
/// at byte `y * stride`; bytes between rows are left as they are.
#[derive(Debug)]
pub struct RowsMut<'a> {
    data: &'a mut [u8],
    /// Bytes from one row to the next; 0 means packed rows.
    stride: usize,
}

impl<'a> RowsMut<'a> {
    /// Rows packed back to back, as many bytes apart as a row has.
    pub fn packed(data: &'a mut [u8]) -> Self {
        Self { data, stride: 0 }
    }

    /// Rows `stride` bytes apart, for padded buffers. The stride must be at
    /// least the bytes of one row.
    pub fn with_stride(data: &'a mut [u8], stride: usize) -> Self {
        Self { data, stride }
    }
}

/// Bytes from one row to the next: the stride, or `row_bytes` for packed rows.
fn row_step(stride: usize, row_bytes: usize) -> usize {
    if stride == 0 {
        row_bytes
    } else {
        stride
    }
}

/// Checks that `height` rows of `row_bytes`, `step` bytes apart, fit in the
/// `len` bytes of `buffer`.
fn check_rows(
    buffer: Buffer,
    len: usize,
    step: usize,
    row_bytes: usize,
    height: usize,
) -> Result<(), BufferError> {
    if step < row_bytes {
        return Err(BufferError::StrideTooShort {
            buffer,
            stride: step,
            row_bytes,
        });
    }
    let needed = match height {
        0 => 0,
        _ => (height - 1)
            .checked_mul(step)
            .and_then(|start| start.checked_add(row_bytes))
            .ok_or(BufferError::TooLarge)?,
    };
    if len < needed {
        return Err(BufferError::TooSmall {
            buffer,
            len,
            needed,
        });
    }
    Ok(())
}

/// Pixel dimensions as `usize`, with the bytes of an interleaved row of
/// `channels` bytes per pixel.
fn dimensions(
    width: u32,
    height: u32,
    channels: usize,
) -> Result<(usize, usize, usize), BufferError> {
    let width = usize::try_from(width).map_err(|_| BufferError::TooLarge)?;
    let height = usize::try_from(height).map_err(|_| BufferError::TooLarge)?;
    let row_bytes = width.checked_mul(channels).ok_or(BufferError::TooLarge)?;
    Ok((width, height, row_bytes))
}

/// Calls `row` with each row of `src` (`N` bytes per pixel) and the matching
/// rows of the `N` planes.
fn split_rows<const N: usize>(
    src: Rows,
    width: u32,
    height: u32,
    planes: [RowsMut; N],
    mut row: impl FnMut(&[u8], [&mut [u8]; N]),
) -> Result<(), BufferError> {
    let (width, height, row_bytes) = dimensions(width, height, N)?;
    let src_step = row_step(src.stride, row_bytes);
    check_rows(Buffer::Source, src.data.len(), src_step, row_bytes, height)?;
    let mut planes = planes.map(|plane| (row_step(plane.stride, width), plane.data));
    for (n, (step, data)) in planes.iter().enumerate() {
        check_rows(Buffer::Plane(n), data.len(), *step, width, height)?;
    }
    for y in 0..height {
        let dst = planes
            .each_mut()
            .map(|(step, data)| &mut data[y * *step..][..width]);
        row(&src.data[y * src_step..][..row_bytes], dst);
    }
    Ok(())
}

/// Calls `row` with the rows of the `N` planes and each matching row of `dst`
/// (`N` bytes per pixel).
fn merge_rows<const N: usize>(
    planes: [Rows; N],
    width: u32,
    height: u32,
    dst: RowsMut,
    mut row: impl FnMut([&[u8]; N], &mut [u8]),
) -> Result<(), BufferError> {
    let (width, height, row_bytes) = dimensions(width, height, N)?;
    let dst_step = row_step(dst.stride, row_bytes);
    check_rows(
        Buffer::Destination,
        dst.data.len(),
        dst_step,
        row_bytes,
        height,
    )?;
    let planes = planes.map(|plane| (row_step(plane.stride, width), plane.data));
    for (n, (step, data)) in planes.iter().enumerate() {
        check_rows(Buffer::Plane(n), data.len(), *step, width, height)?;
    }
    for y in 0..height {
        let src = planes.map(|(step, data)| &data[y * step..][..width]);
        row(src, &mut dst.data[y * dst_step..][..row_bytes]);
    }
    Ok(())
}

/// Splits packed RGB pixels into R, G and B planes.
///
/// # Errors
/// [`BufferError`] when a buffer is too small for `width` x `height` pixels or
/// a stride is shorter than a row.
pub fn deinterleave_rgb(
    src: Rows,
    width: u32,
    height: u32,
    planes: [RowsMut; 3],
) -> Result<(), BufferError> {
    let kernels = color::Kernels::detect();
    split_rows(src, width, height, planes, |src, dst| {
        kernels.split3(color::Transform::None, src, dst);
    })
}

/// Merges R, G and B planes into packed RGB pixels; the inverse of
/// [`deinterleave_rgb`].
///
/// # Errors
/// As [`deinterleave_rgb`].
pub fn interleave_rgb(
    planes: [Rows; 3],
    width: u32,
    height: u32,
    dst: RowsMut,
) -> Result<(), BufferError> {
    let kernels = color::Kernels::detect();
    merge_rows(planes, width, height, dst, |src, dst| {
        kernels.merge3(color::Transform::None, src, dst);
    })
}

/// Splits packed RGBA pixels into R, G, B and A planes.
///
/// # Errors
/// As [`deinterleave_rgb`].
pub fn deinterleave_rgba(
    src: Rows,
    width: u32,
    height: u32,
    planes: [RowsMut; 4],
) -> Result<(), BufferError> {
    let kernels = color::Kernels::detect();
    split_rows(src, width, height, planes, |src, dst| {
        kernels.split4(src, dst);
    })
}

/// Merges R, G, B and A planes into packed RGBA pixels; the inverse of
/// [`deinterleave_rgba`].
///
/// # Errors
/// As [`deinterleave_rgb`].
pub fn interleave_rgba(
    planes: [Rows; 4],
    width: u32,
    height: u32,
    dst: RowsMut,
) -> Result<(), BufferError> {
    let kernels = color::Kernels::detect();
    merge_rows(planes, width, height, dst, |src, dst| {
        kernels.merge4(src, dst);
    })
}

/// Converts packed RGB pixels to Y, Co and Cg planes with the reversible
/// YCoCg-R transform, for lossless coding.
///
/// The lifting steps `Co = R - B`, `t = B + (Co >> 1)`, `Cg = G - t`,
/// `Y = t + (Cg >> 1)` run modulo 256, so every color round-trips exactly
/// through [`interleave_ycocg_r_to_rgb`] and each plane stays 8 bits. Co and
/// Cg are stored offset by 128, so grays give 128. Where `R - B` or `G - t`
/// falls outside -128..=127 the values wrap, which may cost compression but
/// never exactness. Use [`deinterleave_rgb_to_ycocg`] for lossy coding.
///
/// # Errors
/// As [`deinterleave_rgb`].
pub fn deinterleave_rgb_to_ycocg_r(
    src: Rows,
    width: u32,
    height: u32,
    planes: [RowsMut; 3],
) -> Result<(), BufferError> {
    let kernels = color::Kernels::detect();
    split_rows(src, width, height, planes, |src, dst| {
        kernels.split3(color::Transform::YCoCgR, src, dst);
    })
}

/// Converts Y, Co and Cg planes from [`deinterleave_rgb_to_ycocg_r`] back to
/// packed RGB pixels, exactly.
///
/// # Errors
/// As [`deinterleave_rgb`].
pub fn interleave_ycocg_r_to_rgb(
    planes: [Rows; 3],
    width: u32,
    height: u32,
    dst: RowsMut,
) -> Result<(), BufferError> {
    let kernels = color::Kernels::detect();
    merge_rows(planes, width, height, dst, |src, dst| {
        kernels.merge3(color::Transform::YCoCgR, src, dst);
    })
}

/// Converts packed RGB pixels to Y, Co and Cg planes with the YCoCg transform,
/// for lossy coding.
///
/// `Y = (R + 2G + B) / 4`, `Co = 128 + (R - B) / 2` and
/// `Cg = 128 + (2G - R - B) / 4`, built from rounded byte averages so that
/// nothing wraps. [`interleave_ycocg_to_rgb`] restores R exactly and G and B
/// within 1, and saturates, so codec errors stay small. Grays map to
/// `(gray, 128, 128)`.
///
/// # Errors
/// As [`deinterleave_rgb`].
pub fn deinterleave_rgb_to_ycocg(
    src: Rows,
    width: u32,
    height: u32,
    planes: [RowsMut; 3],
) -> Result<(), BufferError> {
    let kernels = color::Kernels::detect();
    split_rows(src, width, height, planes, |src, dst| {
        kernels.split3(color::Transform::YCoCg, src, dst);
    })
}

/// Converts Y, Co and Cg planes from [`deinterleave_rgb_to_ycocg`] back to
/// packed RGB pixels, saturating each channel to 0..=255.
///
/// # Errors
/// As [`deinterleave_rgb`].
pub fn interleave_ycocg_to_rgb(
    planes: [Rows; 3],
    width: u32,
    height: u32,
    dst: RowsMut,
) -> Result<(), BufferError> {
    let kernels = color::Kernels::detect();
    merge_rows(planes, width, height, dst, |src, dst| {
        kernels.merge3(color::Transform::YCoCg, src, dst);
    })
}

#[cfg(test)]
mod tests;
