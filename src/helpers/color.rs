//! Per-pixel color transforms and the row kernels that apply them: SIMD for
//! sixteen pixels at a time where the CPU has it, then the scalar code for the
//! rest of the row.

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use super::color_simd128 as simd;
#[cfg(target_arch = "x86_64")]
use super::color_sse as simd;

/// No SIMD kernels on this target: the scalar code does every pixel.
#[cfg(not(any(
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
)))]
mod simd {
    use super::Transform;

    pub(super) fn available() -> bool {
        false
    }

    pub(super) unsafe fn split3(
        _: Transform,
        _: &[u8],
        _: &mut [u8],
        _: &mut [u8],
        _: &mut [u8],
    ) -> usize {
        0
    }

    pub(super) unsafe fn merge3(_: Transform, _: &[u8], _: &[u8], _: &[u8], _: &mut [u8]) -> usize {
        0
    }

    pub(super) unsafe fn split4(
        _: &[u8],
        _: &mut [u8],
        _: &mut [u8],
        _: &mut [u8],
        _: &mut [u8],
    ) -> usize {
        0
    }

    pub(super) unsafe fn merge4(_: &[u8], _: &[u8], _: &[u8], _: &[u8], _: &mut [u8]) -> usize {
        0
    }
}

#[cfg(test)]
thread_local! {
    /// Makes [`Kernels::detect`] choose the scalar code on this thread, to
    /// compare it with the SIMD kernels.
    pub(super) static SIMD_DISABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// How three planes relate to R, G and B.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Transform {
    /// The planes are R, G and B.
    None,
    /// Reversible YCoCg-R, modulo 256.
    YCoCgR,
    /// Lossy YCoCg from rounded byte averages, with a saturating inverse.
    YCoCg,
}

/// YCoCg-R lifting modulo 256. Co and Cg are stored offset by 128, so
/// `(c >> 1) - 64` is half the signed value rounded down, as in the integer
/// YCoCg-R definition.
#[inline(always)]
pub(super) fn rgb_to_ycocg_r(r: u8, g: u8, b: u8) -> [u8; 3] {
    let co = r.wrapping_sub(b).wrapping_add(128);
    let t = b.wrapping_add(co >> 1).wrapping_sub(64);
    let cg = g.wrapping_sub(t).wrapping_add(128);
    let y = t.wrapping_add(cg >> 1).wrapping_sub(64);
    [y, co, cg]
}

#[inline(always)]
pub(super) fn ycocg_r_to_rgb(y: u8, co: u8, cg: u8) -> [u8; 3] {
    let t = y.wrapping_sub(cg >> 1).wrapping_add(64);
    let g = cg.wrapping_sub(128).wrapping_add(t);
    let b = t.wrapping_sub(co >> 1).wrapping_add(64);
    let r = b.wrapping_add(co).wrapping_sub(128);
    [r, g, b]
}

/// Rounded-up byte average, as `PAVGB`.
#[inline(always)]
fn average(a: u8, b: u8) -> u8 {
    ((u16::from(a) + u16::from(b) + 1) >> 1) as u8
}

/// Lossy YCoCg from rounded byte averages, so nothing wraps. Flooring Y keeps
/// the round-trip error within 1.
#[inline(always)]
pub(super) fn rgb_to_ycocg(r: u8, g: u8, b: u8) -> [u8; 3] {
    let m = average(r, b);
    let y = ((u16::from(g) + u16::from(m)) >> 1) as u8;
    [y, average(r, !b), average(g, !m)]
}

#[inline(always)]
pub(super) fn ycocg_to_rgb(y: u8, co: u8, cg: u8) -> [u8; 3] {
    let saturate = |v: i32| v.clamp(0, 255) as u8;
    let (y, co, cg) = (i32::from(y), i32::from(co) - 128, i32::from(cg) - 128);
    let m = i32::from(saturate(y - cg));
    [saturate(m + co), saturate(y + cg), saturate(m - co)]
}

#[inline(always)]
fn forward(transform: Transform, r: u8, g: u8, b: u8) -> [u8; 3] {
    match transform {
        Transform::None => [r, g, b],
        Transform::YCoCgR => rgb_to_ycocg_r(r, g, b),
        Transform::YCoCg => rgb_to_ycocg(r, g, b),
    }
}

#[inline(always)]
fn inverse(transform: Transform, p0: u8, p1: u8, p2: u8) -> [u8; 3] {
    match transform {
        Transform::None => [p0, p1, p2],
        Transform::YCoCgR => ycocg_r_to_rgb(p0, p1, p2),
        Transform::YCoCg => ycocg_to_rgb(p0, p1, p2),
    }
}

/// Byte shuffle masks of the SIMD kernels.
#[cfg(any(
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
pub(super) mod masks {
    /// A byte shuffle index that clears the byte (`PSHUFB`, `i8x16.swizzle`).
    const ZERO: u8 = 0x80;

    /// Byte shuffle masks gathering channel `c` of 16 packed RGB pixels from their
    /// 16-byte block `k`: `SPLIT_MASKS[c][k]`.
    pub(in crate::helpers) static SPLIT_MASKS: [[[u8; 16]; 3]; 3] = {
        let mut masks = [[[ZERO; 16]; 3]; 3];
        let mut pixel = 0;
        while pixel < 16 {
            let mut channel = 0;
            while channel < 3 {
                let byte = 3 * pixel + channel;
                masks[channel][byte / 16][pixel] = (byte % 16) as u8;
                channel += 1;
            }
            pixel += 1;
        }
        masks
    };

    /// Byte shuffle masks scattering 16 pixels of plane `c` into 16-byte block `k`
    /// of packed RGB pixels: `MERGE_MASKS[c][k]`.
    pub(in crate::helpers) static MERGE_MASKS: [[[u8; 16]; 3]; 3] = {
        let mut masks = [[[ZERO; 16]; 3]; 3];
        let mut byte = 0;
        while byte < 48 {
            masks[byte % 3][byte / 16][byte % 16] = (byte / 3) as u8;
            byte += 1;
        }
        masks
    };

    /// Bytes of four RGBA pixels grouped by channel: R0-R3, G0-G3, B0-B3, A0-A3.
    pub(in crate::helpers) const RGBA_BY_CHANNEL: [u8; 16] =
        [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15];
}

/// The row kernels the CPU supports, chosen once per image.
#[derive(Debug, Clone, Copy)]
pub(super) struct Kernels {
    simd: bool,
}

impl Kernels {
    pub(super) fn detect() -> Self {
        #[cfg(test)]
        if SIMD_DISABLED.with(std::cell::Cell::get) {
            return Self { simd: false };
        }
        Self {
            simd: simd::available(),
        }
    }

    /// Splits a row of RGB pixels into three planes.
    pub(super) fn split3(self, transform: Transform, src: &[u8], [p0, p1, p2]: [&mut [u8]; 3]) {
        // SAFETY: `available` confirmed the instructions.
        let done = if self.simd {
            unsafe { simd::split3(transform, src, p0, p1, p2) }
        } else {
            0
        };
        let pixels = src[done * 3..].chunks_exact(3);
        let planes = p0[done..]
            .iter_mut()
            .zip(&mut p1[done..])
            .zip(&mut p2[done..]);
        for (pixel, ((d0, d1), d2)) in pixels.zip(planes) {
            [*d0, *d1, *d2] = forward(transform, pixel[0], pixel[1], pixel[2]);
        }
    }

    /// Merges a row of three planes into RGB pixels.
    pub(super) fn merge3(self, transform: Transform, [p0, p1, p2]: [&[u8]; 3], dst: &mut [u8]) {
        // SAFETY: `available` confirmed the instructions.
        let done = if self.simd {
            unsafe { simd::merge3(transform, p0, p1, p2, dst) }
        } else {
            0
        };
        let planes = p0[done..].iter().zip(&p1[done..]).zip(&p2[done..]);
        for (pixel, ((&s0, &s1), &s2)) in dst[done * 3..].chunks_exact_mut(3).zip(planes) {
            pixel.copy_from_slice(&inverse(transform, s0, s1, s2));
        }
    }

    /// Splits a row of RGBA pixels into four planes.
    pub(super) fn split4(self, src: &[u8], [p0, p1, p2, p3]: [&mut [u8]; 4]) {
        // SAFETY: `available` confirmed the instructions.
        let done = if self.simd {
            unsafe { simd::split4(src, p0, p1, p2, p3) }
        } else {
            0
        };
        let pixels = src[done * 4..].chunks_exact(4);
        let planes = (p0[done..].iter_mut().zip(&mut p1[done..]))
            .zip(p2[done..].iter_mut().zip(&mut p3[done..]));
        for (pixel, ((d0, d1), (d2, d3))) in pixels.zip(planes) {
            [*d0, *d1, *d2, *d3] = [pixel[0], pixel[1], pixel[2], pixel[3]];
        }
    }

    /// Merges a row of four planes into RGBA pixels.
    pub(super) fn merge4(self, [p0, p1, p2, p3]: [&[u8]; 4], dst: &mut [u8]) {
        // SAFETY: `available` confirmed the instructions.
        let done = if self.simd {
            unsafe { simd::merge4(p0, p1, p2, p3, dst) }
        } else {
            0
        };
        let planes = (p0[done..].iter().zip(&p1[done..])).zip(p2[done..].iter().zip(&p3[done..]));
        for (pixel, ((&s0, &s1), (&s2, &s3))) in dst[done * 4..].chunks_exact_mut(4).zip(planes) {
            pixel.copy_from_slice(&[s0, s1, s2, s3]);
        }
    }
}
