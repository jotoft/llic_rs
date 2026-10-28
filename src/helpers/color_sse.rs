//! SSSE3 row kernels for the color helpers, after the C++ reference's SSE
//! paths: sixteen pixels at a time, with the same results as the scalar code.
//!
//! RGB rows are split and merged with one `PSHUFB` per plane and 16-byte
//! block, each mask placing its bytes at their final position (the C++ code
//! shifts them there instead). The kernels need SSSE3, detected at run time by
//! [`available`].

use std::arch::x86_64::*;

use super::color::masks::{MERGE_MASKS, RGBA_BY_CHANNEL, SPLIT_MASKS};
use super::color::Transform;

/// Whether the CPU supports the instructions these kernels use.
pub(super) fn available() -> bool {
    is_x86_feature_detected!("ssse3")
}

const NONE: u8 = 0;
const YCOCG_R: u8 = 1;
const YCOCG: u8 = 2;

#[inline]
#[target_feature(enable = "ssse3")]
unsafe fn load(bytes: &[u8]) -> __m128i {
    debug_assert!(bytes.len() >= 16);
    _mm_loadu_si128(bytes.as_ptr().cast())
}

#[inline]
#[target_feature(enable = "ssse3")]
unsafe fn store(bytes: &mut [u8], value: __m128i) {
    debug_assert!(bytes.len() >= 16);
    _mm_storeu_si128(bytes.as_mut_ptr().cast(), value);
}

/// SSE has no byte shift: shift 16-bit lanes and clear the bit carried in from
/// the neighboring byte.
#[inline]
#[target_feature(enable = "ssse3")]
unsafe fn shr1(v: __m128i) -> __m128i {
    _mm_and_si128(_mm_srli_epi16(v, 1), _mm_set1_epi8(0x7F))
}

/// The three channels of 16 packed RGB pixels (48 bytes).
#[inline]
#[target_feature(enable = "ssse3")]
unsafe fn load_rgb(block: &[u8]) -> [__m128i; 3] {
    let bytes = [load(block), load(&block[16..]), load(&block[32..])];
    std::array::from_fn(|channel| {
        let masks = &SPLIT_MASKS[channel];
        let part = |k: usize| _mm_shuffle_epi8(bytes[k], load(&masks[k]));
        _mm_or_si128(_mm_or_si128(part(0), part(1)), part(2))
    })
}

/// Stores 16 pixels of three channels as packed RGB (48 bytes).
#[inline]
#[target_feature(enable = "ssse3")]
unsafe fn store_rgb(channels: [__m128i; 3], block: &mut [u8]) {
    for (k, out) in block.chunks_exact_mut(16).enumerate() {
        let part =
            |channel: usize| _mm_shuffle_epi8(channels[channel], load(&MERGE_MASKS[channel][k]));
        store(out, _mm_or_si128(_mm_or_si128(part(0), part(1)), part(2)));
    }
}

/// See `color::rgb_to_ycocg_r`.
#[inline]
#[target_feature(enable = "ssse3")]
unsafe fn rgb_to_ycocg_r([r, g, b]: [__m128i; 3]) -> [__m128i; 3] {
    let bias = _mm_set1_epi8(-128);
    let half = _mm_set1_epi8(64);
    let co = _mm_add_epi8(_mm_sub_epi8(r, b), bias);
    let t = _mm_sub_epi8(_mm_add_epi8(b, shr1(co)), half);
    let cg = _mm_add_epi8(_mm_sub_epi8(g, t), bias);
    let y = _mm_sub_epi8(_mm_add_epi8(t, shr1(cg)), half);
    [y, co, cg]
}

/// See `color::ycocg_r_to_rgb`.
#[inline]
#[target_feature(enable = "ssse3")]
unsafe fn ycocg_r_to_rgb([y, co, cg]: [__m128i; 3]) -> [__m128i; 3] {
    let bias = _mm_set1_epi8(-128);
    let half = _mm_set1_epi8(64);
    let t = _mm_add_epi8(_mm_sub_epi8(y, shr1(cg)), half);
    let g = _mm_add_epi8(_mm_sub_epi8(cg, bias), t);
    let b = _mm_add_epi8(_mm_sub_epi8(t, shr1(co)), half);
    let r = _mm_sub_epi8(_mm_add_epi8(b, co), bias);
    [r, g, b]
}

/// See `color::rgb_to_ycocg`: `(a + b) >> 1` is `!avg(!a, !b)`.
#[inline]
#[target_feature(enable = "ssse3")]
unsafe fn rgb_to_ycocg([r, g, b]: [__m128i; 3]) -> [__m128i; 3] {
    let ones = _mm_set1_epi8(-1);
    let not_m = _mm_xor_si128(_mm_avg_epu8(r, b), ones);
    let y = _mm_xor_si128(_mm_avg_epu8(_mm_xor_si128(g, ones), not_m), ones);
    let co = _mm_avg_epu8(r, _mm_xor_si128(b, ones));
    let cg = _mm_avg_epu8(g, not_m);
    [y, co, cg]
}

/// Adds the signed value `c - 128` to `v` with saturation, from its two
/// halves `positive` and `negative`.
#[inline]
#[target_feature(enable = "ssse3")]
unsafe fn add_offset(v: __m128i, positive: __m128i, negative: __m128i) -> __m128i {
    _mm_subs_epu8(_mm_adds_epu8(v, positive), negative)
}

/// See `color::ycocg_to_rgb`.
#[inline]
#[target_feature(enable = "ssse3")]
unsafe fn ycocg_to_rgb([y, co, cg]: [__m128i; 3]) -> [__m128i; 3] {
    let bias = _mm_set1_epi8(-128);
    let cg_pos = _mm_subs_epu8(cg, bias);
    let cg_neg = _mm_subs_epu8(bias, cg);
    let co_pos = _mm_subs_epu8(co, bias);
    let co_neg = _mm_subs_epu8(bias, co);
    let m = add_offset(y, cg_neg, cg_pos);
    [
        add_offset(m, co_pos, co_neg),
        add_offset(y, cg_pos, cg_neg),
        add_offset(m, co_neg, co_pos),
    ]
}

/// Splits whole 16-pixel blocks of an RGB row into three planes; returns the
/// pixels done.
///
/// # Safety
/// The CPU must support SSSE3.
pub(super) unsafe fn split3(
    transform: Transform,
    src: &[u8],
    p0: &mut [u8],
    p1: &mut [u8],
    p2: &mut [u8],
) -> usize {
    match transform {
        Transform::None => split3_kind::<NONE>(src, p0, p1, p2),
        Transform::YCoCgR => split3_kind::<YCOCG_R>(src, p0, p1, p2),
        Transform::YCoCg => split3_kind::<YCOCG>(src, p0, p1, p2),
    }
}

#[target_feature(enable = "ssse3")]
unsafe fn split3_kind<const KIND: u8>(
    src: &[u8],
    p0: &mut [u8],
    p1: &mut [u8],
    p2: &mut [u8],
) -> usize {
    let blocks = src.chunks_exact(48);
    let planes = p0
        .chunks_exact_mut(16)
        .zip(p1.chunks_exact_mut(16))
        .zip(p2.chunks_exact_mut(16));
    let mut done = 0;
    for (block, ((d0, d1), d2)) in blocks.zip(planes) {
        let rgb = load_rgb(block);
        let [c0, c1, c2] = match KIND {
            YCOCG_R => rgb_to_ycocg_r(rgb),
            YCOCG => rgb_to_ycocg(rgb),
            _ => rgb,
        };
        store(d0, c0);
        store(d1, c1);
        store(d2, c2);
        done += 16;
    }
    done
}

/// Merges whole 16-pixel blocks of three planes into an RGB row; returns the
/// pixels done.
///
/// # Safety
/// The CPU must support SSSE3.
pub(super) unsafe fn merge3(
    transform: Transform,
    p0: &[u8],
    p1: &[u8],
    p2: &[u8],
    dst: &mut [u8],
) -> usize {
    match transform {
        Transform::None => merge3_kind::<NONE>(p0, p1, p2, dst),
        Transform::YCoCgR => merge3_kind::<YCOCG_R>(p0, p1, p2, dst),
        Transform::YCoCg => merge3_kind::<YCOCG>(p0, p1, p2, dst),
    }
}

#[target_feature(enable = "ssse3")]
unsafe fn merge3_kind<const KIND: u8>(p0: &[u8], p1: &[u8], p2: &[u8], dst: &mut [u8]) -> usize {
    let planes = p0
        .chunks_exact(16)
        .zip(p1.chunks_exact(16))
        .zip(p2.chunks_exact(16));
    let mut done = 0;
    for (block, ((s0, s1), s2)) in dst.chunks_exact_mut(48).zip(planes) {
        let planes = [load(s0), load(s1), load(s2)];
        let rgb = match KIND {
            YCOCG_R => ycocg_r_to_rgb(planes),
            YCOCG => ycocg_to_rgb(planes),
            _ => planes,
        };
        store_rgb(rgb, block);
        done += 16;
    }
    done
}

/// Splits whole 16-pixel blocks of an RGBA row into four planes: each 16-byte
/// block is grouped by channel, then the 4x4 matrix of 32-bit groups is
/// transposed. Returns the pixels done.
///
/// # Safety
/// The CPU must support SSSE3.
#[target_feature(enable = "ssse3")]
pub(super) unsafe fn split4(
    src: &[u8],
    p0: &mut [u8],
    p1: &mut [u8],
    p2: &mut [u8],
    p3: &mut [u8],
) -> usize {
    let by_channel = load(&RGBA_BY_CHANNEL);
    let planes = (p0.chunks_exact_mut(16).zip(p1.chunks_exact_mut(16)))
        .zip(p2.chunks_exact_mut(16).zip(p3.chunks_exact_mut(16)));
    let mut done = 0;
    for (block, ((d0, d1), (d2, d3))) in src.chunks_exact(64).zip(planes) {
        let q: [__m128i; 4] =
            std::array::from_fn(|k| _mm_shuffle_epi8(load(&block[16 * k..]), by_channel));
        let rg01 = _mm_unpacklo_epi32(q[0], q[1]);
        let ba01 = _mm_unpackhi_epi32(q[0], q[1]);
        let rg23 = _mm_unpacklo_epi32(q[2], q[3]);
        let ba23 = _mm_unpackhi_epi32(q[2], q[3]);
        store(d0, _mm_unpacklo_epi64(rg01, rg23));
        store(d1, _mm_unpackhi_epi64(rg01, rg23));
        store(d2, _mm_unpacklo_epi64(ba01, ba23));
        store(d3, _mm_unpackhi_epi64(ba01, ba23));
        done += 16;
    }
    done
}

/// Merges whole 16-pixel blocks of four planes into an RGBA row; returns the
/// pixels done.
///
/// # Safety
/// The CPU must support SSSE3.
#[target_feature(enable = "ssse3")]
pub(super) unsafe fn merge4(p0: &[u8], p1: &[u8], p2: &[u8], p3: &[u8], dst: &mut [u8]) -> usize {
    let planes = (p0.chunks_exact(16).zip(p1.chunks_exact(16)))
        .zip(p2.chunks_exact(16).zip(p3.chunks_exact(16)));
    let mut done = 0;
    for (block, ((s0, s1), (s2, s3))) in dst.chunks_exact_mut(64).zip(planes) {
        let (r, g, b, a) = (load(s0), load(s1), load(s2), load(s3));
        let rg_lo = _mm_unpacklo_epi8(r, g);
        let rg_hi = _mm_unpackhi_epi8(r, g);
        let ba_lo = _mm_unpacklo_epi8(b, a);
        let ba_hi = _mm_unpackhi_epi8(b, a);
        store(block, _mm_unpacklo_epi16(rg_lo, ba_lo));
        store(&mut block[16..], _mm_unpackhi_epi16(rg_lo, ba_lo));
        store(&mut block[32..], _mm_unpacklo_epi16(rg_hi, ba_hi));
        store(&mut block[48..], _mm_unpackhi_epi16(rg_hi, ba_hi));
        done += 16;
    }
    done
}
