//! WebAssembly SIMD128 row kernels for the color helpers: the same results as
//! `color_sse.rs`, sixteen pixels at a time. `PSHUFB` becomes `i8x16.swizzle`
//! (indices above 15 clear the byte in both), the unpacks fixed shuffles.
//! Constant two-input `i8x16.shuffle`s for the RGB bytes ran a third slower in
//! V8. Compiled when the target enables `simd128`.

use std::arch::wasm32::*;

use super::color::masks::{MERGE_MASKS, RGBA_BY_CHANNEL, SPLIT_MASKS};
use super::color::Transform;

/// Always true: the module is only compiled with `simd128` enabled.
pub(super) fn available() -> bool {
    true
}

const NONE: u8 = 0;
const YCOCG_R: u8 = 1;
const YCOCG: u8 = 2;

#[inline(always)]
fn load(bytes: &[u8]) -> v128 {
    let chunk = &bytes[..16];
    // SAFETY: `chunk` holds 16 bytes; v128_load needs no alignment.
    unsafe { v128_load(chunk.as_ptr().cast()) }
}

#[inline(always)]
fn store(bytes: &mut [u8], value: v128) {
    let chunk = &mut bytes[..16];
    // SAFETY: as in `load`.
    unsafe { v128_store(chunk.as_mut_ptr().cast(), value) }
}

/// The three channels of 16 packed RGB pixels (48 bytes).
#[inline(always)]
fn load_rgb(block: &[u8]) -> [v128; 3] {
    let bytes = [load(block), load(&block[16..]), load(&block[32..])];
    std::array::from_fn(|channel| {
        let masks = &SPLIT_MASKS[channel];
        let part = |k: usize| i8x16_swizzle(bytes[k], load(&masks[k]));
        v128_or(v128_or(part(0), part(1)), part(2))
    })
}

/// Stores 16 pixels of three channels as packed RGB (48 bytes).
#[inline(always)]
fn store_rgb(channels: [v128; 3], block: &mut [u8]) {
    for (k, out) in block.chunks_exact_mut(16).enumerate() {
        let part =
            |channel: usize| i8x16_swizzle(channels[channel], load(&MERGE_MASKS[channel][k]));
        store(out, v128_or(v128_or(part(0), part(1)), part(2)));
    }
}

/// See `color::rgb_to_ycocg_r`.
#[inline(always)]
fn rgb_to_ycocg_r([r, g, b]: [v128; 3]) -> [v128; 3] {
    let bias = u8x16_splat(128);
    let half = u8x16_splat(64);
    let co = i8x16_add(i8x16_sub(r, b), bias);
    let t = i8x16_sub(i8x16_add(b, u8x16_shr(co, 1)), half);
    let cg = i8x16_add(i8x16_sub(g, t), bias);
    let y = i8x16_sub(i8x16_add(t, u8x16_shr(cg, 1)), half);
    [y, co, cg]
}

/// See `color::ycocg_r_to_rgb`.
#[inline(always)]
fn ycocg_r_to_rgb([y, co, cg]: [v128; 3]) -> [v128; 3] {
    let bias = u8x16_splat(128);
    let half = u8x16_splat(64);
    let t = i8x16_add(i8x16_sub(y, u8x16_shr(cg, 1)), half);
    let g = i8x16_add(i8x16_sub(cg, bias), t);
    let b = i8x16_add(i8x16_sub(t, u8x16_shr(co, 1)), half);
    let r = i8x16_sub(i8x16_add(b, co), bias);
    [r, g, b]
}

/// See `color::rgb_to_ycocg`: `(a + b) >> 1` is `!avgr(!a, !b)`.
#[inline(always)]
fn rgb_to_ycocg([r, g, b]: [v128; 3]) -> [v128; 3] {
    let not_m = v128_not(u8x16_avgr(r, b));
    let y = v128_not(u8x16_avgr(v128_not(g), not_m));
    let co = u8x16_avgr(r, v128_not(b));
    let cg = u8x16_avgr(g, not_m);
    [y, co, cg]
}

/// Adds the signed value `c - 128` to `v` with saturation, from its two
/// halves `positive` and `negative`.
#[inline(always)]
fn add_offset(v: v128, positive: v128, negative: v128) -> v128 {
    u8x16_sub_sat(u8x16_add_sat(v, positive), negative)
}

/// See `color::ycocg_to_rgb`.
#[inline(always)]
fn ycocg_to_rgb([y, co, cg]: [v128; 3]) -> [v128; 3] {
    let bias = u8x16_splat(128);
    let cg_pos = u8x16_sub_sat(cg, bias);
    let cg_neg = u8x16_sub_sat(bias, cg);
    let co_pos = u8x16_sub_sat(co, bias);
    let co_neg = u8x16_sub_sat(bias, co);
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
/// None; `unsafe` matches the SSE version.
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

fn split3_kind<const KIND: u8>(src: &[u8], p0: &mut [u8], p1: &mut [u8], p2: &mut [u8]) -> usize {
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
/// None; `unsafe` matches the SSE version.
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

fn merge3_kind<const KIND: u8>(p0: &[u8], p1: &[u8], p2: &[u8], dst: &mut [u8]) -> usize {
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

/// Splits whole 16-pixel blocks of an RGBA row into four planes; see
/// `color_sse::split4`. Returns the pixels done.
///
/// # Safety
/// None; `unsafe` matches the SSE version.
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
        let q: [v128; 4] =
            std::array::from_fn(|k| i8x16_swizzle(load(&block[16 * k..]), by_channel));
        let rg01 = i32x4_shuffle::<0, 4, 1, 5>(q[0], q[1]);
        let ba01 = i32x4_shuffle::<2, 6, 3, 7>(q[0], q[1]);
        let rg23 = i32x4_shuffle::<0, 4, 1, 5>(q[2], q[3]);
        let ba23 = i32x4_shuffle::<2, 6, 3, 7>(q[2], q[3]);
        store(d0, i64x2_shuffle::<0, 2>(rg01, rg23));
        store(d1, i64x2_shuffle::<1, 3>(rg01, rg23));
        store(d2, i64x2_shuffle::<0, 2>(ba01, ba23));
        store(d3, i64x2_shuffle::<1, 3>(ba01, ba23));
        done += 16;
    }
    done
}

/// Merges whole 16-pixel blocks of four planes into an RGBA row; returns the
/// pixels done.
///
/// # Safety
/// None; `unsafe` matches the SSE version.
pub(super) unsafe fn merge4(p0: &[u8], p1: &[u8], p2: &[u8], p3: &[u8], dst: &mut [u8]) -> usize {
    let planes = (p0.chunks_exact(16).zip(p1.chunks_exact(16)))
        .zip(p2.chunks_exact(16).zip(p3.chunks_exact(16)));
    let mut done = 0;
    for (block, ((s0, s1), (s2, s3))) in dst.chunks_exact_mut(64).zip(planes) {
        let (r, g, b, a) = (load(s0), load(s1), load(s2), load(s3));
        let rg_lo = i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(r, g);
        let rg_hi =
            i8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(r, g);
        let ba_lo = i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(b, a);
        let ba_hi =
            i8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(b, a);
        store(
            block,
            i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(rg_lo, ba_lo),
        );
        store(
            &mut block[16..],
            i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(rg_lo, ba_lo),
        );
        store(
            &mut block[32..],
            i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(rg_hi, ba_hi),
        );
        store(
            &mut block[48..],
            i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(rg_hi, ba_hi),
        );
        done += 16;
    }
    done
}
