//! WebAssembly SIMD128 residual rows for the entropy encoder: the same results
//! as the scalar and SSE code, sixteen pixels at a time.
//!
//! `PALIGNR` becomes a fixed shuffle, the rounded-down average is `avgr`
//! minus the rounding bit, and the code length sums that SSE takes with
//! `PSADBW` use pairwise widening adds.

use std::arch::wasm32::*;

use super::{med_predictor, single_entry};

/// Bytes `[previous[15], current[0..15]]`: each pixel's left neighbour.
#[inline(always)]
fn shift_in(previous: v128, current: v128) -> v128 {
    i8x16_shuffle::<15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30>(
        previous, current,
    )
}

/// Rounded-down average: `avgr` rounds up, so subtract the dropped bit.
#[inline(always)]
fn average(a: v128, b: v128) -> v128 {
    i8x16_sub(u8x16_avgr(a, b), v128_and(v128_xor(a, b), u8x16_splat(1)))
}

#[inline(always)]
fn load(bytes: &[u8], x: usize) -> v128 {
    let chunk = &bytes[x..x + 16];
    // SAFETY: `chunk` holds 16 bytes; v128_load needs no alignment.
    unsafe { v128_load(chunk.as_ptr().cast()) }
}

#[inline(always)]
fn store(bytes: &mut [u8], x: usize, value: v128) {
    let chunk = &mut bytes[x..x + 16];
    // SAFETY: as in `load`.
    unsafe { v128_store(chunk.as_mut_ptr().cast(), value) }
}

/// First row: horizontal deltas, the first pixel as is.
pub(super) fn first_row(row: &[u8], out: &mut [u8]) {
    let width = row.len();
    let mut previous = u8x16_splat(0);
    let mut x = 0;
    while x + 16 <= width {
        let current = load(row, x);
        store(out, x, i8x16_sub(current, shift_in(previous, current)));
        previous = current;
        x += 16;
    }
    for x in x..width {
        out[x] = row[x].wrapping_sub(if x == 0 { 0 } else { row[x - 1] });
    }
}

/// Residuals from the rounded-down average of left and top (the first pixel
/// predicts from the top: with left equal to top, the average is top).
pub(super) fn average_row(row: &[u8], prev: &[u8], out: &mut [u8]) {
    let width = row.len();
    let mut previous = u8x16_splat(prev[0]);
    let mut x = 0;
    while x + 16 <= width {
        let current = load(row, x);
        let left = shift_in(previous, current);
        store(out, x, i8x16_sub(current, average(left, load(prev, x))));
        previous = current;
        x += 16;
    }
    for x in x..width {
        let left = if x == 0 { prev[0] } else { row[x - 1] };
        let average = ((u16::from(left) + u16::from(prev[x])) >> 1) as u8;
        out[x] = row[x].wrapping_sub(average);
    }
}

/// Code lengths of sixteen residuals added into four 32-bit lanes. Lengths
/// depend on the signed magnitude: a table up to 16, 13 bits beyond.
#[inline(always)]
fn code_length_sums(residuals: v128) -> v128 {
    let lengths = u8x16(2, 3, 3, 5, 5, 7, 7, 7, 7, 9, 9, 9, 9, 9, 9, 9);
    let magnitude = i8x16_abs(residuals);
    let long_code = u8x16_gt(magnitude, u8x16_splat(16));
    let length = i8x16_add(
        i8x16_swizzle(lengths, u8x16_min(magnitude, u8x16_splat(15))),
        v128_and(long_code, u8x16_splat(4)),
    );
    u32x4_extadd_pairwise_u16x8(u16x8_extadd_pairwise_u8x16(length))
}

#[inline(always)]
fn add_lanes(sums: v128) -> u64 {
    u64::from(u32x4_extract_lane::<0>(sums))
        + u64::from(u32x4_extract_lane::<1>(sums))
        + u64::from(u32x4_extract_lane::<2>(sums))
        + u64::from(u32x4_extract_lane::<3>(sums))
}

/// The average and MED residual rows and their coded sizes in bits, in one
/// pass (see `dynamic_residual_rows_sse`).
pub(super) fn dynamic_rows(
    row: &[u8],
    prev: &[u8],
    avg_out: &mut [u8],
    med_out: &mut [u8],
) -> (u64, u64) {
    let width = row.len();
    let mut avg_sums = u32x4_splat(0);
    let mut med_sums = u32x4_splat(0);
    // With left equal to top, both predictors return top for the first pixel.
    let mut previous = u8x16_splat(prev[0]);
    let mut previous_top = previous;
    let mut x = 0;
    while x + 16 <= width {
        let current = load(row, x);
        let top = load(prev, x);
        let left = shift_in(previous, current);
        let top_left = shift_in(previous_top, top);
        previous = current;
        previous_top = top;
        let min_ab = u8x16_min(left, top);
        let max_ab = u8x16_max(left, top);
        let clamped = u8x16_min(u8x16_max(top_left, min_ab), max_ab);
        let med = i8x16_sub(i8x16_add(left, top), clamped);
        let avg_residuals = i8x16_sub(current, average(left, top));
        let med_residuals = i8x16_sub(current, med);
        store(avg_out, x, avg_residuals);
        store(med_out, x, med_residuals);
        avg_sums = i32x4_add(avg_sums, code_length_sums(avg_residuals));
        med_sums = i32x4_add(med_sums, code_length_sums(med_residuals));
        x += 16;
    }
    let mut avg_bits = add_lanes(avg_sums);
    let mut med_bits = add_lanes(med_sums);
    for x in x..width {
        let left = if x == 0 { prev[0] } else { row[x - 1] };
        let top_left = if x == 0 { prev[0] } else { prev[x - 1] };
        let average = ((u16::from(left) + u16::from(prev[x])) >> 1) as u8;
        let avg_residual = row[x].wrapping_sub(average);
        let med_residual = row[x].wrapping_sub(med_predictor(left, prev[x], top_left));
        avg_out[x] = avg_residual;
        med_out[x] = med_residual;
        avg_bits += u64::from(single_entry(avg_residual) & 0x3F);
        med_bits += u64::from(single_entry(med_residual) & 0x3F);
    }
    (avg_bits, med_bits)
}
