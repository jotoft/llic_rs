//! WebAssembly SIMD128 version of the sixteen-row wavefront in
//! `wavefront_sse.rs`, with the same interface and output.
//!
//! `u8x16.avgr` rounds up like `PAVGB`, so the complemented-pixel scheme
//! carries over unchanged; `PALIGNR` and the unpacks become fixed shuffles and
//! `PBLENDVB` a bit select. Compiled when the target enables `simd128`.

use std::arch::wasm32::*;

use super::{reconstruct_parts, WAVE_ROWS};

/// Always true: the module is only compiled with `simd128` enabled.
pub(super) fn available() -> bool {
    #[cfg(test)]
    if tests::DISABLED.with(std::cell::Cell::get) {
        return false;
    }
    true
}

/// Four rounds of interleaving rows i and i + 8 transpose a 16x16 byte matrix.
#[inline(always)]
fn transpose16x16(r: &mut [v128; 16]) {
    for _ in 0..4 {
        let mut t = [u8x16_splat(0); 16];
        for i in 0..8 {
            t[2 * i] = i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(
                r[i],
                r[i + 8],
            );
            t[2 * i + 1] =
                i8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(
                    r[i],
                    r[i + 8],
                );
        }
        *r = t;
    }
}

/// Reconstructs sixteen rows in place; see `wavefront_sse::reconstruct`.
///
/// # Safety
/// None beyond the asserted bounds; `unsafe` matches the SSE version.
///
/// # Panics
/// If `width < 32`, `prev` is shorter than `width` or `rows` than 16 rows.
pub(super) unsafe fn reconstruct(prev: &[u8], rows: &mut [u8], width: usize, med_rows: u32) {
    match med_rows {
        0 => reconstruct_kind::<0>(prev, rows, width, med_rows),
        0xFFFF => reconstruct_kind::<1>(prev, rows, width, med_rows),
        _ => reconstruct_kind::<2>(prev, rows, width, med_rows),
    }
}

/// Kind: 0 = average, 1 = MED, 2 = per-row selection by `med_rows`.
#[inline(always)]
unsafe fn reconstruct_kind<const KIND: u8>(
    prev: &[u8],
    rows: &mut [u8],
    width: usize,
    med_rows: u32,
) {
    assert!(width >= 2 * WAVE_ROWS);
    assert!(prev.len() >= width && rows.len() >= WAVE_ROWS * width);
    let uses_med = |k: usize| (med_rows >> k) & 1 != 0;

    // Leading triangle: row k, pixels [0, 16 - k).
    reconstruct_parts(prev, rows, width, uses_med, |k| (0, WAVE_ROWS - k));

    // Lane k: left = row k at t - k - 1, top-left = row k - 1 at t - k - 1.
    let pixel = |k: usize, x: usize| {
        if k == 0 {
            prev[x]
        } else {
            rows[(k - 1) * width + x]
        }
    };
    let left_init: [u8; 16] = std::array::from_fn(|k| !rows[k * width + WAVE_ROWS - 1 - k]);
    let top_left_init: [u8; 16] = std::array::from_fn(|k| !pixel(k, WAVE_ROWS - 1 - k));
    let lane_mask: [u8; 16] = std::array::from_fn(|k| if uses_med(k) { 0xFF } else { 0 });
    let ones = u8x16_splat(0xFF);
    let mut left = v128_load(left_init.as_ptr().cast());
    let mut top_left = v128_load(top_left_init.as_ptr().cast());
    let med_lanes = v128_load(lane_mask.as_ptr().cast());

    let base = rows.as_mut_ptr();
    let mut t0 = WAVE_ROWS;
    while t0 + WAVE_ROWS <= width {
        // Row k starts at pixel t0 - k: one row minus one pixel apart.
        let mut v: [v128; 16] =
            std::array::from_fn(|k| v128_load(base.add(k * width + t0 - k).cast()));
        transpose16x16(&mut v);
        for (j, residuals) in v.iter_mut().enumerate() {
            // Lane 0 takes its top pixel from the row above: byte 15 of this load.
            let above = v128_xor(v128_load(prev.as_ptr().add(t0 + j - 15).cast()), ones);
            // Bytes [above[15], left[0..15]] (PALIGNR by 15).
            let top = i8x16_shuffle::<15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30>(
                above, left,
            );
            let predictor = if KIND == 0 {
                u8x16_avgr(left, top)
            } else {
                let min_ab = u8x16_min(left, top);
                let max_ab = u8x16_max(left, top);
                let clamped = u8x16_min(u8x16_max(top_left, min_ab), max_ab);
                let med = i8x16_sub(i8x16_add(left, top), clamped);
                if KIND == 1 {
                    med
                } else {
                    v128_bitselect(med, u8x16_avgr(left, top), med_lanes)
                }
            };
            left = i8x16_sub(predictor, *residuals);
            top_left = top;
            *residuals = left;
        }
        transpose16x16(&mut v);
        for (k, pixels) in v.iter().enumerate() {
            v128_store(base.add(k * width + t0 - k).cast(), v128_xor(*pixels, ones));
        }
        t0 += WAVE_ROWS;
    }

    // Trailing part: row k, pixels [t0 - k, width), in row order.
    reconstruct_parts(prev, rows, width, uses_med, |k| (t0 - k, width));
}

#[cfg(test)]
pub(super) mod tests {
    use std::cell::Cell;

    thread_local! {
        /// Makes [`super::available`] report false on this thread, to compare
        /// the wavefront with the per-row code.
        pub(in crate::entropy_coder) static DISABLED: Cell<bool> = const { Cell::new(false) };
    }
}
