//! SSE wavefront reconstruction of sixteen rows at once, ported from the C++
//! reference (llic 3.4+, `LLIC_SSE_WAVEFRONT`).
//!
//! Each pixel depends on its left and top neighbours, so a row cannot be
//! vectorised along x. Pixel (x, y) depends only on (x - 1, y) and (x, y - 1),
//! though: at step t, lane k handles row k, pixel t - k, so all sixteen lanes
//! are independent and lane k's top neighbour is lane k - 1 from the previous
//! step. Residuals and pixels are transposed in 16x16 blocks.
//!
//! Pixels are kept complemented: `PAVGB` rounds up, and `!avg(!a, !b)` is the
//! rounded-down average; MED commutes with complementing. A complemented pixel
//! is then the complemented predictor minus the residual, with no correction on
//! the dependency chain.

use std::arch::x86_64::*;

use super::{reconstruct_parts, WAVE_ROWS};

/// Whether the CPU supports the instructions this module uses.
pub(super) fn available() -> bool {
    #[cfg(test)]
    if tests::DISABLED.with(std::cell::Cell::get) {
        return false;
    }
    is_x86_feature_detected!("ssse3") && is_x86_feature_detected!("sse4.1")
}

/// Four rounds of interleaving rows i and i + 8 transpose a 16x16 byte matrix.
#[inline]
#[target_feature(enable = "ssse3,sse4.1")]
unsafe fn transpose16x16(r: &mut [__m128i; 16]) {
    for _ in 0..4 {
        let mut t = [_mm_setzero_si128(); 16];
        for i in 0..8 {
            t[2 * i] = _mm_unpacklo_epi8(r[i], r[i + 8]);
            t[2 * i + 1] = _mm_unpackhi_epi8(r[i], r[i + 8]);
        }
        *r = t;
    }
}

/// Reconstructs sixteen rows in place: `rows` holds their residuals, `width`
/// apart, and receives the pixels. `prev` is the image row above the first.
/// Bit k of `med_rows` selects the MED predictor for row k.
///
/// # Panics
/// If `width < 32`, `prev` is shorter than `width` or `rows` than 16 rows.
#[target_feature(enable = "ssse3,sse4.1")]
pub(super) unsafe fn reconstruct(prev: &[u8], rows: &mut [u8], width: usize, med_rows: u32) {
    match med_rows {
        0 => reconstruct_kind::<0>(prev, rows, width, med_rows),
        0xFFFF => reconstruct_kind::<1>(prev, rows, width, med_rows),
        _ => reconstruct_kind::<2>(prev, rows, width, med_rows),
    }
}

/// Kind: 0 = average, 1 = MED, 2 = per-row selection by `med_rows`.
#[inline]
#[target_feature(enable = "ssse3,sse4.1")]
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
    let ones = _mm_set1_epi8(-1);
    let mut left = _mm_loadu_si128(left_init.as_ptr().cast());
    let mut top_left = _mm_loadu_si128(top_left_init.as_ptr().cast());
    let med_lanes = _mm_loadu_si128(lane_mask.as_ptr().cast());

    let base = rows.as_mut_ptr();
    let mut t0 = WAVE_ROWS;
    while t0 + WAVE_ROWS <= width {
        // Row k starts at pixel t0 - k: one row minus one pixel apart.
        let mut v: [__m128i; 16] =
            std::array::from_fn(|k| _mm_loadu_si128(base.add(k * width + t0 - k).cast()));
        transpose16x16(&mut v);
        for (j, residuals) in v.iter_mut().enumerate() {
            // Lane 0 takes its top pixel from the row above: byte 15 of this load.
            let above = _mm_xor_si128(_mm_loadu_si128(prev.as_ptr().add(t0 + j - 15).cast()), ones);
            let top = _mm_alignr_epi8::<15>(left, above);
            let predictor = if KIND == 0 {
                _mm_avg_epu8(left, top)
            } else {
                let min_ab = _mm_min_epu8(left, top);
                let max_ab = _mm_max_epu8(left, top);
                let clamped = _mm_min_epu8(_mm_max_epu8(top_left, min_ab), max_ab);
                let med = _mm_sub_epi8(_mm_add_epi8(left, top), clamped);
                if KIND == 1 {
                    med
                } else {
                    _mm_blendv_epi8(_mm_avg_epu8(left, top), med, med_lanes)
                }
            };
            left = _mm_sub_epi8(predictor, *residuals);
            top_left = top;
            *residuals = left;
        }
        transpose16x16(&mut v);
        for (k, pixels) in v.iter().enumerate() {
            _mm_storeu_si128(
                base.add(k * width + t0 - k).cast(),
                _mm_xor_si128(*pixels, ones),
            );
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
        /// Makes [`super::available`] report false on this thread, to compare the
        /// wavefront with the per-row code.
        pub(in crate::entropy_coder) static DISABLED: Cell<bool> = const { Cell::new(false) };
    }
}
