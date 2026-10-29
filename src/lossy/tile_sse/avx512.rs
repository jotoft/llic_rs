//! AVX-512 VBMI tile decoder at 256-bit width (AVX-512VL): the AVX2
//! decoder's eight-block groups, two blocks per register (block `i` in the low
//! lane, `i + 4` in the high one), with the bit unpacking done by VBMI.
//!
//! Per pair of bit counts a table entry holds four controls: a `VPERMB` moving
//! the second 8-value half of each block (at byte `b`) to byte 8, a
//! `VPMULTISHIFTQB` extracting value `k` of each half at bit `k * b`, the mask
//! keeping `b` bits, and a `VPERMB` putting the values in pixel order. The
//! AVX2 decoder needs two shuffles, multiplies, shifts and masks instead.
//! Rows are finished by the SSE decoder.
//!
//! Only lossless streams use it: lossy ones widen the values to 16 bits for
//! the bucket multiply anyway, which made this decoder no faster than AVX2
//! (up to 2% slower), so they keep the AVX2 decoder.
//!
//! Sixteen blocks per 512-bit register were tried first, with the controls
//! built per lane in registers: on Zen 4, which splits 512-bit operations in
//! two, that decoded fast-mode streams 10-50% slower than AVX2.

use std::arch::x86_64::*;

use super::{avx2, decode_row_groups, PACK_PARAMS};

/// Whether the CPU has AVX-512 F, BW, VL and VBMI (and the AVX2 decoder's
/// instructions).
pub(in crate::lossy) fn available() -> bool {
    #[cfg(test)]
    if tests::DISABLED.with(std::cell::Cell::get) {
        return false;
    }
    avx2::available()
        && is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vl")
        && is_x86_feature_detected!("avx512vbmi")
}

/// Unpack controls for a pair of bit counts, the low lane's block first.
#[repr(C, align(32))]
struct PairControls {
    /// Byte `p` of a lane: `p` for the first half, `p - 8 + b` for the second.
    align: [u8; 32],
    /// Bit `k * b` for byte `k` of each 8-value half.
    shifts: [u8; 32],
    /// `b` low bits.
    masks: [u8; 32],
    /// The packed position of each pixel of the row-major block.
    order: [u8; 32],
}

static PAIR_CONTROLS: [PairControls; 81] = {
    let mut table = [const {
        PairControls {
            align: [0; 32],
            shifts: [0; 32],
            masks: [0; 32],
            order: [0; 32],
        }
    }; 81];
    let mut index = 0;
    while index < 81 {
        let lane_bits = [index / 9, index % 9];
        let mut lane = 0;
        while lane < 2 {
            let bits = lane_bits[lane];
            let base = lane * 16;
            let mut p = 0;
            while p < 16 {
                let entry = &mut table[index];
                entry.align[base + p] = (base + if p < 8 { p } else { p - 8 + bits }) as u8;
                entry.shifts[base + p] = ((p % 8) * bits) as u8;
                entry.masks[base + p] = ((1u32 << bits) - 1) as u8;
                entry.order[base + PACK_PARAMS[bits].index[p] as usize] = (base + p) as u8;
                p += 1;
            }
            lane += 1;
        }
        index += 1;
    }
    table
};

/// Loads a 32-byte table entry.
#[inline(always)]
unsafe fn load(value: &[u8; 32]) -> __m256i {
    _mm256_load_si256(value.as_ptr().cast())
}

/// Decodes two lossless blocks to row-major pixels, one per lane, like
/// `super::decode_tile`. Reads 16 bytes at each source.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx2")]
unsafe fn decode_tile_pair(src: [*const u8; 2], lut_entry: [u32; 2], min: [u8; 2]) -> __m256i {
    let bits = |entry: u32| ((entry >> 8) & 0xFF) as usize;
    let controls = &PAIR_CONTROLS[bits(lut_entry[0]) * 9 + bits(lut_entry[1])];
    let packed = _mm256_loadu2_m128i(src[1].cast(), src[0].cast());
    let halves = _mm256_permutexvar_epi8(load(&controls.align), packed);
    let values = _mm256_and_si256(
        _mm256_multishift_epi64_epi8(load(&controls.shifts), halves),
        load(&controls.masks),
    );
    let residuals = _mm256_permutexvar_epi8(load(&controls.order), values);
    #[allow(clippy::cast_possible_wrap)]
    let min = _mm256_shuffle_epi8(
        _mm256_set1_epi16(i16::from_le_bytes(min)),
        _mm256_setr_epi8(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ),
    );
    _mm256_adds_epu8(residuals, min)
}

/// Transposes the 4x4 matrix of 32-bit values in each lane.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn transpose4x32(r: &mut [__m256i; 4]) {
    let t0 = _mm256_unpacklo_epi32(r[0], r[1]);
    let t1 = _mm256_unpacklo_epi32(r[2], r[3]);
    let t2 = _mm256_unpackhi_epi32(r[0], r[1]);
    let t3 = _mm256_unpackhi_epi32(r[2], r[3]);
    r[0] = _mm256_unpacklo_epi64(t0, t1);
    r[1] = _mm256_unpackhi_epi64(t0, t1);
    r[2] = _mm256_unpacklo_epi64(t2, t3);
    r[3] = _mm256_unpackhi_epi64(t2, t3);
}

/// [`decode_row_groups`] with 32-pixel groups for lossless streams; the SSE
/// decoder finishes the row. Lossy streams go to the AVX2 decoder. Same
/// arguments and result.
///
/// # Panics
/// If the four rows do not fit in `dst` or a header stream is too short.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx2,ssse3,sse4.1")]
pub(in crate::lossy) unsafe fn decode_row_groups_avx512(
    dst: &mut [u8],
    row_start: usize,
    stride: usize,
    width: usize,
    first_block: usize,
    min_stream: &[u8],
    dist_stream: &[u8],
    lossless: bool,
    lut: &[u32; 256],
    pixel_stream: &[u8],
) -> (usize, usize) {
    if !lossless {
        return avx2::decode_row_groups_avx2(
            dst,
            row_start,
            stride,
            width,
            first_block,
            min_stream,
            dist_stream,
            lossless,
            lut,
            pixel_stream,
        );
    }
    let groups = width / 32;
    assert!(row_start + 3 * stride + groups * 32 <= dst.len());
    let mins = &min_stream[first_block..first_block + groups * 8];
    let dists = &dist_stream[first_block..first_block + groups * 8];
    let out: [*mut u8; 4] = std::array::from_fn(|r| dst.as_mut_ptr().add(row_start + r * stride));
    let (done, read) = decode_groups(out, mins, dists, lut, pixel_stream);
    let (end, rest) = decode_row_groups(
        dst,
        row_start + done,
        stride,
        width - done,
        first_block + done / 4,
        min_stream,
        dist_stream,
        lossless,
        lut,
        &pixel_stream[read..],
    );
    (done + end, read + rest)
}

/// Decodes the lossless groups of `mins.len() / 8` blocks into the four rows
/// at `out` while their reads stay inside `pixel_stream`. Returns the pixels
/// decoded and the bytes read. As in the AVX2 decoder, the four block pairs of
/// a group are written out rather than looped over.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx2,ssse3,sse4.1")]
unsafe fn decode_groups(
    out: [*mut u8; 4],
    mins: &[u8],
    dists: &[u8],
    lut: &[u32; 256],
    pixel_stream: &[u8],
) -> (usize, usize) {
    let mut read = 0usize;
    let mut group = 0usize;
    while group < mins.len() / 8 {
        let block_mins = &mins[group * 8..group * 8 + 8];
        let entries: [u32; 8] = std::array::from_fn(|i| lut[usize::from(dists[group * 8 + i])]);
        let no_payload = entries.iter().fold(0, |acc, e| acc | e) & 0xFF00 == 0;
        // Eight 16-byte reads at most 112 bytes in.
        if !no_payload && pixel_stream.len() - read < 128 {
            break;
        }
        let rows = if no_payload {
            // Lossless blocks without payload are their minimum.
            let bases: [u8; 8] = block_mins.try_into().expect("8 blocks");
            #[allow(clippy::cast_possible_wrap)]
            let row = _mm256_shuffle_epi8(
                _mm256_set1_epi64x(u64::from_le_bytes(bases) as i64),
                _mm256_setr_epi8(
                    0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, //
                    4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7,
                ),
            );
            [row; 4]
        } else {
            let src = pixel_stream.as_ptr().add(read);
            let mut offsets = [0usize; 9];
            for i in 0..8 {
                offsets[i + 1] = offsets[i] + 2 * ((entries[i] >> 8) & 0xFF) as usize;
            }
            macro_rules! pair {
                ($i:literal) => {
                    decode_tile_pair(
                        [src.add(offsets[$i]), src.add(offsets[$i + 4])],
                        [entries[$i], entries[$i + 4]],
                        [block_mins[$i], block_mins[$i + 4]],
                    )
                };
            }
            let mut blocks = [pair!(0), pair!(1), pair!(2), pair!(3)];
            read += offsets[8];
            transpose4x32(&mut blocks);
            blocks
        };
        for (row, pixels) in out.iter().zip(rows) {
            _mm256_storeu_si256(row.add(group * 32).cast(), pixels);
        }
        group += 1;
    }
    (group * 32, read)
}

#[cfg(test)]
pub(in crate::lossy) mod tests {
    use std::cell::Cell;

    thread_local! {
        /// Makes [`available`](super::available) report false on this
        /// thread, to compare the AVX-512 decoder with the AVX2 one.
        pub(in crate::lossy) static DISABLED: Cell<bool> = const { Cell::new(false) };
    }
}
