//! AVX2 tile decoder: eight blocks (32 pixels of four rows) per
//! group, two blocks per 256-bit register. Block `i` sits in the low lane and
//! block `i + 4` in the high one, so the per-lane SSE unpacking and transpose
//! apply unchanged and each row comes out as 32 contiguous pixels. The unpack
//! parameters of both lanes come from one table entry per pair of bit counts;
//! combining two 128-bit entries per pair instead made it slower than SSE.
//! Rows are finished by the SSE decoder.

use std::arch::x86_64::*;

use super::{decode_row_groups, UNPACK_PARAMS};

/// Whether the CPU has AVX2 (and the SSE kernels' instructions).
pub(in crate::lossy) fn available() -> bool {
    #[cfg(test)]
    if tests::DISABLED.with(std::cell::Cell::get) {
        return false;
    }
    super::available() && is_x86_feature_detected!("avx2")
}

/// Unpack parameters for a pair of bit counts, the low lane's block first:
/// both lanes' tables side by side, so they load as 256-bit values.
#[repr(C, align(32))]
struct PairParams {
    gather: [[u8; 32]; 2],
    scale: [[u16; 16]; 2],
    value_mask: [u16; 16],
}

static PAIR_PARAMS: [PairParams; 81] = {
    let mut table = [const {
        PairParams {
            gather: [[0; 32]; 2],
            scale: [[0; 16]; 2],
            value_mask: [0; 16],
        }
    }; 81];
    let mut index = 0;
    while index < 81 {
        let halves = [&UNPACK_PARAMS[index / 9], &UNPACK_PARAMS[index % 9]];
        let mut lane = 0;
        while lane < 2 {
            let params = halves[lane];
            let mut j = 0;
            while j < 16 {
                table[index].gather[0][lane * 16 + j] = params.gather[0][j];
                table[index].gather[1][lane * 16 + j] = params.gather[1][j];
                j += 1;
            }
            let mut j = 0;
            while j < 8 {
                table[index].scale[0][lane * 8 + j] = params.scale[0][j];
                table[index].scale[1][lane * 8 + j] = params.scale[1][j];
                table[index].value_mask[lane * 8 + j] = params.value_mask[j];
                j += 1;
            }
            lane += 1;
        }
        index += 1;
    }
    table
};

/// Loads a 32-byte table entry.
#[inline(always)]
unsafe fn load<T>(value: &T) -> __m256i {
    const { assert!(std::mem::size_of::<T>() == 32) };
    _mm256_load_si256((value as *const T).cast())
}

/// Two 128-bit values, `lo` in the low lane.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn lanes(lo: __m128i, hi: __m128i) -> __m256i {
    _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi)
}

/// Decodes two blocks to row-major pixels, one per lane, like
/// `super::decode_tile`. Reads 16 bytes at each source.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn decode_tile_pair<const LOSSLESS: bool>(
    src: [*const u8; 2],
    lut_entry: [u32; 2],
    min: [u8; 2],
) -> __m256i {
    let bits = |entry: u32| ((entry >> 8) & 0xFF) as usize;
    let params = &PAIR_PARAMS[bits(lut_entry[0]) * 9 + bits(lut_entry[1])];
    let packed = _mm256_loadu2_m128i(src[1].cast(), src[0].cast());
    let value_mask = load(&params.value_mask);
    let mut halves = [_mm256_setzero_si256(); 2];
    for (i, half) in halves.iter_mut().enumerate() {
        let words = _mm256_shuffle_epi8(packed, load(&params.gather[i]));
        let scaled = _mm256_mullo_epi16(words, load(&params.scale[i]));
        *half = _mm256_and_si256(_mm256_srli_epi16::<8>(scaled), value_mask);
    }
    if LOSSLESS {
        #[allow(clippy::cast_possible_wrap)]
        let min = _mm256_shuffle_epi8(
            _mm256_set1_epi16(i16::from_le_bytes(min)),
            _mm256_setr_epi8(
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            ),
        );
        _mm256_adds_epu8(_mm256_packus_epi16(halves[0], halves[1]), min)
    } else {
        // Lossy streams add half the bucket size; products saturate like the
        // per-block decoder.
        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
        let [(base0, scale0), (base1, scale1)] = std::array::from_fn(|i| {
            let bucket_size = lut_entry[i] & 0xFF;
            let base = min[i].wrapping_add((bucket_size >> 1) as u8);
            (
                _mm_set1_epi16(i16::from(base)),
                _mm_set1_epi16(bucket_size as i16),
            )
        });
        let (base, scale) = (lanes(base0, base1), lanes(scale0, scale1));
        _mm256_packus_epi16(
            _mm256_add_epi16(_mm256_mullo_epi16(halves[0], scale), base),
            _mm256_add_epi16(_mm256_mullo_epi16(halves[1], scale), base),
        )
    }
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

/// [`decode_row_groups`] with 32-pixel groups; the SSE decoder finishes the
/// row. Same arguments and result.
///
/// # Panics
/// If the four rows do not fit in `dst` or a header stream is too short.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,ssse3,sse4.1")]
pub(in crate::lossy) unsafe fn decode_row_groups_avx2(
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
    let groups = width / 32;
    assert!(row_start + 3 * stride + groups * 32 <= dst.len());
    let mins = &min_stream[first_block..first_block + groups * 8];
    let dists = &dist_stream[first_block..first_block + groups * 8];
    let out: [*mut u8; 4] = std::array::from_fn(|r| dst.as_mut_ptr().add(row_start + r * stride));
    let (done, read) = if lossless {
        decode_groups::<true>(out, mins, dists, lut, pixel_stream)
    } else {
        decode_groups::<false>(out, mins, dists, lut, pixel_stream)
    };
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

/// Decodes the groups of `mins.len() / 8` blocks into the four rows at `out`
/// while their reads stay inside `pixel_stream`. Returns the pixels decoded
/// and the bytes read.
///
/// The four block pairs of a group are written out rather than looped over:
/// fat LTO kept such a loop, with the lossless branch inside, and decoded
/// about 25% slower.
#[inline]
#[target_feature(enable = "avx2,ssse3,sse4.1")]
unsafe fn decode_groups<const LOSSLESS: bool>(
    out: [*mut u8; 4],
    mins: &[u8],
    dists: &[u8],
    lut: &[u32; 256],
    pixel_stream: &[u8],
) -> (usize, usize) {
    let midpoint_mask: u32 = if LOSSLESS { 0 } else { 0xFF };
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
            let bases = std::array::from_fn::<u8, 8, _>(|i| {
                #[allow(clippy::cast_possible_truncation)]
                let midpoint = (((entries[i] & 0xFF) >> 1) & midpoint_mask) as u8;
                block_mins[i].wrapping_add(midpoint)
            });
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
                    decode_tile_pair::<LOSSLESS>(
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
        /// thread, to compare the AVX2 decoder with the SSE one.
        pub(in crate::lossy) static DISABLED: Cell<bool> = const { Cell::new(false) };
    }
}
