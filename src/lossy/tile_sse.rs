//! SSSE3/SSE4.1 tile codec working on groups of blocks, ported from the C++
//! reference's `LLIC_SSE_TILE_CODEC` (llic 3.4+).
//!
//! The encoder reduces sixteen blocks (64 pixels of four rows) to their minima
//! and distances at once and packs every bit width without branching. The
//! decoder unpacks four blocks (16 pixels) at once. Streams and pixels are
//! identical to the per-block code; rows are finished by the caller's
//! per-block code where a group does not fit.
//!
//! The kernels need SSSE3 and SSE4.1, detected at run time by [`available`].

use std::arch::x86_64::*;

use super::BucketLutEntry;

/// Whether the CPU supports the instructions these kernels use.
pub(super) fn available() -> bool {
    #[cfg(test)]
    if tests::DISABLED.with(std::cell::Cell::get) {
        return false;
    }
    is_x86_feature_detected!("ssse3") && is_x86_feature_detected!("sse4.1")
}

/// Bucket LUT in the C++ layout, as the kernels read it: bits 0-7 hold the
/// bucket size, bits 8-15 the bit count, bits 16-31 the Q15 reciprocal
/// `ceil(32768 / bucket_size)` (0 for bucket size 1, which keeps residuals).
pub(super) const fn packed_lut(lut: &[BucketLutEntry; 256]) -> [u32; 256] {
    let mut packed = [0u32; 256];
    let mut dist = 0;
    while dist < 256 {
        let (bits, bucket_size) = lut[dist];
        let bucket_size = bucket_size as u32;
        let reciprocal = if bucket_size > 1 {
            32768u32.div_ceil(bucket_size)
        } else {
            0
        };
        packed[dist] = (reciprocal << 16) | ((bits as u32) << 8) | bucket_size;
        dist += 1;
    }
    packed
}

/// Per bit count: the pixel order of the LSB-first packed value sequence,
/// multipliers merging 8- and 16-bit lane pairs (`PMADDUBSW`, `PMADDWD`) and
/// moving the upper 32-bit half of each 64-bit lane down (`PMULUDQ`), and a mask
/// selecting unpacked 8-bit values. Eight-bit tiles have zero multipliers.
#[repr(C, align(16))]
struct PackParams {
    index: [u8; 16],
    pair_factor: [u8; 16],
    quad_factor: [i16; 8],
    octet_factor: [u32; 4],
    keep_residual: [u8; 16],
}

const fn pack(index: [u8; 16], pair: u8, quad: i16, octet: u32, keep_residual: u8) -> PackParams {
    PackParams {
        index,
        pair_factor: [
            1, pair, 1, pair, 1, pair, 1, pair, 1, pair, 1, pair, 1, pair, 1, pair,
        ],
        quad_factor: [1, quad, 1, quad, 1, quad, 1, quad],
        octet_factor: [octet, 0, octet, 0],
        keep_residual: [keep_residual; 16],
    }
}

const IDENTITY: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
const COLUMN_MAJOR: [u8; 16] = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15];

static PACK_PARAMS: [PackParams; 9] = [
    pack(IDENTITY, 1, 1, 1, 0),
    pack(
        [1, 5, 9, 13, 0, 4, 8, 12, 3, 7, 11, 15, 2, 6, 10, 14],
        2,
        4,
        1 << 4,
        0,
    ),
    pack(COLUMN_MAJOR, 4, 16, 1 << 8, 0),
    pack(
        [0, 4, 8, 12, 2, 6, 10, 14, 1, 5, 9, 13, 3, 7, 11, 15],
        8,
        64,
        1 << 12,
        0,
    ),
    pack(
        [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15],
        16,
        256,
        1 << 16,
        0,
    ),
    pack(COLUMN_MAJOR, 32, 1024, 1 << 20, 0),
    pack(COLUMN_MAJOR, 64, 4096, 1 << 24, 0),
    // Seven bits is not a tile format.
    PackParams {
        index: IDENTITY,
        pair_factor: [0; 16],
        quad_factor: [0; 8],
        octet_factor: [0; 4],
        keep_residual: [0; 16],
    },
    PackParams {
        index: IDENTITY,
        pair_factor: [0; 16],
        quad_factor: [0; 8],
        octet_factor: [0; 4],
        keep_residual: [0xFF; 16],
    },
];

/// Inverse of [`PACK_PARAMS`]. Per 8-pixel half: the two bytes holding each
/// value, and a multiplier moving it to bit 8.
#[repr(C, align(16))]
struct UnpackParams {
    gather: [[u8; 16]; 2],
    scale: [[u16; 8]; 2],
    value_mask: [u16; 8],
}

const NONE: u8 = 0x80;

static UNPACK_PARAMS: [UnpackParams; 9] = [
    UnpackParams {
        gather: [[NONE; 16], [NONE; 16]],
        scale: [[0; 8], [0; 8]],
        value_mask: [0; 8],
    },
    UnpackParams {
        gather: [
            [0, 1, 0, 1, 1, 2, 1, 2, 0, 1, 0, 1, 1, 2, 1, 2],
            [0, 1, 0, 1, 1, 2, 1, 2, 0, 1, 0, 1, 1, 2, 1, 2],
        ],
        scale: [
            [16, 256, 16, 256, 8, 128, 8, 128],
            [4, 64, 4, 64, 2, 32, 2, 32],
        ],
        value_mask: [0x01; 8],
    },
    UnpackParams {
        gather: [
            [0, 1, 1, 2, 2, 3, 3, 4, 0, 1, 1, 2, 2, 3, 3, 4],
            [0, 1, 1, 2, 2, 3, 3, 4, 0, 1, 1, 2, 2, 3, 3, 4],
        ],
        scale: [
            [256, 256, 256, 256, 64, 64, 64, 64],
            [16, 16, 16, 16, 4, 4, 4, 4],
        ],
        value_mask: [0x03; 8],
    },
    UnpackParams {
        gather: [
            [0, 1, 3, 4, 1, 2, 4, 5, 0, 1, 3, 4, 1, 2, 4, 5],
            [0, 1, 3, 4, 2, 3, 5, 6, 1, 2, 4, 5, 2, 3, 5, 6],
        ],
        scale: [
            [256, 256, 16, 16, 32, 32, 2, 2],
            [4, 4, 64, 64, 128, 128, 8, 8],
        ],
        value_mask: [0x07; 8],
    },
    UnpackParams {
        gather: [
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8],
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8],
        ],
        scale: [[256; 8], [16; 8]],
        value_mask: [0x0F; 8],
    },
    UnpackParams {
        gather: [
            [0, 1, 2, 3, 5, 6, 7, 8, 0, 1, 3, 4, 5, 6, 8, 9],
            [1, 2, 3, 4, 6, 7, 8, 9, 1, 2, 4, 5, 6, 7, 9, 10],
        ],
        scale: [
            [256, 16, 256, 16, 8, 128, 8, 128],
            [64, 4, 64, 4, 2, 32, 2, 32],
        ],
        value_mask: [0x1F; 8],
    },
    UnpackParams {
        gather: [
            [0, 1, 3, 4, 6, 7, 9, 10, 0, 1, 3, 4, 6, 7, 9, 10],
            [1, 2, 4, 5, 7, 8, 10, 11, 2, 3, 5, 6, 8, 9, 11, 12],
        ],
        scale: [
            [256, 256, 256, 256, 4, 4, 4, 4],
            [16, 16, 16, 16, 64, 64, 64, 64],
        ],
        value_mask: [0x3F; 8],
    },
    UnpackParams {
        gather: [
            [0, 1, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
            [7, 8, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14],
        ],
        scale: [
            [256, 2, 4, 8, 16, 32, 64, 128],
            [256, 2, 4, 8, 16, 32, 64, 128],
        ],
        value_mask: [0x7F; 8],
    },
    UnpackParams {
        gather: [
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8],
            [
                8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, NONE,
            ],
        ],
        scale: [[256; 8], [256; 8]],
        value_mask: [0xFF; 8],
    },
];

/// Loads a 16-byte table entry. Unaligned, since `const` tables used as
/// temporaries have no 16-byte alignment (same cost as aligned on current CPUs).
#[inline(always)]
unsafe fn load<T>(value: &T) -> __m128i {
    const { assert!(std::mem::size_of::<T>() == 16) };
    _mm_loadu_si128((value as *const T).cast())
}

/// Transposes a 4x4 matrix of 32-bit lanes: four image rows to four row-major
/// blocks and back.
#[inline]
#[target_feature(enable = "ssse3,sse4.1")]
unsafe fn transpose4x32(r: &mut [__m128i; 4]) {
    let t0 = _mm_unpacklo_epi32(r[0], r[1]);
    let t1 = _mm_unpacklo_epi32(r[2], r[3]);
    let t2 = _mm_unpackhi_epi32(r[0], r[1]);
    let t3 = _mm_unpackhi_epi32(r[2], r[3]);
    r[0] = _mm_unpacklo_epi64(t0, t1);
    r[1] = _mm_unpackhi_epi64(t0, t1);
    r[2] = _mm_unpacklo_epi64(t2, t3);
    r[3] = _mm_unpackhi_epi64(t2, t3);
}

/// Reduces the column minima (`MAX = false`) or maxima of 16 blocks to one
/// byte per block. Each input holds four blocks, one 32-bit lane per block.
#[inline]
#[target_feature(enable = "ssse3,sse4.1")]
unsafe fn reduce_blocks<const MAX: bool>(columns: &[__m128i; 4]) -> __m128i {
    let combine = |a, b| {
        if MAX {
            _mm_max_epu8(a, b)
        } else {
            _mm_min_epu8(a, b)
        }
    };
    // Gather equal columns of the four blocks into one lane, then combine lanes.
    let column_major = load(&COLUMN_MAJOR);
    let mut halves = [_mm_setzero_si128(); 2];
    for (i, half) in halves.iter_mut().enumerate() {
        let a = _mm_shuffle_epi8(columns[2 * i], column_major);
        let b = _mm_shuffle_epi8(columns[2 * i + 1], column_major);
        *half = combine(_mm_unpacklo_epi64(a, b), _mm_unpackhi_epi64(a, b));
    }
    let h0 = _mm_castsi128_ps(halves[0]);
    let h1 = _mm_castsi128_ps(halves[1]);
    let even = _mm_castps_si128(_mm_shuffle_ps::<0b10_00_10_00>(h0, h1));
    let odd = _mm_castps_si128(_mm_shuffle_ps::<0b11_01_11_01>(h0, h1));
    combine(even, odd)
}

/// Sixteen blocks at a time; see `super::select_lossless_header`. The width
/// bounds come from per-nibble tables.
#[inline]
#[target_feature(enable = "ssse3,sse4.1")]
unsafe fn select_lossless_header(
    mins: __m128i,
    dists: __m128i,
    top_min: __m128i,
    top_dist: __m128i,
) -> (__m128i, __m128i) {
    let nibble_mask = _mm_set1_epi8(0x0F);
    let low_nibble = _mm_and_si128(dists, nibble_mask);
    let high_nibble = _mm_and_si128(_mm_srli_epi16::<4>(dists), nibble_mask);
    let hi_low = _mm_setr_epi8(0, 1, 3, 3, 7, 7, 7, 7, 15, 15, 15, 15, 15, 15, 15, 15);
    let hi_high = _mm_setr_epi8(
        0, 31, 63, 63, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    );
    let lo_low = _mm_setr_epi8(0, 1, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8);
    let lo_high = _mm_setr_epi8(
        0, 16, 32, 32, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    );
    let hi = _mm_max_epu8(
        _mm_shuffle_epi8(hi_low, low_nibble),
        _mm_shuffle_epi8(hi_high, high_nibble),
    );
    let lo = _mm_max_epu8(
        _mm_shuffle_epi8(lo_low, low_nibble),
        _mm_shuffle_epi8(lo_high, high_nibble),
    );
    let maxs = _mm_add_epi8(mins, dists);
    let mins = _mm_max_epu8(_mm_min_epu8(top_min, mins), _mm_subs_epu8(maxs, hi));
    let dists = _mm_min_epu8(_mm_max_epu8(top_dist, lo), hi);
    (mins, dists)
}

/// Packs the residuals of one row-major block without branching on the bit
/// count and returns the packed size. Writes up to 16 bytes at `dst`: the
/// second store overwrites the unused bytes of the first.
#[inline]
#[target_feature(enable = "ssse3,sse4.1")]
unsafe fn encode_tile<const QUANTIZE: bool>(block: __m128i, lut_entry: u32, dst: *mut u8) -> usize {
    let bits = ((lut_entry >> 8) & 0xFF) as usize;
    let params = &PACK_PARAMS[bits];
    let residuals = _mm_shuffle_epi8(block, load(&params.index));
    let mut values = residuals;
    if QUANTIZE {
        // Q15 reciprocal division, exact for byte residuals and bucket sizes
        // 2..33: (2 * r * x) >> 16. Odd bytes are divided in place; masking
        // their product drops the fraction below bit 8.
        #[allow(clippy::cast_possible_wrap)] // a bit pattern for the shuffle
        let reciprocal = _mm_shuffle_epi8(
            _mm_cvtsi32_si128(lut_entry as i32),
            _mm_setr_epi8(2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3),
        );
        let low_bytes = _mm_set1_epi16(0x00FF);
        let twice_reciprocal = _mm_add_epi16(reciprocal, reciprocal);
        let even = _mm_mulhi_epu16(_mm_and_si128(residuals, low_bytes), twice_reciprocal);
        let odd = _mm_mulhi_epu16(_mm_andnot_si128(low_bytes, residuals), twice_reciprocal);
        values = _mm_or_si128(even, _mm_andnot_si128(low_bytes, odd));
    }
    let pairs = _mm_maddubs_epi16(values, load(&params.pair_factor));
    let quads = _mm_madd_epi16(pairs, load(&params.quad_factor));
    let octets = _mm_add_epi64(
        _mm_blend_epi16::<0xCC>(quads, _mm_setzero_si128()),
        _mm_mul_epu32(
            _mm_shuffle_epi32::<0b11_11_01_01>(quads),
            load(&params.octet_factor),
        ),
    );
    // Eight-bit tiles (bucket size one) store the residuals unchanged.
    let packed = _mm_or_si128(
        octets,
        _mm_and_si128(residuals, load(&params.keep_residual)),
    );
    _mm_storel_epi64(dst.cast(), packed);
    _mm_storeh_pd(dst.add(bits).cast(), _mm_castsi128_pd(packed));
    2 * bits
}

/// Encodes the 16 blocks of one group, whose header values are already in
/// `mins` and `dist_bytes`. Returns the bytes written to `dst`.
#[inline]
#[target_feature(enable = "ssse3,sse4.1")]
unsafe fn encode_group<const QUANTIZE: bool>(
    rows: [*const u8; 4],
    mins: __m128i,
    dists: __m128i,
    dist_bytes: &[u8],
    error_limit: u8,
    lut: &[u32; 256],
    dst: *mut u8,
) -> usize {
    // Blocks with dist <= error_limit use zero bits.
    #[allow(clippy::cast_possible_wrap)]
    let no_payload = _mm_cmpeq_epi8(
        _mm_subs_epu8(dists, _mm_set1_epi8(error_limit as i8)),
        _mm_setzero_si128(),
    );
    #[allow(clippy::cast_sign_loss)]
    let payload_mask = !(_mm_movemask_epi8(no_payload) as u32);
    let mut written = 0usize;
    for chunk in 0..4usize {
        if (payload_mask >> (chunk * 4)) & 0xF == 0 {
            continue;
        }
        // Each block's minimum repeated over its four columns.
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let min = _mm_shuffle_epi8(
            mins,
            _mm_add_epi8(
                _mm_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3),
                _mm_set1_epi8((chunk * 4) as i8),
            ),
        );
        let mut blocks =
            rows.map(|row| _mm_sub_epi8(_mm_loadu_si128(row.add(chunk * 16).cast()), min));
        transpose4x32(&mut blocks);
        for (block, &dist) in blocks.iter().zip(&dist_bytes[chunk * 4..chunk * 4 + 4]) {
            written += encode_tile::<QUANTIZE>(*block, lut[usize::from(dist)], dst.add(written));
        }
    }
    written
}

/// Encodes the whole 64-pixel groups of one row of blocks.
///
/// The row of blocks starts at `src[row_start]` with rows `stride` apart;
/// `first_block` indexes its first block in `min_stream` and `dist_stream`. With `select_header`, lossless header values are chosen
/// against the row above (the caller passes it only from the second row on).
/// Returns the pixel column the caller continues from and the bytes written to
/// `dst`.
///
/// # Panics
/// If the four rows do not fit in `src` or `dst` cannot take 16 bytes per block.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "ssse3,sse4.1")]
pub(super) unsafe fn encode_row_groups(
    src: &[u8],
    row_start: usize,
    stride: usize,
    width: usize,
    first_block: usize,
    select_header: bool,
    min_stream: &mut [u8],
    dist_stream: &mut [u8],
    error_limit: u8,
    lut: &[u32; 256],
    dst: &mut [u8],
) -> (usize, usize) {
    let groups = width / 64;
    assert!(row_start + 3 * stride + groups * 64 <= src.len());
    assert!(dst.len() >= groups * 16 * 16, "16 bytes per block");
    assert!(min_stream.len() >= first_block + groups * 16);
    assert!(dist_stream.len() >= first_block + groups * 16);
    let blocks_per_row = width / 4;
    let dst = dst.as_mut_ptr();
    let mut written = 0usize;
    for group in 0..groups {
        let x = group * 64;
        let block = first_block + group * 16;
        let row_ptrs: [*const u8; 4] =
            std::array::from_fn(|r| src.as_ptr().add(row_start + r * stride + x));
        let mut column_mins = [_mm_setzero_si128(); 4];
        let mut column_maxs = [_mm_setzero_si128(); 4];
        for i in 0..4 {
            let [r0, r1, r2, r3] = row_ptrs.map(|row| _mm_loadu_si128(row.add(i * 16).cast()));
            column_mins[i] = _mm_min_epu8(_mm_min_epu8(r0, r1), _mm_min_epu8(r2, r3));
            column_maxs[i] = _mm_max_epu8(_mm_max_epu8(r0, r1), _mm_max_epu8(r2, r3));
        }
        let mut mins = reduce_blocks::<false>(&column_mins);
        let mut dists = _mm_sub_epi8(reduce_blocks::<true>(&column_maxs), mins);
        if select_header {
            let above = block - blocks_per_row;
            (mins, dists) = select_lossless_header(
                mins,
                dists,
                _mm_loadu_si128(min_stream[above..above + 16].as_ptr().cast()),
                _mm_loadu_si128(dist_stream[above..above + 16].as_ptr().cast()),
            );
        }
        _mm_storeu_si128(min_stream[block..block + 16].as_mut_ptr().cast(), mins);
        _mm_storeu_si128(dist_stream[block..block + 16].as_mut_ptr().cast(), dists);
        let dist_bytes = &dist_stream[block..block + 16];
        written += if error_limit == 0 {
            encode_group::<false>(row_ptrs, mins, dists, dist_bytes, 0, lut, dst.add(written))
        } else {
            encode_group::<true>(
                row_ptrs,
                mins,
                dists,
                dist_bytes,
                error_limit,
                lut,
                dst.add(written),
            )
        };
    }
    (groups * 64, written)
}

/// Decodes one block to row-major pixels without branching on the bit count.
/// Reads 16 bytes at `src`; returns the pixels and the packed size.
#[inline]
#[target_feature(enable = "ssse3,sse4.1")]
unsafe fn decode_tile<const LOSSLESS: bool>(
    src: *const u8,
    lut_entry: u32,
    min: u8,
) -> (__m128i, usize) {
    let bits = ((lut_entry >> 8) & 0xFF) as usize;
    let params = &UNPACK_PARAMS[bits];
    let packed = _mm_loadu_si128(src.cast());
    let value_mask = load(&params.value_mask);
    let mut halves = [_mm_setzero_si128(); 2];
    for (i, half) in halves.iter_mut().enumerate() {
        let words = _mm_shuffle_epi8(packed, load(&params.gather[i]));
        let scaled = _mm_mullo_epi16(words, load(&params.scale[i]));
        *half = _mm_and_si128(_mm_srli_epi16::<8>(scaled), value_mask);
    }
    let pixels = if LOSSLESS {
        #[allow(clippy::cast_possible_wrap)]
        _mm_adds_epu8(
            _mm_packus_epi16(halves[0], halves[1]),
            _mm_set1_epi8(min as i8),
        )
    } else {
        // Lossy streams add half the bucket size; products saturate like the
        // per-block decoder.
        let bucket_size = lut_entry & 0xFF;
        #[allow(clippy::cast_possible_truncation)]
        let base = min.wrapping_add((bucket_size >> 1) as u8);
        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
        let (base, scale) = (
            _mm_set1_epi16(i16::from(base)),
            _mm_set1_epi16(bucket_size as i16),
        );
        _mm_packus_epi16(
            _mm_add_epi16(_mm_mullo_epi16(halves[0], scale), base),
            _mm_add_epi16(_mm_mullo_epi16(halves[1], scale), base),
        )
    };
    (pixels, 2 * bits)
}

/// Decodes the 16-pixel groups of one row of blocks while their 16-byte reads
/// stay inside `pixel_stream`.
///
/// The row of blocks starts at `dst[row_start]` with rows `stride` apart;
/// `first_block` indexes its first block in `min_stream` and `dist_stream`.
/// Returns the pixel column the caller continues from and the bytes read.
///
/// # Panics
/// If the four rows do not fit in `dst` or a header stream is too short.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "ssse3,sse4.1")]
pub(super) unsafe fn decode_row_groups(
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
    let groups = width / 16;
    assert!(row_start + 3 * stride + groups * 16 <= dst.len());
    let mins = &min_stream[first_block..first_block + groups * 4];
    let dists = &dist_stream[first_block..first_block + groups * 4];
    let out: [*mut u8; 4] = std::array::from_fn(|r| dst.as_mut_ptr().add(row_start + r * stride));
    let midpoint_mask: u32 = if lossless { 0 } else { 0xFF };
    let mut read = 0usize;
    let mut group = 0usize;
    while group < groups {
        let block_mins = &mins[group * 4..group * 4 + 4];
        let entries: [u32; 4] = std::array::from_fn(|i| lut[usize::from(dists[group * 4 + i])]);
        let no_payload = entries.iter().fold(0, |acc, e| acc | e) & 0xFF00 == 0;
        // Groups without payload read nothing, so only the others need the guard.
        if !no_payload && pixel_stream.len() - read < 64 {
            break;
        }
        let rows = if no_payload {
            // Four blocks without payload, common in flat or low-quality regions.
            let bases = std::array::from_fn::<u8, 4, _>(|i| {
                #[allow(clippy::cast_possible_truncation)]
                let midpoint = (((entries[i] & 0xFF) >> 1) & midpoint_mask) as u8;
                block_mins[i].wrapping_add(midpoint)
            });
            #[allow(clippy::cast_possible_wrap)]
            let row = _mm_shuffle_epi8(
                _mm_cvtsi32_si128(u32::from_le_bytes(bases) as i32),
                _mm_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3),
            );
            [row; 4]
        } else {
            let src = pixel_stream.as_ptr().add(read);
            let mut offset = 0usize;
            let mut blocks = [_mm_setzero_si128(); 4];
            for (i, block) in blocks.iter_mut().enumerate() {
                let (pixels, size) = if lossless {
                    decode_tile::<true>(src.add(offset), entries[i], block_mins[i])
                } else {
                    decode_tile::<false>(src.add(offset), entries[i], block_mins[i])
                };
                *block = pixels;
                offset += size;
            }
            read += offset;
            transpose4x32(&mut blocks);
            blocks
        };
        for (row, pixels) in out.iter().zip(rows) {
            _mm_storeu_si128(row.add(group * 16).cast(), pixels);
        }
        group += 1;
    }
    (group * 16, read)
}

#[cfg(test)]
pub(super) mod tests {
    use std::cell::Cell;

    use super::*;
    use crate::lossy::generate_bucket_lut;

    thread_local! {
        /// Makes [`available`] report false on this thread, to compare the
        /// SSE path with the per-block code.
        pub(in crate::lossy) static DISABLED: Cell<bool> = const { Cell::new(false) };
    }

    #[test]
    fn sse_header_selection_matches_scalar() {
        if !available() {
            return;
        }
        let lut = generate_bucket_lut(0);
        let mut state = 0x1234_5678_u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for _ in 0..20_000 {
            let mut mins = [0u8; 16];
            let mut dists = [0u8; 16];
            let mut top_min = [0u8; 16];
            let mut top_dist = [0u8; 16];
            for i in 0..16 {
                #[allow(clippy::cast_possible_truncation)]
                {
                    mins[i] = next() as u8;
                    dists[i] = (next() % (256 - u32::from(mins[i]))) as u8;
                    top_min[i] = next() as u8;
                    top_dist[i] = next() as u8;
                }
            }
            let mut got = ([0u8; 16], [0u8; 16]);
            // SAFETY: SSSE3 and SSE4.1 are available; all buffers are 16 bytes.
            unsafe {
                let (m, d) = select_lossless_header(
                    _mm_loadu_si128(mins.as_ptr().cast()),
                    _mm_loadu_si128(dists.as_ptr().cast()),
                    _mm_loadu_si128(top_min.as_ptr().cast()),
                    _mm_loadu_si128(top_dist.as_ptr().cast()),
                );
                _mm_storeu_si128(got.0.as_mut_ptr().cast(), m);
                _mm_storeu_si128(got.1.as_mut_ptr().cast(), d);
            }
            for i in 0..16 {
                let want = crate::lossy::select_lossless_header(
                    mins[i],
                    dists[i],
                    top_min[i],
                    top_dist[i],
                );
                assert_eq!((got.0[i], got.1[i]), want, "block {i}");
                assert_eq!(lut[usize::from(got.1[i])].0, lut[usize::from(dists[i])].0);
            }
        }
    }
}
