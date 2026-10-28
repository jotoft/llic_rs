//! WebAssembly SIMD128 version of the grouped tile codec in `tile_sse.rs`,
//! with the same interface, streams and pixels.
//!
//! Most SSE operations have direct SIMD128 equivalents (`PSHUFB` is
//! `i8x16.swizzle`, which also zeroes out-of-range lanes). The packing
//! multiplies (`PMADDUBSW`, `PMADDWD`, `PMULUDQ`) all multiply by powers of
//! two, so they become shift-and-or steps by `bits`, `2 * bits` and
//! `4 * bits`; the Q15 quantization uses widening 16x16->32-bit multiplies.
//!
//! Compiled when the target enables `simd128` (WebAssembly has no run-time
//! feature detection).

use std::arch::wasm32::*;

use super::BucketLutEntry;

/// Always true: the module is only compiled with `simd128` enabled.
pub(super) fn available() -> bool {
    #[cfg(test)]
    if tests::DISABLED.with(std::cell::Cell::get) {
        return false;
    }
    true
}

/// Bucket LUT in the layout the SSE kernels use (see `tile_sse::packed_lut`):
/// bucket size in bits 0-7, bit count in bits 8-15, and the Q16 reciprocal
/// `ceil(65536 / bucket_size)` in bits 16-31 here (0 for bucket size 1).
pub(super) fn packed_lut(lut: &[BucketLutEntry; 256]) -> [u32; 256] {
    lut.map(|(bits, bucket_size)| {
        let bucket_size = u32::from(bucket_size);
        let reciprocal = if bucket_size > 1 {
            // At most 32768, so it fits the 16 bits.
            65536u32.div_ceil(bucket_size)
        } else {
            0
        };
        (reciprocal << 16) | (u32::from(bits) << 8) | bucket_size
    })
}

/// Pixel order of the LSB-first packed value sequence per bit count, as in
/// `tile_sse.rs`.
const IDENTITY: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
const COLUMN_MAJOR: [u8; 16] = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15];
static PACK_ORDER: [[u8; 16]; 9] = [
    IDENTITY,
    [1, 5, 9, 13, 0, 4, 8, 12, 3, 7, 11, 15, 2, 6, 10, 14],
    COLUMN_MAJOR,
    [0, 4, 8, 12, 2, 6, 10, 14, 1, 5, 9, 13, 3, 7, 11, 15],
    [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15],
    COLUMN_MAJOR,
    COLUMN_MAJOR,
    IDENTITY,
    IDENTITY,
];

/// Per bit count and 8-pixel half: the two packed bytes holding each value
/// and a multiplier moving it to bit 8 (see `tile_sse.rs`).
struct UnpackParams {
    gather: [[u8; 16]; 2],
    scale: [[u16; 8]; 2],
}

const NONE: u8 = 0x80;

static UNPACK_PARAMS: [UnpackParams; 9] = [
    UnpackParams {
        gather: [[NONE; 16], [NONE; 16]],
        scale: [[0; 8], [0; 8]],
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
    },
    UnpackParams {
        gather: [
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8],
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8],
        ],
        scale: [[256; 8], [16; 8]],
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
    },
    UnpackParams {
        gather: [
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8],
            [
                8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, NONE,
            ],
        ],
        scale: [[256; 8], [256; 8]],
    },
];

#[inline(always)]
fn bytes(table: &[u8; 16]) -> v128 {
    // SAFETY: the table holds 16 bytes; v128_load needs no alignment.
    unsafe { v128_load(table.as_ptr().cast()) }
}

#[inline(always)]
fn halves(table: &[u16; 8]) -> v128 {
    // SAFETY: as above.
    unsafe { v128_load(table.as_ptr().cast()) }
}

/// Transposes a 4x4 matrix of 32-bit lanes (see `tile_sse::transpose4x32`).
#[inline(always)]
fn transpose4x32(r: &mut [v128; 4]) {
    let t0 = i32x4_shuffle::<0, 4, 1, 5>(r[0], r[1]);
    let t1 = i32x4_shuffle::<0, 4, 1, 5>(r[2], r[3]);
    let t2 = i32x4_shuffle::<2, 6, 3, 7>(r[0], r[1]);
    let t3 = i32x4_shuffle::<2, 6, 3, 7>(r[2], r[3]);
    r[0] = i64x2_shuffle::<0, 2>(t0, t1);
    r[1] = i64x2_shuffle::<1, 3>(t0, t1);
    r[2] = i64x2_shuffle::<0, 2>(t2, t3);
    r[3] = i64x2_shuffle::<1, 3>(t2, t3);
}

/// Reduces column minima (`MAX = false`) or maxima of 16 blocks to one byte
/// per block (see `tile_sse::reduce_blocks`).
#[inline(always)]
fn reduce_blocks<const MAX: bool>(columns: &[v128; 4]) -> v128 {
    let combine = |a, b| {
        if MAX {
            u8x16_max(a, b)
        } else {
            u8x16_min(a, b)
        }
    };
    let column_major = bytes(&COLUMN_MAJOR);
    let half = |i: usize| {
        let a = i8x16_swizzle(columns[2 * i], column_major);
        let b = i8x16_swizzle(columns[2 * i + 1], column_major);
        combine(i64x2_shuffle::<0, 2>(a, b), i64x2_shuffle::<1, 3>(a, b))
    };
    let (h0, h1) = (half(0), half(1));
    combine(
        i32x4_shuffle::<0, 2, 4, 6>(h0, h1),
        i32x4_shuffle::<1, 3, 5, 7>(h0, h1),
    )
}

/// Sixteen blocks at a time; see `super::select_lossless_header`.
#[inline(always)]
fn select_lossless_header(mins: v128, dists: v128, top_min: v128, top_dist: v128) -> (v128, v128) {
    let nibble_mask = u8x16_splat(0x0F);
    let low_nibble = v128_and(dists, nibble_mask);
    let high_nibble = u8x16_shr(dists, 4);
    let lookup = |table: [u8; 16], index| i8x16_swizzle(bytes(&table), index);
    let hi = u8x16_max(
        lookup(
            [0, 1, 3, 3, 7, 7, 7, 7, 15, 15, 15, 15, 15, 15, 15, 15],
            low_nibble,
        ),
        lookup(
            [
                0, 31, 63, 63, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            ],
            high_nibble,
        ),
    );
    let lo = u8x16_max(
        lookup([0, 1, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8], low_nibble),
        lookup(
            [
                0, 16, 32, 32, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
            ],
            high_nibble,
        ),
    );
    let maxs = i8x16_add(mins, dists);
    let mins = u8x16_max(u8x16_min(top_min, mins), u8x16_sub_sat(maxs, hi));
    let dists = u8x16_min(u8x16_max(top_dist, lo), hi);
    (mins, dists)
}

/// Divides each byte by the bucket size: `(r * ceil(65536 / b)) >> 16`,
/// exact for byte residuals and bucket sizes up to 256.
#[inline(always)]
fn quantize(residuals: v128, reciprocal: u32) -> v128 {
    #[allow(clippy::cast_possible_truncation)] // at most 32768
    let reciprocal = u16x8_splat(reciprocal as u16);
    let divide = |words: v128| {
        let low = u32x4_shr(u32x4_extmul_low_u16x8(words, reciprocal), 16);
        let high = u32x4_shr(u32x4_extmul_high_u16x8(words, reciprocal), 16);
        u16x8_narrow_i32x4(low, high)
    };
    let low = divide(u16x8_extend_low_u8x16(residuals));
    let high = divide(u16x8_extend_high_u8x16(residuals));
    u8x16_narrow_i16x8(low, high)
}

/// Packs the residuals of one row-major block (see `tile_sse::encode_tile`).
/// Writes up to 16 bytes at `dst`; returns the packed size.
#[inline(always)]
unsafe fn encode_tile<const QUANTIZE: bool>(block: v128, lut_entry: u32, dst: *mut u8) -> usize {
    let bits = ((lut_entry >> 8) & 0xFF) as usize;
    let residuals = i8x16_swizzle(block, bytes(&PACK_ORDER[bits]));
    let packed = if bits == 8 {
        // Eight-bit tiles (bucket size one) store the residuals unchanged.
        residuals
    } else {
        let values = if QUANTIZE {
            quantize(residuals, lut_entry >> 16)
        } else {
            residuals
        };
        // Merge neighbouring values: byte pairs, then 16-bit and 32-bit pairs,
        // each upper value shifted above the lower one's bits.
        #[allow(clippy::cast_possible_truncation)] // bits <= 6
        let shift = bits as u32;
        let pairs = v128_or(
            v128_and(values, u16x8_splat(0x00FF)),
            u16x8_shl(u16x8_shr(values, 8), shift),
        );
        let quads = v128_or(
            v128_and(pairs, u32x4_splat(0xFFFF)),
            u32x4_shl(u32x4_shr(pairs, 16), 2 * shift),
        );
        v128_or(
            v128_and(quads, u64x2_splat(0xFFFF_FFFF)),
            u64x2_shl(u64x2_shr(quads, 32), 4 * shift),
        )
    };
    // Each half holds `bits` bytes; the second store overwrites the unused
    // bytes of the first.
    v128_store64_lane::<0>(packed, dst.cast());
    v128_store64_lane::<1>(packed, dst.add(bits).cast());
    2 * bits
}

/// Encodes the 16 blocks of one group (see `tile_sse::encode_group`).
#[inline(always)]
unsafe fn encode_group<const QUANTIZE: bool>(
    rows: [*const u8; 4],
    mins: v128,
    dists: v128,
    dist_bytes: &[u8],
    error_limit: u8,
    lut: &[u32; 256],
    dst: *mut u8,
) -> usize {
    // Blocks with dist <= error_limit use zero bits.
    let no_payload = i8x16_eq(
        u8x16_sub_sat(dists, u8x16_splat(error_limit)),
        u8x16_splat(0),
    );
    let payload_mask = !u32::from(i8x16_bitmask(no_payload));
    let mut written = 0usize;
    for chunk in 0..4usize {
        if (payload_mask >> (chunk * 4)) & 0xF == 0 {
            continue;
        }
        // Each block's minimum repeated over its four columns.
        #[allow(clippy::cast_possible_truncation)]
        let spread = i8x16_add(
            bytes(&[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]),
            u8x16_splat((chunk * 4) as u8),
        );
        let min = i8x16_swizzle(mins, spread);
        let mut blocks = rows.map(|row| i8x16_sub(v128_load(row.add(chunk * 16).cast()), min));
        transpose4x32(&mut blocks);
        for (block, &dist) in blocks.iter().zip(&dist_bytes[chunk * 4..chunk * 4 + 4]) {
            written += encode_tile::<QUANTIZE>(*block, lut[usize::from(dist)], dst.add(written));
        }
    }
    written
}

/// Encodes the whole 64-pixel groups of one row of blocks; see
/// `tile_sse::encode_row_groups` for the arguments and result.
///
/// # Safety
/// None beyond the asserted bounds; `unsafe` matches the SSE version.
///
/// # Panics
/// If the four rows do not fit in `src` or `dst` cannot take 16 bytes per block.
#[allow(clippy::too_many_arguments)]
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
        let mut column_mins = [u8x16_splat(0); 4];
        let mut column_maxs = [u8x16_splat(0); 4];
        for i in 0..4 {
            let [r0, r1, r2, r3] = row_ptrs.map(|row| v128_load(row.add(i * 16).cast()));
            column_mins[i] = u8x16_min(u8x16_min(r0, r1), u8x16_min(r2, r3));
            column_maxs[i] = u8x16_max(u8x16_max(r0, r1), u8x16_max(r2, r3));
        }
        let mut mins = reduce_blocks::<false>(&column_mins);
        let mut dists = i8x16_sub(reduce_blocks::<true>(&column_maxs), mins);
        if select_header {
            let above = block - blocks_per_row;
            (mins, dists) = select_lossless_header(
                mins,
                dists,
                v128_load(min_stream[above..above + 16].as_ptr().cast()),
                v128_load(dist_stream[above..above + 16].as_ptr().cast()),
            );
        }
        v128_store(min_stream[block..block + 16].as_mut_ptr().cast(), mins);
        v128_store(dist_stream[block..block + 16].as_mut_ptr().cast(), dists);
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

/// Decodes one block to row-major pixels (see `tile_sse::decode_tile`). Reads
/// 16 bytes at `src`; returns the pixels and the packed size.
#[inline(always)]
unsafe fn decode_tile<const LOSSLESS: bool>(
    src: *const u8,
    lut_entry: u32,
    min: u8,
) -> (v128, usize) {
    let bits = ((lut_entry >> 8) & 0xFF) as usize;
    let params = &UNPACK_PARAMS[bits];
    let packed = v128_load(src.cast());
    #[allow(clippy::cast_possible_truncation)]
    let value_mask = u16x8_splat(((1u32 << bits) - 1) as u16);
    let half = |i: usize| {
        let words = i8x16_swizzle(packed, bytes(&params.gather[i]));
        v128_and(
            u16x8_shr(i16x8_mul(words, halves(&params.scale[i])), 8),
            value_mask,
        )
    };
    let (h0, h1) = (half(0), half(1));
    let pixels = if LOSSLESS {
        u8x16_add_sat(u8x16_narrow_i16x8(h0, h1), u8x16_splat(min))
    } else {
        // Lossy streams add half the bucket size; products saturate like the
        // per-block decoder.
        let bucket_size = lut_entry & 0xFF;
        #[allow(clippy::cast_possible_truncation)]
        let base = u16x8_splat(u16::from(min.wrapping_add((bucket_size >> 1) as u8)));
        #[allow(clippy::cast_possible_truncation)]
        let scale = u16x8_splat(bucket_size as u16);
        u8x16_narrow_i16x8(
            i16x8_add(i16x8_mul(h0, scale), base),
            i16x8_add(i16x8_mul(h1, scale), base),
        )
    };
    (pixels, 2 * bits)
}

/// Decodes the 16-pixel groups of one row of blocks; see
/// `tile_sse::decode_row_groups` for the arguments and result.
///
/// # Safety
/// None beyond the asserted bounds; `unsafe` matches the SSE version.
///
/// # Panics
/// If the four rows do not fit in `dst` or a header stream is too short.
#[allow(clippy::too_many_arguments)]
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
    while group < groups && pixel_stream.len() - read >= 64 {
        let block_mins = &mins[group * 4..group * 4 + 4];
        let entries: [u32; 4] = std::array::from_fn(|i| lut[usize::from(dists[group * 4 + i])]);
        let rows = if entries.iter().fold(0, |acc, e| acc | e) & 0xFF00 == 0 {
            // Four blocks without payload, common in flat or low-quality regions.
            let bases = std::array::from_fn::<u8, 4, _>(|i| {
                #[allow(clippy::cast_possible_truncation)]
                let midpoint = (((entries[i] & 0xFF) >> 1) & midpoint_mask) as u8;
                block_mins[i].wrapping_add(midpoint)
            });
            let row = i8x16_swizzle(
                u32x4_splat(u32::from_le_bytes(bases)),
                bytes(&[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]),
            );
            [row; 4]
        } else {
            let src = pixel_stream.as_ptr().add(read);
            let mut offset = 0usize;
            let mut blocks = [u8x16_splat(0); 4];
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
            v128_store(row.add(group * 16).cast(), pixels);
        }
        group += 1;
    }
    (group * 16, read)
}

#[cfg(test)]
pub(super) mod tests {
    use std::cell::Cell;

    thread_local! {
        /// Makes [`super::available`] report false on this thread, to compare
        /// the SIMD128 path with the per-block code.
        pub(in crate::lossy) static DISABLED: Cell<bool> = const { Cell::new(false) };
    }
}
