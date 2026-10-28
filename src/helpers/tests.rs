use super::color::{self, SIMD_DISABLED};
use super::*;

/// Deterministic pseudo-random bytes.
fn noise(len: usize, seed: u32) -> Vec<u8> {
    let mut state = seed.wrapping_mul(0x9E37_79B9) | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state >> 24) as u8
        })
        .collect()
}

/// Runs `f` with the SIMD kernels and with the scalar code only.
fn with_and_without_simd<T>(f: impl Fn() -> T) -> (T, T) {
    let simd = f();
    SIMD_DISABLED.with(|disabled| disabled.set(true));
    let scalar = f();
    SIMD_DISABLED.with(|disabled| disabled.set(false));
    (simd, scalar)
}

type Split3 = fn(Rows, u32, u32, [RowsMut; 3]) -> Result<(), BufferError>;
type Merge3 = fn([Rows; 3], u32, u32, RowsMut) -> Result<(), BufferError>;

const SPLITS3: [(&str, Split3); 3] = [
    ("rgb", deinterleave_rgb),
    ("ycocg_r", deinterleave_rgb_to_ycocg_r),
    ("ycocg", deinterleave_rgb_to_ycocg),
];
const MERGES3: [(&str, Merge3); 3] = [
    ("rgb", interleave_rgb),
    ("ycocg_r", interleave_ycocg_r_to_rgb),
    ("ycocg", interleave_ycocg_to_rgb),
];

fn split3(split: Split3, rgb: &[u8], width: u32, height: u32) -> [Vec<u8>; 3] {
    let size = (width * height) as usize;
    let mut planes = [vec![0; size], vec![0; size], vec![0; size]];
    let [p0, p1, p2] = planes.each_mut().map(|p| RowsMut::packed(p));
    split(Rows::packed(rgb), width, height, [p0, p1, p2]).expect("split");
    planes
}

fn merge3(merge: Merge3, planes: &[Vec<u8>; 3], width: u32, height: u32) -> Vec<u8> {
    let mut rgb = vec![0; (width * height * 3) as usize];
    let rows = planes.each_ref().map(|p| Rows::packed(p));
    merge(rows, width, height, RowsMut::packed(&mut rgb)).expect("merge");
    rgb
}

/// Every 24-bit color, as 256 rows of 65536 pixels with red equal to the row.
fn all_colors() -> Vec<u8> {
    (0..1u32 << 24)
        .flat_map(|c| [(c >> 16) as u8, (c >> 8) as u8, c as u8])
        .collect()
}

#[test]
fn ycocg_r_round_trips_every_color() {
    let rgb = all_colors();
    let planes = split3(deinterleave_rgb_to_ycocg_r, &rgb, 1 << 16, 256);
    assert!(merge3(interleave_ycocg_r_to_rgb, &planes, 1 << 16, 256) == rgb);
}

#[test]
fn ycocg_restores_red_exactly_and_green_and_blue_within_one() {
    let rgb = all_colors();
    let planes = split3(deinterleave_rgb_to_ycocg, &rgb, 1 << 16, 256);
    let restored = merge3(interleave_ycocg_to_rgb, &planes, 1 << 16, 256);
    for (before, after) in rgb.chunks_exact(3).zip(restored.chunks_exact(3)) {
        assert_eq!(before[0], after[0], "{before:?} -> {after:?}");
        assert!(before[1].abs_diff(after[1]) <= 1, "{before:?} -> {after:?}");
        assert!(before[2].abs_diff(after[2]) <= 1, "{before:?} -> {after:?}");
    }
}

#[test]
fn grays_map_to_neutral_chroma() {
    for gray in 0..=255 {
        assert_eq!(color::rgb_to_ycocg_r(gray, gray, gray), [gray, 128, 128]);
        assert_eq!(color::rgb_to_ycocg(gray, gray, gray), [gray, 128, 128]);
    }
}

/// The SIMD kernels give the scalar results, on rows with a partial block.
#[test]
fn simd_matches_scalar() {
    for width in [1, 15, 16, 17, 47, 64, 100] {
        let height = 3;
        let rgb = noise((width * height * 3) as usize, width);
        for ((name, split), (_, merge)) in SPLITS3.into_iter().zip(MERGES3) {
            let (simd, scalar) = with_and_without_simd(|| split3(split, &rgb, width, height));
            assert_eq!(simd, scalar, "{name} split, width {width}");
            let (simd, scalar) = with_and_without_simd(|| merge3(merge, &simd, width, height));
            assert_eq!(simd, scalar, "{name} merge, width {width}");
        }

        let rgba = noise((width * height * 4) as usize, width + 1);
        let size = (width * height) as usize;
        let split4 = || {
            let mut planes = [vec![0; size], vec![0; size], vec![0; size], vec![0; size]];
            let rows = planes.each_mut().map(|p| RowsMut::packed(p));
            deinterleave_rgba(Rows::packed(&rgba), width, height, rows).expect("split");
            planes
        };
        let (planes, scalar) = with_and_without_simd(split4);
        assert_eq!(planes, scalar, "rgba split, width {width}");
        let merge4 = || {
            let mut out = vec![0; rgba.len()];
            let rows = planes.each_ref().map(|p| Rows::packed(p));
            interleave_rgba(rows, width, height, RowsMut::packed(&mut out)).expect("merge");
            out
        };
        let (simd, scalar) = with_and_without_simd(merge4);
        assert_eq!(simd, scalar, "rgba merge, width {width}");
        assert_eq!(simd, rgba, "rgba round trip, width {width}");
    }
}

#[test]
fn rgb_planes_hold_the_channels() {
    let rgb = noise(20 * 2 * 3, 7);
    let planes = split3(deinterleave_rgb, &rgb, 20, 2);
    for (i, pixel) in rgb.chunks_exact(3).enumerate() {
        assert_eq!(
            [planes[0][i], planes[1][i], planes[2][i]],
            [pixel[0], pixel[1], pixel[2]]
        );
    }
}

/// Strided rows are read and written in place; the padding stays untouched.
#[test]
fn strides_skip_padding() {
    let (width, height) = (20u32, 3u32);
    let src_stride = 70;
    let mut padded = vec![0xEE; src_stride * height as usize];
    let packed = noise((width * height * 3) as usize, 3);
    for (row, src) in padded.chunks_mut(src_stride).zip(packed.chunks(60)) {
        row[..60].copy_from_slice(src);
    }
    let plane_stride = 24;
    let mut planes = [
        vec![0xAA; plane_stride * 3],
        vec![0xAA; plane_stride * 3],
        vec![0xAA; 72],
    ];
    let rows = planes
        .each_mut()
        .map(|p| RowsMut::with_stride(p, plane_stride));
    deinterleave_rgb_to_ycocg_r(Rows::with_stride(&padded, src_stride), width, height, rows)
        .expect("split");
    let expected = split3(deinterleave_rgb_to_ycocg_r, &packed, width, height);
    for (plane, expected) in planes.iter().zip(&expected) {
        for (row, expected) in plane.chunks(plane_stride).zip(expected.chunks(20)) {
            assert_eq!(&row[..20], expected);
            assert!(row[20..].iter().all(|&b| b == 0xAA));
        }
    }

    let mut out = vec![0xEE; padded.len()];
    let rows = planes
        .each_ref()
        .map(|p| Rows::with_stride(p, plane_stride));
    interleave_ycocg_r_to_rgb(
        rows,
        width,
        height,
        RowsMut::with_stride(&mut out, src_stride),
    )
    .expect("merge");
    assert_eq!(out, padded);
}

#[test]
fn buffer_errors_name_the_buffer() {
    let rgb = vec![0; 8 * 4 * 3];
    let mut planes = [vec![0; 32], vec![0; 32], vec![0; 31]];
    let rows = planes.each_mut().map(|p| RowsMut::packed(p));
    let result = deinterleave_rgb(Rows::packed(&rgb), 8, 4, rows);
    let expected = BufferError::TooSmall {
        buffer: Buffer::Plane(2),
        len: 31,
        needed: 32,
    };
    assert_eq!(result, Err(expected));
    assert_eq!(expected.to_string(), "plane 2 has 31 bytes but needs 32");

    let mut planes = [vec![0; 32], vec![0; 32], vec![0; 32]];
    let rows = planes.each_mut().map(|p| RowsMut::packed(p));
    let result = deinterleave_rgb(Rows::with_stride(&rgb, 23), 8, 4, rows);
    let expected = BufferError::StrideTooShort {
        buffer: Buffer::Source,
        stride: 23,
        row_bytes: 24,
    };
    assert_eq!(result, Err(expected));

    let rows = planes.each_mut().map(|p| RowsMut::packed(p));
    let result = deinterleave_rgb(Rows::packed(&rgb[1..]), 8, 4, rows);
    let expected = BufferError::TooSmall {
        buffer: Buffer::Source,
        len: 95,
        needed: 96,
    };
    assert_eq!(result, Err(expected));

    let mut out = vec![0; 95];
    let rows = planes.each_ref().map(|p| Rows::packed(p));
    let result = interleave_rgb(rows, 8, 4, RowsMut::packed(&mut out));
    let expected = BufferError::TooSmall {
        buffer: Buffer::Destination,
        len: 95,
        needed: 96,
    };
    assert_eq!(result, Err(expected));

    // A last row needs only its own bytes, not a whole stride.
    let rows = planes.each_mut().map(|p| RowsMut::packed(p));
    let short_last_row = vec![0; 3 * 30 + 24];
    deinterleave_rgb(Rows::with_stride(&short_last_row, 30), 8, 4, rows).expect("split");
}

#[test]
fn oversized_images_are_rejected() {
    let rows = [(); 3].map(|()| RowsMut::packed(&mut []));
    let result = deinterleave_rgb(Rows::packed(&[]), u32::MAX, u32::MAX, rows);
    // 32-bit targets overflow on the row, 64-bit ones on the whole image.
    assert_eq!(result, Err(BufferError::TooLarge));
}

#[test]
fn plane_sizes_follow_the_layouts() {
    let size = |width, height| PlaneSize { width, height };
    let sizes = |layout| plane_sizes(layout, 64, 32).expect("sizes").to_vec();
    assert_eq!(sizes(PlaneLayout::Rgb), [size(64, 32); 3]);
    assert_eq!(sizes(PlaneLayout::Rgba), [size(64, 32); 4]);
    assert_eq!(sizes(PlaneLayout::HighLow), [size(64, 32); 2]);
    assert_eq!(
        sizes(PlaneLayout::Bayer2x2),
        [size(32, 16), size(32, 32), size(32, 16)]
    );
    assert_eq!(
        sizes(PlaneLayout::RgbIr4x4),
        [size(16, 16), size(32, 32), size(16, 16), size(32, 16)]
    );
    assert_eq!(
        sizes(PlaneLayout::YCoCg420),
        [size(64, 32), size(32, 16), size(32, 16)]
    );

    for (layout, width, height) in [
        (PlaneLayout::Bayer2x2, 63, 32),
        (PlaneLayout::YCoCg420, 64, 31),
        (PlaneLayout::RgbIr4x4, 62, 32),
        (PlaneLayout::RgbIr4x4, 64, 30),
    ] {
        let expected = DimensionsError {
            layout,
            width,
            height,
        };
        assert_eq!(plane_sizes(layout, width, height), Err(expected));
    }
    assert_eq!(plane_sizes(PlaneLayout::Rgb, 5, 3).expect("sizes").len(), 3);
    let error = plane_sizes(PlaneLayout::Bayer2x2, 63, 32).unwrap_err();
    assert_eq!(error.to_string(), "63x32 does not fit the Bayer2x2 layout");
}
