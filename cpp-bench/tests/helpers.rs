//! The Rust color helpers give the C++ helpers' bytes.

use std::path::Path;

use llic::helpers::{self, Rows, RowsMut};
use llic_cpp_bench::helpers::ColorHelper;

const HELPERS: [ColorHelper; 4] = [
    ColorHelper::Rgb,
    ColorHelper::Rgba,
    ColorHelper::YCoCgR,
    ColorHelper::YCoCg,
];

fn rust_split(helper: ColorHelper, src: &[u8], width: u32, height: u32) -> Vec<Vec<u8>> {
    let size = width as usize * height as usize;
    let mut planes = vec![vec![0u8; size]; helper.channels()];
    let src = Rows::packed(src);
    let result = match &mut planes[..] {
        [r, g, b, a] => {
            helpers::deinterleave_rgba(src, width, height, [r, g, b, a].map(|p| RowsMut::packed(p)))
        }
        [p0, p1, p2] => {
            let planes = [p0, p1, p2].map(|p| RowsMut::packed(p));
            match helper {
                ColorHelper::Rgb => helpers::deinterleave_rgb(src, width, height, planes),
                ColorHelper::YCoCgR => {
                    helpers::deinterleave_rgb_to_ycocg_r(src, width, height, planes)
                }
                ColorHelper::YCoCg => {
                    helpers::deinterleave_rgb_to_ycocg(src, width, height, planes)
                }
                ColorHelper::Rgba => unreachable!("RGBA has four planes"),
            }
        }
        _ => unreachable!("three or four planes"),
    };
    result.expect("Rust split");
    planes
}

fn rust_merge(helper: ColorHelper, planes: &[Vec<u8>], width: u32, height: u32) -> Vec<u8> {
    let mut dst = vec![0u8; width as usize * height as usize * helper.channels()];
    let out = RowsMut::packed(&mut dst);
    let result = match planes {
        [r, g, b, a] => {
            helpers::interleave_rgba([r, g, b, a].map(|p| Rows::packed(p)), width, height, out)
        }
        [p0, p1, p2] => {
            let planes = [p0, p1, p2].map(|p| Rows::packed(p));
            match helper {
                ColorHelper::Rgb => helpers::interleave_rgb(planes, width, height, out),
                ColorHelper::YCoCgR => {
                    helpers::interleave_ycocg_r_to_rgb(planes, width, height, out)
                }
                ColorHelper::YCoCg => helpers::interleave_ycocg_to_rgb(planes, width, height, out),
                ColorHelper::Rgba => unreachable!("RGBA has four planes"),
            }
        }
        _ => unreachable!("three or four planes"),
    };
    result.expect("Rust merge");
    dst
}

/// Splits `pixels` with both libraries, merges the planes back with both, and
/// merges arbitrary bytes as planes too (outside the forward transform's range).
fn assert_same(helper: ColorHelper, pixels: &[u8], width: u32, height: u32, what: &str) {
    let planes = helper.split(pixels, width, height).expect("C++ split");
    assert!(
        rust_split(helper, pixels, width, height) == planes,
        "{helper:?} split differs: {what}"
    );
    let merged = helper.merge(&planes, width, height).expect("C++ merge");
    assert!(
        rust_merge(helper, &planes, width, height) == merged,
        "{helper:?} merge differs: {what}"
    );
    let raw: Vec<Vec<u8>> = pixels
        .chunks_exact(width as usize * height as usize)
        .map(<[u8]>::to_vec)
        .collect();
    let merged = helper.merge(&raw, width, height).expect("C++ merge");
    assert!(
        rust_merge(helper, &raw, width, height) == merged,
        "{helper:?} merge of raw bytes differs: {what}"
    );
}

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

#[test]
fn every_color_matches_cpp() {
    // 2^24 colors as 256 rows of 65536 pixels; RGBA adds an alpha ramp.
    let (width, height) = (1 << 16, 256);
    for helper in HELPERS {
        let pixels: Vec<u8> = (0..1u32 << 24)
            .flat_map(|c| [(c >> 16) as u8, (c >> 8) as u8, c as u8, (c >> 4) as u8])
            .enumerate()
            .filter(|(i, _)| i % 4 < helper.channels())
            .map(|(_, byte)| byte)
            .collect();
        assert_same(helper, &pixels, width, height, "every color");
    }
}

#[test]
fn noise_matches_cpp_at_any_width() {
    for helper in HELPERS {
        for width in [1, 3, 8, 15, 16, 17, 31, 33, 100, 1921] {
            let height = 5;
            let len = width as usize * height as usize * helper.channels();
            assert_same(
                helper,
                &noise(len, width),
                width,
                height,
                &format!("width {width}"),
            );
        }
    }
}

#[test]
fn photos_match_cpp() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../test_data/photos");
    for name in ["rattan_960x640.webp", "rock_960x640.webp"] {
        let rgba = image::open(dir.join(name))
            .expect("decode photo")
            .into_rgba8();
        let (width, height) = rgba.dimensions();
        for helper in HELPERS {
            let pixels: Vec<u8> = rgba
                .as_raw()
                .iter()
                .enumerate()
                .filter(|(i, _)| i % 4 < helper.channels())
                .map(|(_, &byte)| byte)
                .collect();
            assert_same(helper, &pixels, width, height, name);
        }
    }
}
