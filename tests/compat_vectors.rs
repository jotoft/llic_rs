//! Decoder and encoder compatibility with the C++ reference's vectors.
//!
//! `test_data/vectors` is a copy of `ext/testdata/vectors` from the C++ llic
//! repository (v4.1.1): streams written by every C++ release from 2.0.0, with
//! the FNV-1a 64 hash of the image the writing release decoded. See
//! `manifest.txt` for the columns.

use std::fs;
use std::path::{Path, PathBuf};

use llic::pgm::Pgm;
use llic::{FormatVersion, LlicContext, LlicError, Mode, Quality};

/// One manifest line.
struct Vector {
    file: String,
    input: String,
    quality: u8,
    mode: Mode,
    chunks: u32,
    format: u8,
    hash: u64,
    decoded_quality: u8,
    decoded_mode: Mode,
    /// C++ releases that write this exact stream.
    releases: Vec<String>,
}

fn vectors_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("test_data/vectors")
}

fn parse_mode(name: &str) -> Mode {
    match name {
        "default" => Mode::Default,
        "fast" => Mode::Fast,
        "dynamic" => Mode::Dynamic,
        other => panic!("unknown mode {other:?} in manifest"),
    }
}

fn parse_quality(value: u8) -> Quality {
    match value {
        0 => Quality::Lossless,
        2 => Quality::VeryHigh,
        4 => Quality::High,
        8 => Quality::Medium,
        16 => Quality::Low,
        32 => Quality::VeryLow,
        64 => Quality::LosslessEntropy,
        other => panic!("unknown quality {other} in manifest"),
    }
}

fn load_manifest() -> Vec<Vector> {
    let text = fs::read_to_string(vectors_dir().join("manifest.txt")).expect("read manifest");
    text.lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(|line| {
            let fields: Vec<&str> = line.split_whitespace().collect();
            assert_eq!(fields.len(), 10, "malformed manifest line: {line}");
            Vector {
                file: fields[0].to_string(),
                input: fields[1].to_string(),
                quality: fields[2].parse().expect("quality"),
                mode: parse_mode(fields[3]),
                chunks: fields[4].parse().expect("chunks"),
                format: fields[5].parse().expect("format"),
                hash: u64::from_str_radix(fields[6], 16).expect("hash"),
                decoded_quality: fields[7].parse().expect("decoded quality"),
                decoded_mode: parse_mode(fields[8]),
                releases: fields[9].split(',').map(str::to_string).collect(),
            }
        })
        .collect()
}

/// Width and height from an input name such as `peter_face_64x76`.
fn input_dimensions(input: &str) -> (u32, u32) {
    let size = input.rsplit('_').next().expect("size suffix");
    let (width, height) = size.split_once('x').expect("WxH suffix");
    (
        width.parse().expect("width"),
        height.parse().expect("height"),
    )
}

fn fnv1a64(data: &[u8]) -> u64 {
    data.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

#[test]
fn manifest_covers_formats_2_to_5() {
    let vectors = load_manifest();
    assert_eq!(vectors.len(), 227);
    for format in 2..=5 {
        assert!(
            vectors.iter().any(|v| v.format == format),
            "no vectors for format {format}"
        );
    }
}

#[test]
fn decodes_cpp_vectors_from_format_3_to_5() {
    let mut failures = Vec::new();
    let mut decoded = 0;
    for vector in load_manifest().iter().filter(|v| v.format >= 3) {
        let (width, height) = input_dimensions(&vector.input);
        let stream = fs::read(vectors_dir().join(&vector.file)).expect("read stream");
        let ctx = LlicContext::new(width, height, width, Some(1)).expect("context");
        let mut image = vec![0u8; (width * height) as usize];
        // Entropy-coded streams (stream type byte zero) were reported as lossless
        // before C++ API 3.0 added a quality for them; the C++ test allows the same.
        let expected_quality = if stream.len() > 2
            && stream[2] == 0
            && vector.decoded_quality == Quality::Lossless as u8
        {
            Quality::LosslessEntropy as u8
        } else {
            vector.decoded_quality
        };
        match ctx.decompress_gray8(&stream, &mut image) {
            Ok((quality, mode)) => {
                let hash = fnv1a64(&image);
                if hash != vector.hash
                    || quality as u8 != expected_quality
                    || mode != vector.decoded_mode
                {
                    failures.push(format!(
                        "{}: got hash {hash:016x} q{} {mode:?}, want {:016x} q{} {:?}",
                        vector.file,
                        quality as u8,
                        vector.hash,
                        expected_quality,
                        vector.decoded_mode
                    ));
                }
            }
            Err(err) => failures.push(format!("{}: {err}", vector.file)),
        }
        decoded += 1;
    }
    assert!(decoded > 0);
    assert!(
        failures.is_empty(),
        "{} of {decoded} vectors failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn rejects_format_2_vectors() {
    for vector in load_manifest().iter().filter(|v| v.format == 2) {
        let (width, height) = input_dimensions(&vector.input);
        let stream = fs::read(vectors_dir().join(&vector.file)).expect("read stream");
        let ctx = LlicContext::new(width, height, width, Some(1)).expect("context");
        let mut image = vec![0u8; (width * height) as usize];
        assert!(
            matches!(
                ctx.decompress_gray8(&stream, &mut image),
                Err(LlicError::UnsupportedFormat)
            ),
            "{} should be rejected",
            vector.file
        );
    }
}

/// Single-chunk streams from the last C++ release of each format must match ours
/// byte for byte.
#[test]
fn encodes_formats_4_and_5_like_cpp() {
    let last_release = |format: FormatVersion| match format {
        FormatVersion::V4 => "v3.4.0",
        FormatVersion::V5 => "v4.1.1",
        FormatVersion::V3 => unreachable!("not compared"),
    };
    let mut failures = Vec::new();
    let mut compared = 0;
    for vector in load_manifest().iter().filter(|v| v.chunks == 1) {
        let version = match vector.format {
            4 => FormatVersion::V4,
            5 => FormatVersion::V5,
            _ => continue,
        };
        if !vector.releases.iter().any(|r| r == last_release(version)) {
            continue;
        }
        let (width, height) = input_dimensions(&vector.input);
        let input = Pgm::open(vectors_dir().join(format!("inputs/{}.pgm", vector.input)))
            .expect("read input");
        let expected = fs::read(vectors_dir().join(&vector.file)).expect("read stream");

        let ctx = LlicContext::new(width, height, width, Some(1)).expect("context");
        let mut output = vec![0u8; ctx.compressed_buffer_size()];
        let size = ctx
            .compress_gray8_versioned(
                input.data(),
                parse_quality(vector.quality),
                vector.mode,
                version,
                &mut output,
            )
            .expect("compress");
        if output[..size] != expected[..] {
            failures.push(format!(
                "{}: {size} bytes, want {}",
                vector.file,
                expected.len()
            ));
        }
        compared += 1;
    }
    assert_eq!(compared, 80, "40 single-chunk streams per format");
    assert!(
        failures.is_empty(),
        "{} of {compared} streams differ:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
