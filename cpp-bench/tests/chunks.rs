//! The C++ decoder reads the Rust encoder's multi-chunk streams.

use llic::{FormatVersion, LlicContext, Mode, Quality};
use llic_cpp_bench::CppLlicContext;

#[test]
fn cpp_decodes_rust_chunks() {
    // 76 rows split 24, 24, 28 in three chunks.
    let (width, height) = (72u32, 76u32);
    let src: Vec<u8> = (0..width * height)
        .map(|i| ((i * 7919) % 251 + (i % width) * 2) as u8)
        .collect();
    let mut cpp = CppLlicContext::new(width, height, 1).expect("C++ context");
    for chunks in [1, 2, 3, 5, 16] {
        let rust = LlicContext::new(width, height, width, Some(chunks)).unwrap();
        let mut buf = vec![0u8; rust.compressed_buffer_size()];
        let mut decoded = vec![0u8; src.len()];
        for version in [FormatVersion::V3, FormatVersion::V4, FormatVersion::V5] {
            for quality in [
                Quality::Lossless,
                Quality::VeryHigh,
                Quality::VeryLow,
                Quality::LosslessEntropy,
            ] {
                for mode in [Mode::Default, Mode::Fast, Mode::Dynamic] {
                    // C++ rejects lossless entropy in fast mode (Rust writes it
                    // with mode byte 1); v3 has no dynamic mode.
                    let entropy_fast = quality == Quality::LosslessEntropy && mode == Mode::Fast;
                    if entropy_fast || (version == FormatVersion::V3 && mode == Mode::Dynamic) {
                        continue;
                    }
                    let case = format!("{version:?} {quality:?} {mode:?} {chunks} chunks");
                    let size = rust
                        .compress_gray8_versioned(&src, quality, mode, version, &mut buf)
                        .unwrap();
                    rust.decompress_gray8(&buf[..size], &mut decoded).unwrap();
                    let (cpp_decoded, _, _) = cpp
                        .decompress(&buf[..size])
                        .unwrap_or_else(|e| panic!("{case}: C++ decode {e:?}"));
                    assert_eq!(cpp_decoded, decoded, "{case}");
                }
            }
        }
    }
}
