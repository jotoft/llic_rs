//! Criterion benchmarks comparing Rust and C++ LLIC implementations.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use llic::{LlicContext, Quality, ffi::{CppLlicContext, LlicQuality, LlicMode}};

/// Generate a gradient test image of the specified size
fn generate_gradient_image(width: u32, height: u32) -> Vec<u8> {
    let mut data = vec![0u8; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            data[idx] = ((x + y) % 256) as u8;
        }
    }
    data
}

/// Generate a random-ish test image (deterministic pattern)
fn generate_pattern_image(width: u32, height: u32) -> Vec<u8> {
    let mut data = vec![0u8; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            // Create a complex pattern that's still deterministic
            let val = ((x * 7 + y * 13) ^ (x * y)) % 256;
            data[idx] = val as u8;
        }
    }
    data
}

/// Read test data from the actual llic file format
fn read_llic_file(path: &str) -> Option<(u32, u32, Vec<u8>)> {
    use std::fs::File;
    use std::io::Read;

    let mut file = File::open(path).ok()?;
    let mut data = Vec::new();
    file.read_to_end(&mut data).ok()?;

    // Parse the text header "width height\ncompressed_size\n"
    let first_newline = data.iter().position(|&b| b == b'\n')?;
    let header_str = std::str::from_utf8(&data[..first_newline]).ok()?;
    let parts: Vec<&str> = header_str.split_whitespace().collect();
    if parts.len() != 2 {
        return None;
    }
    let width: u32 = parts[0].parse().ok()?;
    let height: u32 = parts[1].parse().ok()?;

    let second_start = first_newline + 1;
    let second_newline = data[second_start..].iter().position(|&b| b == b'\n')? + second_start;

    let compressed_data = data[second_newline + 1..].to_vec();

    Some((width, height, compressed_data))
}

fn benchmark_decompression(c: &mut Criterion) {
    let mut group = c.benchmark_group("decompression");

    // Test with different image sizes
    let sizes: &[(u32, u32, &str)] = &[
        (64, 64, "64x64"),
        (256, 256, "256x256"),
        (512, 512, "512x512"),
        (1024, 1024, "1024x1024"),
    ];

    for &(width, height, label) in sizes {
        let pixel_count = (width * height) as usize;
        group.throughput(Throughput::Bytes(pixel_count as u64));

        // Generate test image and compress with C++
        let original = generate_gradient_image(width, height);

        // Use C++ to compress (so we have valid compressed data)
        let mut cpp_ctx = match CppLlicContext::new(width, height, 1) {
            Ok(ctx) => ctx,
            Err(e) => {
                eprintln!("Failed to create C++ context for {}: {:?}", label, e);
                continue;
            }
        };

        let compressed = match cpp_ctx.compress(&original, LlicQuality::Lossless, LlicMode::Default) {
            Ok(data) => data,
            Err(e) => {
                eprintln!("Failed to compress with C++ for {}: {:?}", label, e);
                continue;
            }
        };

        // Benchmark Rust decompression
        let rust_ctx = LlicContext::new(width, height, width, Some(1)).unwrap();

        group.bench_with_input(
            BenchmarkId::new("rust", label),
            &compressed,
            |b, compressed| {
                let mut output = vec![0u8; pixel_count];
                b.iter(|| {
                    rust_ctx.decompress_gray8(black_box(compressed), black_box(&mut output)).unwrap();
                });
            },
        );

        // Benchmark C++ decompression
        group.bench_with_input(
            BenchmarkId::new("cpp", label),
            &compressed,
            |b, compressed| {
                b.iter(|| {
                    cpp_ctx.decompress(black_box(compressed)).unwrap();
                });
            },
        );
    }

    group.finish();
}

fn benchmark_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression");

    let sizes: &[(u32, u32, &str)] = &[
        (64, 64, "64x64"),
        (256, 256, "256x256"),
        (512, 512, "512x512"),
        (1024, 1024, "1024x1024"),
    ];

    for &(width, height, label) in sizes {
        let pixel_count = (width * height) as usize;
        group.throughput(Throughput::Bytes(pixel_count as u64));

        let image = generate_gradient_image(width, height);

        // Only benchmark C++ compression (Rust not implemented yet)
        let mut cpp_ctx = match CppLlicContext::new(width, height, 1) {
            Ok(ctx) => ctx,
            Err(e) => {
                eprintln!("Failed to create C++ context for {}: {:?}", label, e);
                continue;
            }
        };

        group.bench_with_input(
            BenchmarkId::new("cpp_lossless", label),
            &image,
            |b, image| {
                b.iter(|| {
                    cpp_ctx.compress(black_box(image), LlicQuality::Lossless, LlicMode::Default).unwrap();
                });
            },
        );

        // Also benchmark lossy compression
        group.bench_with_input(
            BenchmarkId::new("cpp_lossy_high", label),
            &image,
            |b, image| {
                b.iter(|| {
                    cpp_ctx.compress(black_box(image), LlicQuality::High, LlicMode::Default).unwrap();
                });
            },
        );
    }

    group.finish();
}

fn benchmark_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("roundtrip");

    // Benchmark a complete compress-decompress cycle
    let sizes: &[(u32, u32, &str)] = &[
        (256, 256, "256x256"),
        (512, 512, "512x512"),
    ];

    for &(width, height, label) in sizes {
        let pixel_count = (width * height) as usize;
        group.throughput(Throughput::Bytes(pixel_count as u64));

        let image = generate_pattern_image(width, height);

        let mut cpp_ctx = match CppLlicContext::new(width, height, 1) {
            Ok(ctx) => ctx,
            Err(e) => {
                eprintln!("Failed to create C++ context for {}: {:?}", label, e);
                continue;
            }
        };

        group.bench_with_input(
            BenchmarkId::new("cpp", label),
            &image,
            |b, image| {
                b.iter(|| {
                    let compressed = cpp_ctx.compress(black_box(image), LlicQuality::Lossless, LlicMode::Default).unwrap();
                    cpp_ctx.decompress(black_box(&compressed)).unwrap()
                });
            },
        );
    }

    group.finish();
}

fn benchmark_real_file(c: &mut Criterion) {
    let mut group = c.benchmark_group("real_file");

    // Test with the actual 64x64 pattern file
    if let Some((width, height, compressed)) = read_llic_file("test_data/patterns_64x64_q0.llic") {
        let pixel_count = (width * height) as usize;
        group.throughput(Throughput::Bytes(pixel_count as u64));

        let rust_ctx = LlicContext::new(width, height, width, Some(1)).unwrap();
        let mut cpp_ctx = CppLlicContext::new(width, height, 1).unwrap();

        group.bench_function("rust_patterns_64x64", |b| {
            let mut output = vec![0u8; pixel_count];
            b.iter(|| {
                rust_ctx.decompress_gray8(black_box(&compressed), black_box(&mut output)).unwrap();
            });
        });

        group.bench_function("cpp_patterns_64x64", |b| {
            b.iter(|| {
                cpp_ctx.decompress(black_box(&compressed)).unwrap();
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_decompression,
    benchmark_compression,
    benchmark_roundtrip,
    benchmark_real_file,
);

criterion_main!(benches);
