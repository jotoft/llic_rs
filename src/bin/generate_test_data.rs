//! Generate test patterns and reference data for LLIC testing.
//!
//! This binary:
//! 1. Creates various test patterns as PGM files
//! 2. Compresses them with C++ LLIC at all quality levels (0-4)
//! 3. Decompresses with C++ to create reference outputs
//!
//! The Rust implementation should produce byte-identical output to these references.
//!
//! Run with: cargo run --bin generate_test_data

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let llic_compress = project_root.join("llic/build/src/llic_compress/llic_compress");
    let test_data = project_root.join("test_data");
    let patterns_dir = test_data.join("patterns");
    let compressed_dir = test_data.join("compressed");
    let reference_dir = test_data.join("reference");

    // Check if llic_compress exists
    if !llic_compress.exists() {
        eprintln!(
            "Error: llic_compress not found at {}",
            llic_compress.display()
        );
        eprintln!("Please build the C++ LLIC library first:");
        eprintln!("  cd llic/build && cmake .. && make");
        std::process::exit(1);
    }

    // Create directories
    fs::create_dir_all(&patterns_dir)?;
    fs::create_dir_all(&compressed_dir)?;
    fs::create_dir_all(&reference_dir)?;

    // Generate all test patterns
    println!("=== Generating test patterns ===");
    let patterns = generate_patterns(&patterns_dir)?;
    println!("Generated {} patterns\n", patterns.len());

    // Compress and decompress each pattern at each quality level
    println!("=== Compressing patterns at all quality levels ===");
    for pattern in &patterns {
        let base = pattern.file_stem().unwrap().to_str().unwrap();

        for quality in 0..=4 {
            let llic_file = compressed_dir.join(format!("{}_q{}.llic", base, quality));
            let ref_file = reference_dir.join(format!("{}_q{}.pgm", base, quality));

            // Compress
            let status = Command::new(&llic_compress)
                .args(["c", pattern.to_str().unwrap(), llic_file.to_str().unwrap()])
                .arg(quality.to_string())
                .output()?;

            if !status.status.success() {
                eprintln!(
                    "Warning: Failed to compress {} at quality {}",
                    base, quality
                );
                continue;
            }

            // Decompress to reference
            let status = Command::new(&llic_compress)
                .args(["d", llic_file.to_str().unwrap(), ref_file.to_str().unwrap()])
                .output()?;

            if !status.status.success() {
                eprintln!("Warning: Failed to decompress {}", llic_file.display());
                continue;
            }

            println!("  {} q{}: ok", base, quality);
        }
    }

    println!("\n=== Summary ===");
    println!(
        "Patterns:   {} files in {}",
        count_files(&patterns_dir, "pgm")?,
        patterns_dir.display()
    );
    println!(
        "Compressed: {} files in {}",
        count_files(&compressed_dir, "llic")?,
        compressed_dir.display()
    );
    println!(
        "References: {} files in {}",
        count_files(&reference_dir, "pgm")?,
        reference_dir.display()
    );
    println!("\nDone! Use these files for integration testing.");

    Ok(())
}

fn count_files(dir: &Path, ext: &str) -> Result<usize, std::io::Error> {
    Ok(fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|e| e == ext))
        .count())
}

fn write_pgm(path: &Path, width: u32, height: u32, data: &[u8]) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    write!(file, "P5\n{} {}\n255\n", width, height)?;
    file.write_all(data)?;
    Ok(())
}

fn generate_patterns(dir: &Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut patterns = Vec::new();

    // 1. Uniform patterns (all same value)
    for &value in &[0u8, 128, 255] {
        for &size in &[8u32, 16, 64] {
            let data = vec![value; (size * size) as usize];
            let path = dir.join(format!("uniform_{}_{size}x{size}.pgm", value));
            write_pgm(&path, size, size, &data)?;
            patterns.push(path);
        }
    }

    // 2. Horizontal gradient
    for &size in &[8u32, 16, 64] {
        let mut data = Vec::with_capacity((size * size) as usize);
        for _y in 0..size {
            for x in 0..size {
                data.push(((x * 255) / (size - 1).max(1)) as u8);
            }
        }
        let path = dir.join(format!("h_gradient_{size}x{size}.pgm"));
        write_pgm(&path, size, size, &data)?;
        patterns.push(path);
    }

    // 3. Vertical gradient
    for &size in &[8u32, 16, 64] {
        let mut data = Vec::with_capacity((size * size) as usize);
        for y in 0..size {
            for _x in 0..size {
                data.push(((y * 255) / (size - 1).max(1)) as u8);
            }
        }
        let path = dir.join(format!("v_gradient_{size}x{size}.pgm"));
        write_pgm(&path, size, size, &data)?;
        patterns.push(path);
    }

    // 4. Diagonal gradient
    for &size in &[8u32, 16, 64] {
        let mut data = Vec::with_capacity((size * size) as usize);
        for y in 0..size {
            for x in 0..size {
                data.push((((x + y) * 255) / ((size - 1) * 2).max(1)) as u8);
            }
        }
        let path = dir.join(format!("d_gradient_{size}x{size}.pgm"));
        write_pgm(&path, size, size, &data)?;
        patterns.push(path);
    }

    // 5. Checkerboard patterns (different block sizes)
    for &block_size in &[1u32, 2, 4, 8] {
        for &size in &[16u32, 64] {
            if block_size > size / 2 {
                continue;
            }
            let mut data = Vec::with_capacity((size * size) as usize);
            for y in 0..size {
                for x in 0..size {
                    let block_x = x / block_size;
                    let block_y = y / block_size;
                    data.push(if (block_x + block_y) % 2 == 0 { 255 } else { 0 });
                }
            }
            let path = dir.join(format!("checker_{block_size}_{size}x{size}.pgm"));
            write_pgm(&path, size, size, &data)?;
            patterns.push(path);
        }
    }

    // 6. Random noise (deterministic seed using simple LCG)
    for &size in &[16u32, 64] {
        let mut rng = SimpleRng::new(42);
        let data: Vec<u8> = (0..size * size).map(|_| rng.next_u8()).collect();
        let path = dir.join(format!("noise_{size}x{size}.pgm"));
        write_pgm(&path, size, size, &data)?;
        patterns.push(path);
    }

    // 7. Smooth random (low frequency noise with bilinear interpolation)
    for &size in &[16u32, 64] {
        let small_size = (size / 4).max(4);
        let mut rng = SimpleRng::new(123);
        let small: Vec<Vec<u8>> = (0..small_size)
            .map(|_| (0..small_size).map(|_| rng.next_u8()).collect())
            .collect();

        let mut data = Vec::with_capacity((size * size) as usize);
        for y in 0..size {
            for x in 0..size {
                let fx = x as f32 * (small_size - 1) as f32 / (size - 1).max(1) as f32;
                let fy = y as f32 * (small_size - 1) as f32 / (size - 1).max(1) as f32;
                let x0 = fx as usize;
                let y0 = fy as usize;
                let x1 = (x0 + 1).min(small_size as usize - 1);
                let y1 = (y0 + 1).min(small_size as usize - 1);
                let dx = fx - x0 as f32;
                let dy = fy - y0 as f32;

                let val = small[y0][x0] as f32 * (1.0 - dx) * (1.0 - dy)
                    + small[y0][x1] as f32 * dx * (1.0 - dy)
                    + small[y1][x0] as f32 * (1.0 - dx) * dy
                    + small[y1][x1] as f32 * dx * dy;
                data.push(val as u8);
            }
        }
        let path = dir.join(format!("smooth_{size}x{size}.pgm"));
        write_pgm(&path, size, size, &data)?;
        patterns.push(path);
    }

    // 8. Vertical stripes
    for &size in &[16u32, 64] {
        let stripe_width = size / 8;
        let mut data = Vec::with_capacity((size * size) as usize);
        for _y in 0..size {
            for x in 0..size {
                let stripe = x / stripe_width;
                data.push(if stripe % 2 == 0 { 255 } else { 0 });
            }
        }
        let path = dir.join(format!("stripes_{size}x{size}.pgm"));
        write_pgm(&path, size, size, &data)?;
        patterns.push(path);
    }

    // 9. Circle/disk pattern
    for &size in &[16u32, 64] {
        let center = size as f32 / 2.0;
        let radius = size as f32 / 3.0;
        let mut data = Vec::with_capacity((size * size) as usize);
        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - center + 0.5;
                let dy = y as f32 - center + 0.5;
                let dist = (dx * dx + dy * dy).sqrt();
                data.push(if dist < radius { 255 } else { 0 });
            }
        }
        let path = dir.join(format!("circle_{size}x{size}.pgm"));
        write_pgm(&path, size, size, &data)?;
        patterns.push(path);
    }

    // 10. Radial gradient
    for &size in &[16u32, 64] {
        let center = size as f32 / 2.0;
        let max_dist = (2.0 * center * center).sqrt();
        let mut data = Vec::with_capacity((size * size) as usize);
        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - center + 0.5;
                let dy = y as f32 - center + 0.5;
                let dist = (dx * dx + dy * dy).sqrt();
                let val = 255.0 * (1.0 - (dist / max_dist).min(1.0));
                data.push(val as u8);
            }
        }
        let path = dir.join(format!("radial_{size}x{size}.pgm"));
        write_pgm(&path, size, size, &data)?;
        patterns.push(path);
    }

    Ok(patterns)
}

/// Simple deterministic RNG (Linear Congruential Generator)
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        // LCG parameters from Numerical Recipes
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    fn next_u8(&mut self) -> u8 {
        (self.next_u64() >> 56) as u8
    }
}
