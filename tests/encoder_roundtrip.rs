//! Roundtrip tests for the lossless encoder.
//!
//! These tests verify that: compress(image) -> decompress -> original image
//! using the verified decoder implementation.

use llic::entropy_coder::{compress, decompress};

/// Simple deterministic RNG for reproducible test patterns
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    fn next_u8(&mut self) -> u8 {
        (self.next_u64() >> 56) as u8
    }
}

/// Generate test patterns for roundtrip testing
mod patterns {
    use super::SimpleRng;

    /// All zeros
    pub fn zeros(width: usize, height: usize) -> Vec<u8> {
        vec![0u8; width * height]
    }

    /// All same value
    pub fn uniform(width: usize, height: usize, value: u8) -> Vec<u8> {
        vec![value; width * height]
    }

    /// Horizontal gradient (0 to 255 across width)
    pub fn h_gradient(width: usize, height: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(width * height);
        for _y in 0..height {
            for x in 0..width {
                data.push(((x * 255) / (width - 1).max(1)) as u8);
            }
        }
        data
    }

    /// Vertical gradient (0 to 255 down height)
    pub fn v_gradient(width: usize, height: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(width * height);
        for y in 0..height {
            for _x in 0..width {
                data.push(((y * 255) / (height - 1).max(1)) as u8);
            }
        }
        data
    }

    /// Diagonal gradient
    pub fn d_gradient(width: usize, height: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                data.push((((x + y) * 255) / ((width - 1) + (height - 1)).max(1)) as u8);
            }
        }
        data
    }

    /// Checkerboard pattern
    pub fn checkerboard(width: usize, height: usize, block_size: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let block_x = x / block_size;
                let block_y = y / block_size;
                data.push(if (block_x + block_y).is_multiple_of(2) {
                    255
                } else {
                    0
                });
            }
        }
        data
    }

    /// Random noise (deterministic)
    pub fn noise(width: usize, height: usize, seed: u64) -> Vec<u8> {
        let mut rng = SimpleRng::new(seed);
        (0..width * height).map(|_| rng.next_u8()).collect()
    }

    /// Smooth random (low frequency noise)
    pub fn smooth(width: usize, height: usize, seed: u64) -> Vec<u8> {
        let small_size = (width / 4).max(4);
        let mut rng = SimpleRng::new(seed);
        let small: Vec<Vec<u8>> = (0..small_size)
            .map(|_| (0..small_size).map(|_| rng.next_u8()).collect())
            .collect();

        let mut data = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let fx = x as f32 * (small_size - 1) as f32 / (width - 1).max(1) as f32;
                let fy = y as f32 * (small_size - 1) as f32 / (height - 1).max(1) as f32;
                let x0 = fx as usize;
                let y0 = fy as usize;
                let x1 = (x0 + 1).min(small_size - 1);
                let y1 = (y0 + 1).min(small_size - 1);
                let dx = fx - x0 as f32;
                let dy = fy - y0 as f32;

                let val = small[y0][x0] as f32 * (1.0 - dx) * (1.0 - dy)
                    + small[y0][x1] as f32 * dx * (1.0 - dy)
                    + small[y1][x0] as f32 * (1.0 - dx) * dy
                    + small[y1][x1] as f32 * dx * dy;
                data.push(val as u8);
            }
        }
        data
    }

    /// Vertical stripes
    pub fn stripes(width: usize, height: usize) -> Vec<u8> {
        let stripe_width = (width / 8).max(1);
        let mut data = Vec::with_capacity(width * height);
        for _y in 0..height {
            for x in 0..width {
                let stripe = x / stripe_width;
                data.push(if stripe.is_multiple_of(2) { 255 } else { 0 });
            }
        }
        data
    }

    /// Circle/disk pattern
    pub fn circle(width: usize, height: usize) -> Vec<u8> {
        let center_x = width as f32 / 2.0;
        let center_y = height as f32 / 2.0;
        let radius = width.min(height) as f32 / 3.0;
        let mut data = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let dx = x as f32 - center_x + 0.5;
                let dy = y as f32 - center_y + 0.5;
                let dist = (dx * dx + dy * dy).sqrt();
                data.push(if dist < radius { 255 } else { 0 });
            }
        }
        data
    }

    /// Radial gradient
    pub fn radial(width: usize, height: usize) -> Vec<u8> {
        let center_x = width as f32 / 2.0;
        let center_y = height as f32 / 2.0;
        let max_dist = (center_x * center_x + center_y * center_y).sqrt();
        let mut data = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let dx = x as f32 - center_x + 0.5;
                let dy = y as f32 - center_y + 0.5;
                let dist = (dx * dx + dy * dy).sqrt();
                let val = 255.0 * (1.0 - (dist / max_dist).min(1.0));
                data.push(val as u8);
            }
        }
        data
    }

    /// Simple sequential values (0,1,2,3,...)
    pub fn sequential(width: usize, height: usize) -> Vec<u8> {
        (0..width * height).map(|i| (i % 256) as u8).collect()
    }
}

/// Helper to run roundtrip test
fn roundtrip_test(input: &[u8], width: u32, height: u32, name: &str) {
    let compressed = compress(input, width, height, width)
        .unwrap_or_else(|e| panic!("Compression failed for {}: {:?}", name, e));

    let mut output = vec![0u8; input.len()];
    decompress(&compressed, width, height, width, &mut output)
        .unwrap_or_else(|e| panic!("Decompression failed for {}: {:?}", name, e));

    if input != output {
        // Find first difference for debugging
        for (i, (a, b)) in input.iter().zip(output.iter()).enumerate() {
            if a != b {
                let x = i % width as usize;
                let y = i / width as usize;
                panic!(
                    "Roundtrip failed for {} at pixel ({}, {}): expected {}, got {}",
                    name, x, y, a, b
                );
            }
        }
        panic!("Roundtrip failed for {} (length mismatch)", name);
    }
}

// === Basic roundtrip tests ===

#[test]
fn test_roundtrip_zeros_4x4() {
    let input = patterns::zeros(4, 4);
    roundtrip_test(&input, 4, 4, "zeros_4x4");
}

#[test]
fn test_roundtrip_zeros_8x8() {
    let input = patterns::zeros(8, 8);
    roundtrip_test(&input, 8, 8, "zeros_8x8");
}

#[test]
fn test_roundtrip_zeros_64x64() {
    let input = patterns::zeros(64, 64);
    roundtrip_test(&input, 64, 64, "zeros_64x64");
}

#[test]
fn test_roundtrip_uniform_128_4x4() {
    let input = patterns::uniform(4, 4, 128);
    roundtrip_test(&input, 4, 4, "uniform_128_4x4");
}

#[test]
fn test_roundtrip_uniform_255_4x4() {
    let input = patterns::uniform(4, 4, 255);
    roundtrip_test(&input, 4, 4, "uniform_255_4x4");
}

#[test]
fn test_roundtrip_uniform_128_64x64() {
    let input = patterns::uniform(64, 64, 128);
    roundtrip_test(&input, 64, 64, "uniform_128_64x64");
}

// === Gradient tests ===

#[test]
fn test_roundtrip_h_gradient_8x8() {
    let input = patterns::h_gradient(8, 8);
    roundtrip_test(&input, 8, 8, "h_gradient_8x8");
}

#[test]
fn test_roundtrip_h_gradient_64x64() {
    let input = patterns::h_gradient(64, 64);
    roundtrip_test(&input, 64, 64, "h_gradient_64x64");
}

#[test]
fn test_roundtrip_v_gradient_8x8() {
    let input = patterns::v_gradient(8, 8);
    roundtrip_test(&input, 8, 8, "v_gradient_8x8");
}

#[test]
fn test_roundtrip_v_gradient_64x64() {
    let input = patterns::v_gradient(64, 64);
    roundtrip_test(&input, 64, 64, "v_gradient_64x64");
}

#[test]
fn test_roundtrip_d_gradient_8x8() {
    let input = patterns::d_gradient(8, 8);
    roundtrip_test(&input, 8, 8, "d_gradient_8x8");
}

#[test]
fn test_roundtrip_d_gradient_64x64() {
    let input = patterns::d_gradient(64, 64);
    roundtrip_test(&input, 64, 64, "d_gradient_64x64");
}

// === Pattern tests ===

#[test]
fn test_roundtrip_checkerboard_1_16x16() {
    let input = patterns::checkerboard(16, 16, 1);
    roundtrip_test(&input, 16, 16, "checkerboard_1_16x16");
}

#[test]
fn test_roundtrip_checkerboard_4_64x64() {
    let input = patterns::checkerboard(64, 64, 4);
    roundtrip_test(&input, 64, 64, "checkerboard_4_64x64");
}

#[test]
fn test_roundtrip_stripes_16x16() {
    let input = patterns::stripes(16, 16);
    roundtrip_test(&input, 16, 16, "stripes_16x16");
}

#[test]
fn test_roundtrip_stripes_64x64() {
    let input = patterns::stripes(64, 64);
    roundtrip_test(&input, 64, 64, "stripes_64x64");
}

#[test]
fn test_roundtrip_circle_16x16() {
    let input = patterns::circle(16, 16);
    roundtrip_test(&input, 16, 16, "circle_16x16");
}

#[test]
fn test_roundtrip_circle_64x64() {
    let input = patterns::circle(64, 64);
    roundtrip_test(&input, 64, 64, "circle_64x64");
}

#[test]
fn test_roundtrip_radial_16x16() {
    let input = patterns::radial(16, 16);
    roundtrip_test(&input, 16, 16, "radial_16x16");
}

#[test]
fn test_roundtrip_radial_64x64() {
    let input = patterns::radial(64, 64);
    roundtrip_test(&input, 64, 64, "radial_64x64");
}

// === Noise tests ===

#[test]
fn test_roundtrip_noise_16x16() {
    let input = patterns::noise(16, 16, 42);
    roundtrip_test(&input, 16, 16, "noise_16x16");
}

#[test]
fn test_roundtrip_noise_64x64() {
    let input = patterns::noise(64, 64, 42);
    roundtrip_test(&input, 64, 64, "noise_64x64");
}

#[test]
fn test_roundtrip_smooth_16x16() {
    let input = patterns::smooth(16, 16, 123);
    roundtrip_test(&input, 16, 16, "smooth_16x16");
}

#[test]
fn test_roundtrip_smooth_64x64() {
    let input = patterns::smooth(64, 64, 123);
    roundtrip_test(&input, 64, 64, "smooth_64x64");
}

// === Edge cases ===

#[test]
fn test_roundtrip_sequential_4x4() {
    let input = patterns::sequential(4, 4);
    roundtrip_test(&input, 4, 4, "sequential_4x4");
}

#[test]
fn test_roundtrip_sequential_16x16() {
    let input = patterns::sequential(16, 16);
    roundtrip_test(&input, 16, 16, "sequential_16x16");
}

#[test]
fn test_roundtrip_single_row() {
    // 1 row, multiple columns
    let input = patterns::h_gradient(64, 4);
    roundtrip_test(&input, 64, 4, "single_row_64x4");
}

#[test]
fn test_roundtrip_single_column() {
    // Multiple rows, narrow width
    let input = patterns::v_gradient(4, 64);
    roundtrip_test(&input, 4, 64, "single_column_4x64");
}

#[test]
fn test_roundtrip_large_256x256() {
    let input = patterns::noise(256, 256, 999);
    roundtrip_test(&input, 256, 256, "large_256x256");
}

// === Specific value tests ===

#[test]
fn test_roundtrip_all_values() {
    // Test that all 256 possible byte values survive roundtrip
    let mut input = Vec::with_capacity(256 * 4);
    for _ in 0..4 {
        for i in 0..=255u8 {
            input.push(i);
        }
    }
    roundtrip_test(&input, 256, 4, "all_values");
}

#[test]
fn test_roundtrip_alternating_extremes() {
    // Alternating 0 and 255
    let input: Vec<u8> = (0..64 * 64)
        .map(|i| if i % 2 == 0 { 0 } else { 255 })
        .collect();
    roundtrip_test(&input, 64, 64, "alternating_extremes");
}
