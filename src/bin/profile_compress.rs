use llic::{LlicContext, Mode, Quality};

fn main() {
    let width = 512u32;
    let height = 512u32;
    let pixel_count = (width * height) as usize;

    // Generate gradient image
    let mut image = vec![0u8; pixel_count];
    for y in 0..height {
        for x in 0..width {
            image[(y * width + x) as usize] = ((x + y) % 256) as u8;
        }
    }

    let ctx = LlicContext::new(width, height, width, Some(1)).unwrap();
    let mut output = vec![0u8; ctx.compressed_buffer_size()];

    // Run compression many times for profiling
    let iterations = 10000;
    let start = std::time::Instant::now();

    for _ in 0..iterations {
        std::hint::black_box(
            ctx.compress_gray8(
                std::hint::black_box(&image),
                Quality::Lossless,
                Mode::Default,
                std::hint::black_box(&mut output),
            )
            .unwrap(),
        );
    }

    let elapsed = start.elapsed();
    let per_iter = elapsed / iterations;
    let throughput = (pixel_count as f64 / per_iter.as_secs_f64()) / 1_000_000.0;

    println!(
        "Elapsed: {:?}, per iteration: {:?}, throughput: {:.1} MPixels/sec",
        elapsed, per_iter, throughput
    );
}
