//! Codec throughput on grayscale PGM images, without the C++ reference, so it
//! also runs as WebAssembly (the web demo's engine). Lossless round trips are
//! checked.
//!
//! Native: `cargo run --release --example wasm_bench -- image.pgm...`
//!
//! WebAssembly in Node (V8), with SIMD128:
//! ```bash
//! RUSTFLAGS="-C target-feature=+simd128" cargo build --release \
//!     --example wasm_bench --target wasm32-wasip1
//! node scripts/wasm_bench.mjs target/wasm32-wasip1/release/examples/wasm_bench.wasm image.pgm...
//! ```
//! Images are read relative to the directory the runner preopens.

use std::time::{Duration, Instant};

use llic::pgm::Pgm;
use llic::{LlicContext, Mode, Quality};

/// Median time per call of `f` over batches of about `budget / 9` each.
fn measure(budget: Duration, mut f: impl FnMut()) -> Duration {
    f();
    let start = Instant::now();
    f();
    let once = start.elapsed().max(Duration::from_micros(1));
    let iters = ((budget / 9).as_nanos() / once.as_nanos()).max(1) as u32;
    let mut times: Vec<Duration> = (0..9)
        .map(|_| {
            let start = Instant::now();
            for _ in 0..iters {
                f();
            }
            start.elapsed() / iters
        })
        .collect();
    times.sort();
    times[times.len() / 2]
}

fn configs() -> Vec<(Quality, Mode)> {
    let mut configs = Vec::new();
    for quality in [Quality::Lossless, Quality::High, Quality::VeryLow] {
        for mode in [Mode::Default, Mode::Fast, Mode::Dynamic] {
            configs.push((quality, mode));
        }
    }
    configs.push((Quality::LosslessEntropy, Mode::Default));
    configs.push((Quality::LosslessEntropy, Mode::Dynamic));
    configs
}

fn main() {
    let budget = Duration::from_millis(400);
    for path in std::env::args().skip(1) {
        let image = Pgm::open(&path).expect("read PGM");
        let (width, height) = (image.width(), image.height());
        let pixels = image.data();
        let ctx = LlicContext::new(width, height, width, Some(1)).expect("context");
        let mut stream = vec![0u8; ctx.compressed_buffer_size()];
        let mut decoded = vec![0u8; pixels.len()];
        let mpix = |time: Duration| f64::from(width * height) / time.as_secs_f64() / 1e6;
        println!("{path} ({width}x{height})  MP/s");
        println!(
            "{:<24} {:>6} {:>9} {:>9}",
            "quality/mode", "bpp", "encode", "decode"
        );
        let (mut log_enc, mut log_dec, mut count) = (0.0, 0.0, 0.0);
        for (quality, mode) in configs() {
            let size = ctx
                .compress_gray8(pixels, quality, mode, &mut stream)
                .expect("compress");
            ctx.decompress_gray8(&stream[..size], &mut decoded)
                .expect("decompress");
            if matches!(quality, Quality::Lossless | Quality::LosslessEntropy) {
                assert_eq!(decoded, pixels, "{quality:?}/{mode:?} round trip");
            }
            let encode = measure(budget, || {
                ctx.compress_gray8(pixels, quality, mode, &mut stream)
                    .expect("compress");
            });
            let decode = measure(budget, || {
                ctx.decompress_gray8(&stream[..size], &mut decoded)
                    .expect("decompress");
            });
            let (enc, dec) = (mpix(encode), mpix(decode));
            (log_enc, log_dec, count) = (log_enc + enc.ln(), log_dec + dec.ln(), count + 1.0);
            println!(
                "{:<24} {:>6.3} {enc:>9.1} {dec:>9.1}",
                format!("{quality:?}/{mode:?}"),
                size as f64 * 8.0 / f64::from(width * height),
            );
        }
        println!(
            "{:<24} {:>6} {:>9.1} {:>9.1}\n",
            "geometric mean",
            "",
            (log_enc / count).exp(),
            (log_dec / count).exp()
        );
    }
}
