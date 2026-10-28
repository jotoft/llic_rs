//! Rust vs C++ speed and stream equality on the test photos, single-threaded.
//!
//! For each photo (converted to luma like the C++ tools) and every quality and
//! mode, compresses and decompresses with both codecs and prints throughput in
//! megapixels per second, the bits per pixel, and whether the streams match.
//!
//! ```bash
//! RUSTFLAGS="-C target-cpu=native" taskset -c 2 \
//!     cargo run --release -p llic-cpp-bench --example photo_matrix -- [size] [budget_ms]
//! ```
//!
//! `size` picks the photos by width (default 1920); `budget_ms` is the time
//! spent per measurement (default 300).

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use llic::{LlicContext, Mode, Quality};
use llic_cpp_bench as cpp;

struct Photo {
    name: String,
    width: u32,
    height: u32,
    luma: Vec<u8>,
}

/// BT.601 luma, as the C++ image_io `toLuma`.
fn to_luma(rgb: &[u8]) -> Vec<u8> {
    rgb.chunks_exact(3)
        .map(|p| {
            let y = 77 * u32::from(p[0]) + 150 * u32::from(p[1]) + 29 * u32::from(p[2]) + 128;
            #[allow(clippy::cast_possible_truncation)] // at most 255
            let y = (y >> 8) as u8;
            y
        })
        .collect()
}

fn load_photos(width: u32) -> Vec<Photo> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../test_data/photos");
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("test_data/photos")
        .map(|entry| entry.expect("dir entry").path())
        .filter(|path| {
            path.extension().is_some_and(|ext| ext == "webp")
                && path
                    .file_stem()
                    .is_some_and(|stem| stem.to_string_lossy().contains(&format!("_{width}x")))
        })
        .collect();
    paths.sort();
    paths
        .iter()
        .map(|path| {
            let rgb = image::open(path).expect("decode photo").into_rgb8();
            Photo {
                name: path.file_stem().unwrap().to_string_lossy().into_owned(),
                width: rgb.width(),
                height: rgb.height(),
                luma: to_luma(rgb.as_raw()),
            }
        })
        .collect()
}

/// Median time per call of `f`, from batches filling roughly `budget`.
fn measure(budget: Duration, mut f: impl FnMut()) -> Duration {
    f(); // warm up caches and lazily built tables
    let start = Instant::now();
    f();
    let once = start.elapsed().max(Duration::from_micros(1));
    let samples = 15u32;
    let per_sample = (budget / samples).max(once);
    let iters = (per_sample.as_nanos() / once.as_nanos()).max(1) as u32;
    let mut times: Vec<Duration> = (0..samples)
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

fn cpp_quality(quality: Quality) -> cpp::LlicQuality {
    match quality {
        Quality::Lossless => cpp::LlicQuality::Lossless,
        Quality::VeryHigh => cpp::LlicQuality::VeryHigh,
        Quality::High => cpp::LlicQuality::High,
        Quality::Medium => cpp::LlicQuality::Medium,
        Quality::Low => cpp::LlicQuality::Low,
        Quality::VeryLow => cpp::LlicQuality::VeryLow,
        Quality::LosslessEntropy => cpp::LlicQuality::LosslessEntropy,
    }
}

fn cpp_mode(mode: Mode) -> cpp::LlicMode {
    match mode {
        Mode::Default => cpp::LlicMode::Default,
        Mode::Fast => cpp::LlicMode::Fast,
        Mode::Dynamic => cpp::LlicMode::Dynamic,
    }
}

/// Every quality and mode the C++ encoder accepts.
fn configs() -> Vec<(Quality, Mode)> {
    let tile = [
        Quality::Lossless,
        Quality::VeryHigh,
        Quality::High,
        Quality::Medium,
        Quality::Low,
        Quality::VeryLow,
    ];
    let mut configs: Vec<(Quality, Mode)> = tile
        .iter()
        .flat_map(|&q| [Mode::Default, Mode::Fast, Mode::Dynamic].map(|m| (q, m)))
        .collect();
    configs.push((Quality::LosslessEntropy, Mode::Default));
    configs.push((Quality::LosslessEntropy, Mode::Dynamic));
    configs
}

/// C++ context with preallocated buffers; calls the C API directly so that
/// allocation stays out of the timings.
struct CppCodec {
    ctx: *mut cpp::LlicContext,
}

impl CppCodec {
    fn new(width: u32, height: u32) -> Self {
        let mut ctx = std::ptr::null_mut();
        // SAFETY: valid out pointer; dimensions are multiples of 4.
        let status = unsafe { cpp::llic_init(&mut ctx, width, height, width, 1) };
        assert_eq!(status, cpp::LlicError::None, "llic_init");
        Self { ctx }
    }

    fn buffer_size(&self) -> usize {
        // SAFETY: ctx is a live context.
        unsafe { cpp::llic_compressedBufferSize(self.ctx) }
    }

    fn compress(&self, src: &[u8], quality: Quality, mode: Mode, dst: &mut [u8]) -> usize {
        assert!(dst.len() >= self.buffer_size());
        // SAFETY: src holds width * height pixels and dst the buffer size.
        let size = unsafe {
            cpp::llic_compressGray8(
                self.ctx,
                src.as_ptr(),
                cpp_quality(quality),
                cpp_mode(mode),
                dst.as_mut_ptr(),
            )
        };
        assert_ne!(size, 0, "C++ compress failed");
        size
    }

    fn decompress(&self, src: &[u8], dst: &mut [u8]) {
        let (mut quality, mut mode) = (cpp::LlicQuality::Lossless, cpp::LlicMode::Default);
        // SAFETY: sizes are passed with both buffers.
        let size = unsafe {
            cpp::llic_decompressGray8(
                self.ctx,
                src.as_ptr(),
                src.len(),
                dst.as_mut_ptr(),
                dst.len(),
                &mut quality,
                &mut mode,
            )
        };
        assert_eq!(size, dst.len(), "C++ decompress failed");
    }
}

impl Drop for CppCodec {
    fn drop(&mut self) {
        // SAFETY: ctx came from llic_init and is destroyed once.
        unsafe { cpp::llic_destroy(&mut self.ctx) };
    }
}

fn mpix_per_s(pixels: usize, time: Duration) -> f64 {
    pixels as f64 / time.as_secs_f64() / 1e6
}

fn bench_photo(photo: &Photo, budget: Duration) {
    let pixels = photo.luma.len();
    let rust = LlicContext::new(photo.width, photo.height, photo.width, Some(1)).expect("context");
    let cpp = CppCodec::new(photo.width, photo.height);
    let mut rust_stream = vec![0u8; rust.compressed_buffer_size().max(cpp.buffer_size())];
    let mut cpp_stream = vec![0u8; rust_stream.len()];
    let mut decoded = vec![0u8; pixels];

    println!(
        "\n{} ({}x{}, luma)  MP/s, higher is better",
        photo.name, photo.width, photo.height
    );
    println!(
        "{:<24} {:>6}  {:>8} {:>8} {:>6}  {:>8} {:>8} {:>6}  stream",
        "quality/mode", "bpp", "rust enc", "c++ enc", "ratio", "rust dec", "c++ dec", "ratio"
    );
    for (quality, mode) in configs() {
        let rust_size = rust
            .compress_gray8(&photo.luma, quality, mode, &mut rust_stream)
            .expect("Rust compress");
        let cpp_size = cpp.compress(&photo.luma, quality, mode, &mut cpp_stream);
        let same = rust_stream[..rust_size] == cpp_stream[..cpp_size];

        rust.decompress_gray8(&cpp_stream[..cpp_size], &mut decoded)
            .expect("Rust decodes C++ stream");
        let mut cpp_decoded = vec![0u8; pixels];
        cpp.decompress(&cpp_stream[..cpp_size], &mut cpp_decoded);
        let pixels_match = decoded == cpp_decoded;

        let rust_enc = measure(budget, || {
            rust.compress_gray8(&photo.luma, quality, mode, &mut rust_stream)
                .expect("Rust compress");
        });
        let cpp_enc = measure(budget, || {
            cpp.compress(&photo.luma, quality, mode, &mut cpp_stream);
        });
        let stream = &cpp_stream[..cpp_size];
        let rust_dec = measure(budget, || {
            rust.decompress_gray8(stream, &mut decoded)
                .expect("Rust decompress");
        });
        let cpp_dec = measure(budget, || cpp.decompress(stream, &mut decoded));

        let (re, ce) = (mpix_per_s(pixels, rust_enc), mpix_per_s(pixels, cpp_enc));
        let (rd, cd) = (mpix_per_s(pixels, rust_dec), mpix_per_s(pixels, cpp_dec));
        let label = format!("{quality:?}/{mode:?}");
        let status = match (same, pixels_match) {
            (true, true) => "same".to_string(),
            (false, true) => format!("DIFF ({rust_size} vs {cpp_size} B)"),
            (_, false) => "DECODE MISMATCH".to_string(),
        };
        println!(
            "{label:<24} {:>6.3}  {re:>8.1} {ce:>8.1} {:>5.2}x  {rd:>8.1} {cd:>8.1} {:>5.2}x  {status}",
            cpp_size as f64 * 8.0 / pixels as f64,
            re / ce,
            rd / cd,
        );
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let width: u32 = args.next().map_or(1920, |a| a.parse().expect("size"));
    let budget = Duration::from_millis(args.next().map_or(300, |a| a.parse().expect("budget_ms")));

    let photos = load_photos(width);
    assert!(!photos.is_empty(), "no test_data/photos/*_{width}x*.webp");
    println!(
        "C++ llic {} (SSE4.2, runtime BMI2) vs llic_rs (this checkout), single thread",
        cpp::cpp_version()
    );
    println!("ratio = Rust / C++ throughput (above 1 means Rust is faster)");
    for photo in &photos {
        bench_photo(photo, budget);
    }
}
