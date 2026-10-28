//! Rust vs C++ speed of the color helpers on the test photos, single-threaded.
//!
//! ```text
//! cargo build --release -p llic-cpp-bench --example helper_bench
//! taskset -c 2 target/release/examples/helper_bench [size] [budget_ms]
//! ```
//!
//! `size` picks the photos by width (default 1920); `budget_ms` is the time
//! spent on each measurement (default 300).

use std::path::Path;
use std::time::{Duration, Instant};

use llic::helpers::{self, Rows, RowsMut};
use llic_cpp_bench::helpers::ColorHelper;

const HELPERS: [ColorHelper; 4] = [
    ColorHelper::Rgb,
    ColorHelper::Rgba,
    ColorHelper::YCoCgR,
    ColorHelper::YCoCg,
];

fn rust_split(helper: ColorHelper, src: &[u8], width: u32, height: u32, planes: &mut [Vec<u8>]) {
    let src = Rows::packed(src);
    let result = match planes {
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
}

fn rust_merge(helper: ColorHelper, planes: &[Vec<u8>], width: u32, height: u32, dst: &mut [u8]) {
    let out = RowsMut::packed(dst);
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
}

/// Median time per call of `f`, from batches filling roughly `budget`.
fn measure(budget: Duration, mut f: impl FnMut()) -> Duration {
    f(); // warm up caches
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

fn main() {
    let mut args = std::env::args().skip(1);
    let size: u32 = args.next().map_or(1920, |s| s.parse().expect("size"));
    let budget = Duration::from_millis(args.next().map_or(300, |s| s.parse().expect("budget_ms")));
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../test_data/photos");

    println!(
        "{:<16} {:<8} {:>10} {:>10} {:>7}",
        "photo", "helper", "Rust ms", "C++ ms", "C++/Rust"
    );
    for name in ["rattan", "rock"] {
        let path = dir.join(format!("{name}_{size}x{}.webp", size * 2 / 3));
        let rgba = image::open(&path).expect("decode photo").into_rgba8();
        let (width, height) = rgba.dimensions();
        let plane_size = width as usize * height as usize;
        for helper in HELPERS {
            let pixels: Vec<u8> = rgba
                .as_raw()
                .iter()
                .enumerate()
                .filter(|(i, _)| i % 4 < helper.channels())
                .map(|(_, &byte)| byte)
                .collect();
            let mut planes = vec![vec![0u8; plane_size]; helper.channels()];
            let mut out = vec![0u8; pixels.len()];

            let rust = measure(budget, || {
                rust_split(helper, &pixels, width, height, &mut planes)
            });
            let cpp = measure(budget, || {
                helper
                    .split_into(&pixels, width, height, &mut planes)
                    .expect("C++ split");
            });
            report(name, helper, "split", rust, cpp);

            let rust = measure(budget, || {
                rust_merge(helper, &planes, width, height, &mut out)
            });
            let cpp = measure(budget, || {
                helper
                    .merge_into(&planes, width, height, &mut out)
                    .expect("C++ merge");
            });
            report(name, helper, "merge", rust, cpp);
        }
    }
}

fn report(photo: &str, helper: ColorHelper, op: &str, rust: Duration, cpp: Duration) {
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    println!(
        "{:<16} {:<8} {:>10.3} {:>10.3} {:>7.2}x",
        format!("{photo} {op}"),
        format!("{helper:?}"),
        ms(rust),
        ms(cpp),
        ms(cpp) / ms(rust)
    );
}
