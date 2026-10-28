use std::env;
use std::fs;
use std::path::PathBuf;

/// Sources of the C++ codec and helpers (llic 4.x), relative to `llic/src/llic`.
const SOURCES: [&str; 8] = [
    "llic.cpp",
    "llic_helpers.cpp",
    "llic_cpu.cpp",
    "llic_version.cpp",
    "entropycoder/u8v1_compress.cpp",
    "entropycoder/u8v1_compress_table.cpp",
    "entropycoder/u8v1_decompress.cpp",
    "entropycoder/u8v1_decompress_table.cpp",
];

fn main() {
    // The C++ reference lives in the `llic` git submodule at the workspace root.
    // Build scripts run with the cwd set to this crate's manifest dir (cpp-bench/),
    // so the submodule is one directory up.
    let llic_src = PathBuf::from("../llic/src/llic");

    // Only build the C++ library if the submodule has been checked out.
    // Run `git submodule update --init` to populate it.
    if !llic_src.exists() {
        println!(
            "cargo:warning=C++ llic submodule not found at ../llic; \
             run `git submodule update --init` to enable comparison benchmarks"
        );
        return;
    }

    // CMake generates llic_export.h; for a static library its macros are empty.
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    fs::write(
        out_dir.join("llic_export.h"),
        "#pragma once\n\
         #define LLIC_EXPORT\n\
         #define LLIC_NO_EXPORT\n\
         #define LLIC_DEPRECATED\n\
         #define LLIC_DEPRECATED_EXPORT\n\
         #define LLIC_DEPRECATED_NO_EXPORT\n",
    )
    .expect("write llic_export.h");

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++11")
        .opt_level(3)
        .define("NDEBUG", None)
        // Single-threaded, to compare against the single-threaded Rust codec.
        .define("LLIC_DISABLE_MULTITHREADING", None)
        .include(&out_dir)
        .include(&llic_src)
        .include(llic_src.join("entropycoder"));
    for source in SOURCES {
        build.file(llic_src.join(source));
        println!("cargo:rerun-if-changed=../llic/src/llic/{source}");
    }

    // SSE4.2 is the C++ x86 baseline (as in its CMake build); BMI2 is selected at
    // run time.
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if target_arch == "x86_64" {
        build.flag("-msse4.2");
    }

    build.compile("llic_cpp");

    // Tell cargo to link the compiled library
    println!("cargo:rustc-link-lib=static=llic_cpp");

    println!("cargo:rerun-if-changed=../llic/src/llic/llic.h");
    println!("cargo:rerun-if-changed=../llic/src/llic/llic_helpers.h");
    println!("cargo:rerun-if-changed=build.rs");
}
