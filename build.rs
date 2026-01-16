use std::env;
use std::path::PathBuf;

fn main() {
    // Only build C++ if cpp-reference feature is enabled
    if env::var("CARGO_FEATURE_CPP_REFERENCE").is_err() {
        return;
    }

    // Skip C++ build for WASM targets
    let target = env::var("TARGET").unwrap_or_default();
    if target.contains("wasm") {
        println!("cargo:warning=Skipping C++ build for WASM target");
        return;
    }

    let llic_src = PathBuf::from("llic/src/llic");

    // Only build C++ library if the submodule exists
    if !llic_src.exists() {
        println!("cargo:warning=C++ llic submodule not found, skipping C++ build");
        return;
    }

    // Compile the C++ llic library
    let mut build = cc::Build::new();

    // The CMake build generates llic_export.h - include that directory
    let llic_build = PathBuf::from("llic/build/src/llic");

    build
        .cpp(true)
        .std("c++11")
        .opt_level(3)
        .define("NDEBUG", None)
        .define("LLIC_STATIC_DEFINE", None) // Use static library macros
        .include(&llic_src)
        .include(&llic_build) // For llic_export.h
        .include(llic_src.join("entropycoder"))
        .file(llic_src.join("llic.cpp"))
        .file(llic_src.join("entropycoder/u8v1_compress.cpp"))
        .file(llic_src.join("entropycoder/u8v1_compress_table.cpp"))
        .file(llic_src.join("entropycoder/u8v1_decompress.cpp"))
        .file(llic_src.join("entropycoder/u8v1_decompress_table.cpp"));

    // Enable SSE4.2 on x86_64
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if target_arch == "x86_64" {
        build.flag("-msse4.2");
    }

    // Disable multithreading for simpler benchmarking (single-threaded comparison)
    // If you want multithreading, uncomment the next line and add pthread linking
    // build.define("USE_MULTITHREADING", None);

    build.compile("llic_cpp");

    // Tell cargo to link the compiled library
    println!("cargo:rustc-link-lib=static=llic_cpp");

    // Re-run build if C++ sources change
    println!("cargo:rerun-if-changed=llic/src/llic/llic.cpp");
    println!("cargo:rerun-if-changed=llic/src/llic/llic.h");
    println!("cargo:rerun-if-changed=llic/src/llic/entropycoder/u8v1_compress.cpp");
    println!("cargo:rerun-if-changed=llic/src/llic/entropycoder/u8v1_decompress.cpp");
    println!("cargo:rerun-if-changed=build.rs");
}
