# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Format version 5, written by the C++ reference since 3.5.0. Tile chunks with
  an entropy-coded min/dist header now store a 32-bit header offset and put the
  header after the pixel stream (`lossy::TileHeaderLayout`); every other stream
  is unchanged apart from the version byte. The decoder accepts versions 3 to 5,
  so streams from current C++ releases decode, and `FormatVersion::V5` encodes
  them.
- In v5, lossless default- and dynamic-mode tile streams choose each block's
  header values nearest the block above, as the C++ encoder does, which makes
  them 2-3% smaller. Decoders need no change for this.
- Compatibility tests against the C++ reference's decoder vectors
  (`test_data/vectors`, from every C++ release since 2.0.0): all format 3-5
  streams decode to the expected images, and our single-chunk v4 and v5 output
  matches the C++ encoder byte for byte.

### Changed

- **Breaking:** `compress_gray8` writes format v5, like the C++ reference.
  Decoders older than llic_rs 0.5.0 or C++ 3.5.0 cannot read it; use
  `compress_gray8_versioned` with `FormatVersion::V4` for them.
- **Breaking:** `lossy::compress_tile_block`, `decompress_tile_block`,
  `decompress_tile_block_dynamic` and `extract_tile_metadata` take a
  `TileHeaderLayout`.

### Fixed

- Lossy tile encoding in builds without SSE4.1 (the default x86-64 target, and
  WebAssembly) quantized some pixels one bucket too low: the reciprocal table
  rounded `65536 / bucket_size` down, so residuals that are exact multiples of
  the bucket size lost one. Images stayed within the error limit, but streams
  differed from the C++ encoder and the SSE4.1/AVX2 builds. The table now
  rounds up, which is exact for every 8-bit residual, and lossy streams match
  the C++ reference byte for byte in every build.

## [0.4.1] - 2026-10-28

### Fixed

- The BMI2/SSE4.1 pixel-packing paths in tile-based (lossy) compression stored
  u16/u32 values through unaligned pointers into the output stream. This was
  undefined behaviour, and debug builds with those target features enabled
  aborted with "misaligned pointer dereference". The stores now use
  `write_unaligned`, which compiles to the same instructions on x86; output and
  performance are unchanged.

## [0.4.0] - 2026-06-25

### Added

- `LlicContext::compress_gray8_versioned`, which compresses with an explicit
  container format version (`FormatVersion::V3` or `FormatVersion::V4`).
  `compress_gray8` is now a thin wrapper that always emits the current version
  (v4), so existing callers are unaffected. Version 3 output supports
  entropy-coded lossless and tile-based blocks in both `Mode::Default` and
  `Mode::Fast`; for tile-based streams the mode is carried in bit 7 of the first
  payload byte, matching the C++ reference's pre-v4 header layout. `Mode::Dynamic`
  has no representation in v3 and returns `UnsupportedFormat`. The v3 header
  distinguishes entropy-coded from tile-based blocks by "byte 2 == 0"; when a
  tile-based payload length is a multiple of 256 (which would zero that byte), a
  single defensive pad byte is appended so the discriminator stays unambiguous.
  Decoders never read the pad byte.

### Changed

- Prepared the crate for a crates.io release. The published package no longer
  ships test fixtures, the demo, scripts, or docs that consumers don't need
  (`tables/*.bin` lookup data is still included, as it is compiled in via
  `include_bytes!`). Added `authors`, `rust-version`, and an `exclude` list to
  the manifest.

### Removed

- Removed the `cpp-reference` cargo feature and the in-crate `ffi` module. The
  C++ reference FFI, its `cc`/`build.rs` compilation, and the Rust-vs-C++
  comparison benchmarks now live in a separate, unpublished workspace member
  (`cpp-bench/`, crate `llic-cpp-bench`). The published crate is pure Rust with
  no features that depend on the C++ submodule. Rust-only benchmarks remain in
  the main crate (`cargo bench -p llic_rs`).

## [0.3.1] - 2026-04-11

### Fixed

- Skip midpoint brightening for lossless tile decompression, matching C++ v3.3.0
  fix. Previously lossless tiles could have slightly shifted pixel values.
- CI clippy job no longer fails due to C++ submodule compilation or benchmark
  targets requiring the cpp-reference feature.

### Changed

- Updated C++ reference submodule from v3.2.2 to v3.3.0.
- Optimized lossless entropy encoder with 8x unrolled encode loop, pipelined
  row processing, and pointer-based BitWriter for higher throughput.
- Added SSE4.1 SIMD tile processing for lossy compression using float
  reciprocal division matching the C++ reference.
- Added AVX2 dual-tile processing to compress two adjacent 4x4 blocks
  simultaneously.
- Added SSSE3-accelerated first-row delta computation and nibble packing.
- Added BMI2 PEXT/PDEP-based pixel pack/unpack for 2-bit and 4-bit tiles
  (fast on Zen 3+).
- Optimized compression table lookups with direct pointer reads.
- Switched BitWriter to pre-allocated buffer with unsafe pointer writes and
  32-bit batch flushing.

### Added

- `profile_compress` binary and mise tasks for targeted compression profiling.

## [0.3.0] - 2026-01-15

### Added

- Dynamic compression mode with per-row adaptive predictor selection.
- Three compression modes: Default, Fast, and Dynamic.
- Mode comparison benchmarks.

## [0.2.0] - 2025-12-20

### Added

- Lossy decompression support for tile-based compression (Q1-Q4).
- Lossy compression with quality levels and tile-based 4x4 block encoding.
- WebAssembly support with SIMD optimization.
- Interactive demo page with comparison view and tile visualization.
- LLSC file container format with backward-compatible parser.
- Format v4 support for tile-based lossless.
- CI workflow and pre-commit hooks.

## [0.1.0] - 2025-12-10

### Added

- Initial Rust port of the LLIC codec.
- Lossless compression and decompression with entropy coding.
- Optimized entropy decoder with u64 bit container and loop unrolling.
- Criterion benchmarks comparing Rust vs C++ reference performance.
- PGM image format I/O.
- CLI tool for compress/decompress operations.
