# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.1] - 2026-04-11

### Changed

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
