# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `helpers` module, a port of the C++ reference's color helpers
  (`llic_helpers.h`, API 4.0): `deinterleave_rgb` / `interleave_rgb`,
  `deinterleave_rgba` / `interleave_rgba`, reversible YCoCg-R
  (`deinterleave_rgb_to_ycocg_r` / `interleave_ycocg_r_to_rgb`) for lossless
  coding, YCoCg with a saturating inverse (`deinterleave_rgb_to_ycocg` /
  `interleave_ycocg_to_rgb`) for lossy coding, and `plane_sizes`. Outputs equal
  the C++ helpers byte for byte, checked on every 24-bit color. SSSE3 kernels
  are selected at run time and SIMD128 kernels serve WebAssembly; single-threaded,
  they split RGB 1.15-1.9x faster than C++ 4.1.1 and merge at the same speed.
- `helpers::BufferError`, naming the buffer that is too small or has too
  short a stride, and `helpers::DimensionsError`, for sizes that do not fit a
  layout. `LlicError` is unchanged.

### Changed

- Tile-based decoding unpacks eight blocks at once with AVX2 on x86-64 CPUs
  that have it (detected at run time, so default builds get it): about 1.13x
  for lossless and 1.07-1.28x for lossy fast-mode streams, and up to 1.12x in
  the other modes. Single-threaded on the test photos, lossless fast-mode
  decoding is now 1.6x the speed of C++ 4.1.1. Output is unchanged.
- Entropy decoding is about 7% faster on x86-64: the symbol decoder's shift is
  pinned to a bare `SHL`, keeping an AND off the dependency chain between table
  lookups. Lossless entropy streams decode at 0.97-0.98x of C++ 4.1.1 (was
  0.91x), and lossy streams with compressed headers at 0.97x or better.

## [0.5.1] - 2026-10-28

Performance release: the C++ reference's 3.4-4.1 x86 work is ported, with
SIMD128 versions for WebAssembly. Single-threaded on the test photos, encoding
went from about 0.23x to 0.96-0.99x of C++ 4.1.1 and decoding to about 1.0x;
WebAssembly builds are 3-4x faster. Streams are unchanged and byte-identical to
C++. The x86 kernels are selected at run time, so default builds get them.

### Added

- `entropy_coder::compress_into`, `compress_dynamic_into` and
  `max_compressed_size`, and `lossy::compress_tile_block_into` and
  `lossy::tile_block_scratch_size`, to encode into a caller's buffer.

### Changed

- Tile-based (lossy and lossless) coding is about 2.7x faster on x86-64 with
  SSSE3 and SSE4.1, detected at run time, so default builds get it too. A port
  of the C++ reference's SSE tile codec (3.4+): the encoder reduces sixteen
  blocks at once and packs every bit width without branching, and the decoder
  unpacks four blocks at once. Fast-mode decoding now matches or beats C++
  4.1.1; streams are unchanged. The AVX2 two-tile encoder path is gone, as it
  was no faster.
- Tile-based encoding writes straight into the caller's buffer when it has room
  for the worst case (as `compressed_buffer_size` does) instead of zeroing a
  scratch buffer and copying the stream twice: about 15% faster.
- Entropy decoding reconstructs sixteen rows at once as an SSE wavefront on
  x86-64 (SSSE3/SSE4.1, detected at run time), as the C++ reference does: about
  1.5x for lossless entropy streams (2.5x in dynamic mode) and 1.2-1.6x for tile
  streams with compressed headers. Output is unchanged.
- Entropy encoding is rebuilt after the C++ encoder: on x86-64 a build with
  BMI2 shifts and SSE residuals is selected at run time; rows above 5.5 bits per
  symbol pair flush words without a data-dependent branch; dynamic mode computes
  both predictors' residuals and code lengths in one SSE pass; and the output is
  no longer a zero-filled buffer of 2 bytes per pixel. Lossless entropy and
  default/dynamic tile encoding take 7-45% fewer cycles, dynamic mode the most.
  Streams are unchanged.
- WebAssembly builds with `simd128` (the web demo) use a SIMD128 port of the
  grouped tile codec: in Node/V8, tile-based encoding is 3.2-3.7x and decoding
  2.3-2.4x faster, fast-mode decoding 4-7.6 GP/s. In the repository,
  `examples/wasm_bench` and `scripts/wasm_bench.mjs` benchmark and test the
  WebAssembly build in Node.
- WebAssembly builds with `simd128` also reconstruct entropy-coded rows with
  the sixteen-row wavefront: in Node/V8, decoding 1.1-1.4x faster overall and
  1.7-2x in dynamic mode.
- The scalar MED predictor in the decoder is branch-free, like the encoder's.
- Entropy symbol decoding is a port of the C++ 4.1 decoder: a 13-bit lookup
  table (built at compile time, identical to the C++ one) decodes up to three
  symbols per lookup, three lookups per 64-bit refill, with a checked tail for
  the stream end. Lossless entropy decoding is about 1.6x faster (0.57x ->
  0.92x of C++ 4.1.1) and tile streams with compressed headers decode faster
  than C++; WebAssembly gains about 1.3x. A truncated entropy stream is now
  `InvalidData` instead of decoding zeros past its end.
- The tile codec's bucket tables for the six quality levels (and their packed
  SIMD forms) are built at compile time instead of on every call: 5-8% faster
  coding of small images (256x256), up to 20% for fast low-quality encoding.
- Lossy tile encoding in builds with AVX2 enabled (such as
  `-C target-cpu=native`) was 13-23% slower than in default builds: LLVM turned
  the quantization's `PMULHUW` into 256-bit widening 32-bit multiplies. It is
  now pinned with inline assembly; native builds encode at 0.99x of C++ 4.1.1.
- Entropy-coded streams are encoded straight into the caller's buffer when it
  has room for the worst case (as `compressed_buffer_size` does), like tile
  streams, instead of into a vector that is then copied;
  `compressed_buffer_size` grows by 4 bytes. Lossless entropy encoding about 2%
  faster.
- Near the end of a tile stream, groups of blocks without payload kept the
  SIMD tile decoder's 16-byte reads from going past the stream end by falling
  back to per-block code for the rest of the image; they read nothing, so they
  now stay on the SIMD path (flat blocks in the per-block code are a fill).
  Fast-mode decoding of low qualities is up to 1.7x faster (VeryLow 0.58x ->
  0.98x of C++ 4.1.1).
- WebAssembly builds with `simd128` compute the entropy encoder's residual rows
  and dynamic-mode code lengths with SIMD128: lossless entropy encoding in
  dynamic mode is about 1.4x faster in Node/V8.

### Fixed

- Multi-chunk streams (from the C++ encoder with several threads) failed to
  decode with `InvalidData` when `bytes_per_line` exceeds the width, and a
  truncated block-size table or block made `decompress_gray8` panic. Chunks now
  decode straight into their rows of the output, without a temporary buffer
  and copy, with bounds checked.
- `compressed_buffer_size` was smaller than the worst case of lossless entropy
  coding (13 bits per pixel): on noisy images above about 44k pixels,
  `compress_gray8` with `Quality::LosslessEntropy` returned `InvalidArgument`.
  It is now sized like the C++ `llic_compressedBufferSize`.
- Misaligned pointer reads (undefined behaviour): the SSE2/SSE4.1 per-block
  tile encoders read tile rows, and the entropy encoder and its dynamic-mode row
  cost estimate read residual pairs, through misaligned `i32`/`u16` pointers.
- Decoding a truncated tile stream panicked instead of returning
  `InvalidData`.
- `lossy::compress_tile_block` did not check that the source buffer holds the
  image; it now returns `InvalidArgument`.

## [0.5.0] - 2026-10-28

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
- Format version 2 decoding (C++ 2.0.x). It is v3 with a different row split
  between chunks, so only multi-chunk streams needed a change. Formats 0 and 1
  are still rejected.
- Compatibility tests against the C++ reference's decoder vectors
  (`test_data/vectors`, from every C++ release since 2.0.0): all format 2-5
  streams decode to the expected images, and our single-chunk v4 and v5 output
  matches the C++ encoder byte for byte.

### Changed

- **Breaking:** `compress_gray8` writes format v5, like the C++ reference.
  Decoders older than llic_rs 0.5.0 or C++ 3.5.0 cannot read it; use
  `compress_gray8_versioned` with `FormatVersion::V4` for them.
- **Breaking:** the `llic_compress` tool needs the new `cli` feature
  (`cargo install llic_rs --features cli`). `image` and `clap` moved from the
  default `std` feature to `cli`, so the library itself depends only on
  `thiserror` and, with `std`, `num_cpus`.
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
