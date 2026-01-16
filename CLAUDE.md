# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLIC (Low Latency Image Codec) is a fast lossy and lossless grayscale image compression library. This is a Rust port of the original C++ implementation by Peter Rundberg. The codec prioritizes extremely fast compression/decompression over maximum compression ratio.

**Key constraints:**
- Only 8-bit grayscale images supported
- Image dimensions must be non-zero multiples of 4
- Maximum format version: 4

## Build Commands

```bash
# Standard build
cargo build --release

# Run all tests
cargo test --release

# Run linting (clippy + rustfmt)
mise run lint

# Run benchmarks (native optimizations)
mise run bench              # All benchmarks
mise run bench:256          # 256x256 only
mise run bench:rust         # Rust-only (no C++ comparison)

# Profile with flamegraph
mise run profile:all

# WASM builds
mise run wasm:build         # Web (ES modules) with SIMD
mise run wasm:build:nodejs  # Node.js
mise run wasm:release       # All targets (pkg/, pkg-node/, pkg-bundler/)

# Run demo locally
mise run demo:dev

# Setup git hooks (runs tests/lint on commit when Rust files change)
mise run setup:hooks
```

### CLI Tool

```bash
cargo run --release --bin llic_compress -- compress input.png output.llic
cargo run --release --bin llic_compress -- decompress input.llic output.png
```

## Architecture

### Module Structure

```
src/
├── lib.rs           # Main API: LlicContext, Quality, Mode, file format parsing
├── lossy.rs         # Tile-based lossy compression (4x4 blocks)
├── entropy_coder/   # Lossless compression engine
│   ├── u8v1_compress.rs   # Encoder
│   ├── u8v1_decompress.rs # Decoder with two-symbol optimization
│   ├── bit_reader.rs      # Bit stream reader with carryover
│   ├── bit_writer.rs      # Bit stream writer
│   └── tables.rs          # Huffman-like lookup tables
├── wasm.rs          # WebAssembly bindings
├── pgm.rs           # PGM image format I/O
└── ffi.rs           # FFI bindings to C++ reference (benchmarking)
```

### Compression Algorithms

**Lossless (Q0):** Entropy-coded with predictive encoding
- First row: horizontal delta (pixel - previous)
- Other rows: prediction using average of left and top pixels
- Two-symbol decoding: processes symbol pairs via lookup tables

**Lossy (Q1-Q4):** Tile-based 4x4 block compression
- Quantizes each block based on min/max range
- Three streams: min values, dist values, packed pixel indices
- Quality determines quantization bits (Q1=2, Q2=4, Q3=8, Q4=16 error limit)

### File Format (LLSC Container)

The file container format matches the original C++ `llic_compress` tool:
```
LLSC\n
<width> <height>\n
<compressed_size>\n
<binary_compressed_data>
```

The parser also supports legacy files without the "LLSC" magic header for backward compatibility.

Header byte structure of the compressed stream varies by format version (3 or 4). See `LLIC_FORMAT_SPEC.md` for full specification.

### Core API

```rust
let ctx = LlicContext::new(width, height, bytes_per_line, num_threads)?;
let size = ctx.compress_gray8(&pixels, Quality::Lossless, Mode::Default, &mut output)?;
let (quality, mode) = ctx.decompress_gray8(&compressed, &mut output)?;
```

## Testing

Test data is in `test_data/` with pre-compressed files from the C++ reference implementation.

```bash
# Run specific test
cargo test test_gradient_decode

# Run tests with output
cargo test -- --nocapture
```

## Features

- `default` / `std`: CLI and native features (num_cpus, image, clap)
- `wasm`: WebAssembly bindings
- `cpp-reference`: Enable C++ FFI for benchmarking (requires submodule)

## Key Documentation

- `LLIC_FORMAT_SPEC.md` - Complete format specification
- `COMPRESSION_TABLES_GUIDE.md` - Compression table details
- `llic/CLAUDE.md` - C++ implementation notes (in submodule)