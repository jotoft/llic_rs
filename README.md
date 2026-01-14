# LLIC - Low Latency Image Codec (Rust Port)

This is a Rust port of the LLIC image compression library originally written in C++ by Peter Rundberg. The original C++
implementation is included as a git submodule in the `llic/` directory (from https://gitlab.com/llic/llic.git).

## Overview

LLIC (Low Latency Image Codec) is a fast lossy and lossless grayscale image compression format designed for real-time applications where speed is critical. The codec prioritizes extremely fast compression and decompression over maximum compression ratio.

This repository contains:

- **Rust implementation** (main directory) - A port of the original C++ code
- **Original C++ implementation** (`llic/` submodule) - The reference implementation by Peter Rundberg

Both implementations support multiple quality levels from lossless (Q0) to lossy compression with increasing compression
ratios.

**[Try the live demo](https://jotoft.github.io/llic_rs/)**

### Features

- **Multiple quality levels**: Q0 (lossless) through Q4 (highest compression)
- **Parallel implementations**: Both Rust and C++ versions available
- **Multi-threaded processing**: Efficient parallel encoding/decoding
- **PGM format support**: Native support for Portable GrayMap images
- **Custom entropy coding**: Optimized compression tables for efficient encoding

## Building

### Rust Implementation

```bash
# Build the project
cargo build --release

# Run tests
cargo test

# Build and run the CLI tool
cargo run --release -- [OPTIONS]
```

### C++ Implementation

```bash
cd llic
mkdir build && cd build
cmake ..
make -j$(nproc)

# Run tests
./llic_test

# Use the compression tool
./llic_compress input.pgm output.llic [quality]
```

## Usage

### Command Line Interface

```bash
# Compress an image to LLIC (lossless)
cargo run --release --bin llic_compress -- compress input.png output.llic

# Decompress an LLIC file (supports both lossless and lossy)
cargo run --release --bin llic_compress -- decompress input.llic output.png
```

Output format (PNG or PGM) is determined by the file extension.

### Quality Levels

| Level | Name | Description |
|-------|------|-------------|
| Q0 | Lossless | No quality loss |
| Q1 | VeryHigh | Near-lossless (error limit 2) |
| Q2 | High | Balanced quality/compression |
| Q3 | Medium | Higher compression |
| Q4 | Low | Maximum compression |

### Current Status

- **Lossless compression/decompression**: Fully implemented
- **Lossy compression/decompression**: Fully implemented (Q1-Q4)

### Library Usage

```rust
use llic_rs::{LlicContext, Quality, Mode};

// Image dimensions must be multiples of 4
let width = 64;
let height = 64;
let pixels: Vec<u8> = vec![128; width * height]; // grayscale image data

// Create context
let ctx = LlicContext::new(width as u32, height as u32, width as u32, None)?;

// Compress (lossless)
let mut compressed = vec![0u8; ctx.compressed_buffer_size()];
let size = ctx.compress_gray8(&pixels, Quality::Lossless, Mode::Default, &mut compressed)?;
compressed.truncate(size);

// Decompress
let mut output = vec![0u8; width * height];
let (quality, mode) = ctx.decompress_gray8(&compressed, &mut output)?;
```

## Technical Details

LLIC uses a custom entropy coding scheme with pre-computed compression tables optimized for image data patterns. The
format specification can be found in [LLIC_FORMAT_SPEC.md](LLIC_FORMAT_SPEC.md).

### Key Components

- **Entropy Coder**: Custom u8v1 compression/decompression algorithm
- **Block Processing**: Images are processed in blocks for efficient parallelization
- **Adaptive Quantization**: Quality levels control the quantization of image data

## Performance

The Rust implementation leverages Rayon for parallel processing, while the C++ version uses OpenMP. Both implementations
are optimized for multi-core processors.

## Acknowledgements

This project is based on the original LLIC C++ implementation by **Peter Rundberg**. The Rust implementation is a port
and enhancement of the original work.

### Credits

- **Original Author**: Peter Rundberg - Original C++ LLIC implementation
- **C++ Dependencies**:
    - [doctest](https://github.com/onqtam/doctest) - C++ testing framework
    - [rmgr-ssim](https://github.com/rmgr/ssim) - SSIM calculation library

### License Note

The C++ implementation is licensed under the zlib License (see `llic/LICENCE`). The Rust implementation follows the same
licensing terms.

## Contributing

Contributions are welcome! Please ensure that:

1. All tests pass (`cargo test` and `./llic_test`)
2. Code follows the existing style conventions
3. New features include appropriate tests
4. Documentation is updated as needed

## Project Status

This is an experimental image compression format under active development. While functional, it may not be suitable for
production use without further testing and optimization.

## Repository Structure

```
.
├── src/                    # Rust implementation
├── llic/                   # C++ implementation (git submodule)
├── demo/                   # Interactive WASM demo
├── pkg/                    # WASM package (web/ES modules)
├── pkg-node/               # WASM package (Node.js)
├── pkg-bundler/            # WASM package (bundlers)
├── tables/                 # Pre-computed compression tables
├── test_data/              # Test images and compressed samples
├── LLIC_FORMAT_SPEC.md     # Format specification
└── COMPRESSION_TABLES_GUIDE.md  # Compression tables documentation
```

## Contact

For questions, issues, or contributions, please open an issue on GitHub.