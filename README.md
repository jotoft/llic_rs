# LLIC - Lossless/Lossy Image Compression (Rust Port)

This is a Rust port of the LLIC image compression library originally written in C++ by Peter Rundberg. The original C++
implementation is included as a git submodule in the `llic/` directory (from https://gitlab.com/llic/llic.git).

## Overview

LLIC (Lossless/Lossy Image Compression) is an experimental image compression format designed for efficient encoding of
grayscale images. This repository contains:

- **Rust implementation** (main directory) - A port and enhancement of the original C++ code
- **Original C++ implementation** (`llic/` submodule) - The reference implementation by Peter Rundberg

Both implementations support multiple quality levels from lossless (Q0) to lossy compression with increasing compression
ratios.

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
# Compress an image (Rust)
cargo run --release -- encode input.pgm output.llic --quality 2

# Decompress an image (Rust)
cargo run --release -- decode input.llic output.pgm

# Compress an image (C++)
./llic_compress input.pgm output.llic 2
```

### Quality Levels

- **Q0**: Lossless compression
- **Q1**: Near-lossless with minimal quality loss
- **Q2**: Balanced quality/compression ratio
- **Q3**: Higher compression with visible quality reduction
- **Q4**: Maximum compression

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
├── llic/                   # C++ implementation
├── tables/                 # Pre-computed compression tables
├── test_data/             # Test images and compressed samples
├── LLIC_FORMAT_SPEC.md    # Format specification
└── COMPRESSION_TABLES_GUIDE.md  # Compression tables documentation
```

## Contact

For questions, issues, or contributions, please open an issue on GitHub.