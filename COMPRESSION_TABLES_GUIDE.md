# LLIC Compression Tables Guide

This guide explains the structure and usage of the LLIC compression and decompression tables for implementing the Rust version.

## Overview

LLIC uses a custom variable-length entropy coding scheme optimized for fast compression and decompression of image data. The core innovation is processing **two symbols at a time** using precomputed lookup tables, which significantly improves performance.

## Table Structures

### Compression Table (`U8V1_COMPRESS_TABLE_2X`)

```c
extern const uint32_t U8V1_COMPRESS_TABLE_2X[256 * 256];
```

- **Size**: 65,536 entries (256 × 256)
- **Index**: Two consecutive bytes form a 16-bit index: `byte1 | (byte2 << 8)`
- **Entry format**: 32-bit value with two parts:
  - **Upper 26 bits**: Variable-length codes (left-justified)
  - **Lower 6 bits**: Total number of bits for both codes

#### Entry Layout:
```
Bit 31-6: Variable-length codes (left-justified)
Bit 5-0:  Number of bits used (4-32 bits)
```

### Decompression Table (`U8V1_DECOMPRESS_TABLE_2X`)

```c
struct decompress_table_2x_t {
  uint8_t bits;         // Number of bits consumed
  uint8_t num_symbols;  // Number of symbols decoded (1 or 2)
  uint8_t symbol[2];    // The decoded symbols
};

extern const decompress_table_2x_t U8V1_DECOMPRESS_TABLE_2X[1 << 12];
```

- **Size**: 4,096 entries (2^12)
- **Index**: First 12 bits of the compressed stream
- **Entry contains**:
  - `bits`: How many bits this entry consumes (4-16)
  - `num_symbols`: Whether this decodes 1 or 2 symbols
  - `symbol[2]`: The decoded symbol values

## Encoding Scheme

The variable-length codes are designed based on symbol frequency in typical image delta data:

| Symbol | Code Length | Code Pattern | Frequency |
|--------|------------|--------------|-----------|
| 0      | 2 bits     | 00           | Most common (no change) |
| 1, 255 | 3 bits     | 01x, 10x    | Common (small deltas) |
| 2-15, 240-254 | 4-7 bits | Various | Less common |
| 16-239 | 8-13 bits  | Various | Least common |

### Special Cases:
- Symbols requiring 13 bits use a special encoding where the symbol value is directly embedded in bits 19-12
- The compression table can encode pairs of symbols in 4-32 bits total

## Compression Process

### 1. Symbol Preparation
Before compression, image data is transformed to delta encoding:
- **First row**: Horizontal delta only (`pixel[x] - pixel[x-1]`)
- **Other rows**: Combined predictor (`pixel - average(left, above)`)

### 2. Table Lookup
```c
// Process two symbols at once
uint16_t index = symbol1 | (symbol2 << 8);
uint32_t table_entry = U8V1_COMPRESS_TABLE_2X[index];

// Extract components
uint32_t codes = table_entry & 0xFFFFFFC0;  // Upper 26 bits
uint32_t num_bits = table_entry & 0x3F;     // Lower 6 bits
```

### 3. Bit Packing
```c
// Left-justify to 64-bit container
uint64_t bits = ((uint64_t)codes) << 32;

// Add to bit stream
bitContainer |= (bits >> numBits);
numBits += num_bits;

// Flush when >= 32 bits accumulated
if (numBits >= 32) {
    output_32_bits();
    bitContainer <<= 32;
    numBits -= 32;
}
```

## Decompression Process

### 1. Table Lookup
```c
// Get next 12 bits from stream
uint32_t index = bitContainer >> 20;  // Top 12 bits

// Special case: 13-bit codes
if (index >= 0xF80) {  // 0xF80 = 3968
    // Direct symbol extraction
    symbol = (bitContainer >> 19) & 0xFF;
    consume_bits(13);
    return 1;  // One symbol
}

// Normal case: use lookup table
decompress_table_2x_t entry = U8V1_DECOMPRESS_TABLE_2X[index];
symbols[0] = entry.symbol[0];
symbols[1] = entry.symbol[1];
consume_bits(entry.bits);
return entry.num_symbols;
```

### 2. Symbol Reconstruction
After decompression, reconstruct the original values:
- **First row**: `pixel[x] = delta[x] + pixel[x-1]`
- **Other rows**: `pixel[x] = delta[x] + average(pixel[x-1], pixel_above[x])`

## Example Encodings

### Two zeros (most common case):
- Symbols: [0, 0]
- Table index: 0
- Table entry: 0x00000004
- Codes: 0x000000 (empty, as each 0 is 2 bits = 00)
- Total bits: 4
- Bit stream: 0000

### Zero followed by one:
- Symbols: [0, 1] 
- Table index: 256 (0x100)
- Table entry: 0x40000005
- Codes: 0x400000
- Total bits: 5
- Bit stream: 00010 (00 for 0, 010 for 1)

### Handling 13-bit codes:
When a symbol requires 13 bits, it's encoded as:
- Bits 12-5: Fixed prefix (0xF8 or higher)
- Bits 4-0: Lower bits contain part of the symbol value
- The decompressor extracts the symbol directly from bits 19-12

## Performance Optimizations

### 1. Two-Symbol Processing
Processing two symbols at once:
- Reduces table lookups by 50%
- Better cache utilization
- Enables SIMD optimizations for delta computation

### 2. Unrolled Loops
The compression loop is unrolled 8x (16 symbols per iteration):
- Reduces branch overhead
- Allows prefetching
- Better instruction pipelining

### 3. Bit Stream Management
- 64-bit accumulator for efficient bit packing
- Flush only when necessary (>= 32 bits)
- Aligned memory access for output

### 4. Fast Paths
- 12-bit table lookup handles ~99% of symbol pairs
- 13-bit special case for rare symbols
- Optimized for common delta values near zero

## Implementation Notes for Rust

1. **Table Storage**: Consider using `static` arrays or memory-mapped files for the large tables
2. **Bit Manipulation**: Use Rust's built-in bit operations (`<<`, `>>`, `&`, `|`)
3. **Memory Safety**: Ensure bounds checking on table lookups
4. **SIMD**: Consider using Rust's SIMD intrinsics for delta computation
5. **Endianness**: The C++ version assumes little-endian; handle appropriately
6. **Error Handling**: Add bounds checking for compressed data to prevent overruns

## Table Generation

The tables appear to be pre-generated based on:
1. Statistical analysis of typical image delta distributions
2. Huffman-like variable-length coding
3. Optimization for paired symbol encoding
4. Special handling for rare symbols (13-bit codes)

The exact algorithm for generating these tables is not provided in the source, but they encode a custom variable-length code optimized for image compression scenarios where small delta values (especially 0) are most common.