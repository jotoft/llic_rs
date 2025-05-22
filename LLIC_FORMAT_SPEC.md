# LLIC Compressed Image Format Specification

## Overview

LLIC (Lossy and Lossless Image Compression) is a fast grayscale image compression format that supports
both lossless and lossy compression modes. It is optimized for 8-bit grayscale images and requires image dimensions to
be multiples of 4.

## File Structure

LLIC files have two distinct structures:

### 1. Simple Container Format (used by llic_compress tool)

This is a simple text header followed by binary data:

```
<width> <height>\n
<compressed_size>\n
<binary_compressed_data>
```

- **width, height**: ASCII decimal numbers representing image dimensions
- **compressed_size**: ASCII decimal number of compressed data bytes
- **binary_compressed_data**: Raw compressed stream (see below)

### 2. Raw Compressed Stream Format

The actual compressed data stream has the following structure:

#### Header (3 bytes)

| Offset | Size | Description                         |
|--------|------|-------------------------------------|
| 0      | 1    | Format version (currently 3)        |
| 1      | 1    | Number of compressed blocks/threads |
| 2      | 1    | Quality/mode byte (see below)       |

#### Quality/Mode Byte (offset 2)

For lossless compression with default mode:

- Value: 0x00

For lossy/tile-based compression:

- Contains the quality value (0, 2, 4, 8, 16) and is written by the tile-based compressor

#### Block Size Table

Following the header:

- Array of `uint32_t` values (4 bytes each)
- Number of entries = number of blocks (from header byte 1)
- Each entry contains the compressed size of that block

#### Compressed Data Blocks

The actual compressed data follows, with each block's data concatenated.

## Compression Algorithms

### 1. Lossless Compression (u8v1 entropy coder)

Used when quality=LOSSLESS and mode=DEFAULT.

**Algorithm:**

- First row: Horizontal delta encoding (pixel - previous_pixel)
- Subsequent rows: Combined horizontal and vertical prediction
    - Prediction = (left_pixel + top_pixel) / 2
    - Store: pixel - prediction
- Variable-length encoding using pre-computed Huffman-like tables
- Processes pairs of symbols for efficiency

**Two-Symbol Decoding:**

The u8v1 decoder uses an optimized two-symbol decoding approach:

1. **Pair-based decoding**: Each table lookup can yield 1 or 2 symbols
   - Common symbol pairs are encoded together for efficiency
   - Less common symbols are encoded individually
   - Reduces number of bit stream reads and table lookups

2. **Carryover buffer system**: Maintains extra decoded symbols between rows
   - When decoding produces 2 symbols but only 1 is needed for current row
   - Extra symbol is saved in a carryover buffer
   - Next row starts by using carryover symbols before decoding more

3. **Row processing algorithm**:
   ```
   For each row:
   1. Copy any carryover symbols from previous row
   2. Decode symbol pairs until row is filled (width + 1 extra for next row's predictor)
   3. Save excess symbols as carryover for next row
   4. Apply inverse delta/prediction to reconstruct pixels
   ```

This approach minimizes table lookups and improves cache efficiency by processing multiple symbols per operation.

### 2. Tile-Based Compression (4x4 blocks)

Used for all other quality/mode combinations.

**Algorithm:**

For each 4x4 pixel block:

1. Find min and max pixel values
2. Calculate range (dist = max - min)
3. Determine number of bits needed based on quality setting:
    - 0 bits: if range fits within error limit (all pixels = min)
    - 1-8 bits: quantization levels based on quality

**Data Layout per Block:**

- **Min stream**: 1 byte per block (minimum pixel value)
- **Dist stream**: 1 byte per block (pixel range)
- **Pixel stream**: Variable bits per block (quantized pixel indices)

**Block Streams Structure:**

```
[General Header: 1 byte]
[Min values: numBlocks bytes]
[Dist values: numBlocks bytes]
[Pixel data: variable size]
```

**General Header Byte:**

- Bit 7: Header compression flag (1 = compressed, 0 = uncompressed)
- Bits 0-6: Error limit value

**Header Compression:**
When enabled (mode=DEFAULT or quality=LOSSLESS), the min/dist streams are further compressed using the u8v1 entropy
coder.

**Pixel Data Packing:**
Quantized indices are packed efficiently based on bit count:

- 1 bit: 16 values packed into 2 bytes
- 2 bits: 16 values packed into 4 bytes
- 3 bits: 16 values packed into 6 bytes (with specific bit layout)
- 4 bits: 16 values packed into 8 bytes
- 5 bits: 16 values packed into 10 bytes
- 6 bits: 16 values packed into 12 bytes
- 8 bits: 16 values unpacked (16 bytes)

## Quality Levels

| Quality   | Value | Error Limit | Description            |
|-----------|-------|-------------|------------------------|
| LOSSLESS  | 0     | 0           | Perfect reconstruction |
| VERY_HIGH | 2     | 2           | Near-lossless          |
| HIGH      | 4     | 4           | High quality           |
| MEDIUM    | 8     | 8           | Medium quality         |
| LOW       | 16    | 16          | Low quality            |

## Constraints

1. Image dimensions must be non-zero multiples of 4
2. Only 8-bit grayscale images are supported
3. Maximum supported format version is 3
4. Multi-threaded compression splits image into horizontal strips

## Implementation Notes

### Bucket Size Algorithm (Format v2+)

The algorithm determines quantization buckets dynamically:

- Tests 1, 2, 4, 8, 16, 32, 64, or 256 buckets
- Selects minimum buckets where: ceil(range/buckets) <= error_limit + 1

### Task Splitting (Format v3+)

For multi-threaded processing:

- Base block size = (height / num_threads) / 4 * 4
- Last block gets remainder pixels

### Decoding Process

1. Read format version and validate
2. Read number of blocks and quality/mode
3. Read block size table
4. For each block:
    - Decompress using appropriate algorithm
    - Write to correct position in output image

### Reconstruction (Tile-Based)

For each 4x4 block:

1. Read min, dist, and bits from streams
2. If bits > 0: Read and unpack quantized indices
3. Reconstruct: pixel = index * bucketSize + min + (bucketSize/2)
4. If bits = 0: All pixels = min

## Example File Analysis

A lossless 8x8 image compressed file:

```
Header: 03 01 00  (version=3, blocks=1, quality=lossless)
Block size: 10 00 00 00  (16 bytes)
Compressed data: [entropy coded data]
```

A lossy 8x8 image with quality=4:

```
Header: 03 01 04  (version=3, blocks=1, quality=4)
Block size: [size]
General header: 04  (error_limit=4, no header compression)
Min stream: [4 bytes for 4 blocks]
Dist stream: [4 bytes for 4 blocks]
Pixel stream: [variable packed data]
```