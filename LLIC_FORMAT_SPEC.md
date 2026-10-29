# LLIC Compressed Image Format Specification

**Note**: This specification has been reverse-engineered from the original C++ implementation by Peter Rundberg. It is not an official specification but rather aims to document the format as implemented in the reference code. The format and this documentation should be considered experimental.

## Overview

LLIC (Low Latency Image Codec) is a fast grayscale image compression format that supports
both lossless and lossy compression modes. It is optimized for 8-bit grayscale images and requires image dimensions to
be multiples of 4.

## File Structure

LLIC files have two distinct structures:

### 1. Simple Container Format (used by llic_compress tool)

This is a simple text header followed by binary data:

```
LLSC\n
<width> <height>\n
<compressed_size>\n
<binary_compressed_data>
```

- **LLSC**: Magic identifier. Parsers also accept legacy files without it.
- **width, height**: ASCII decimal numbers representing image dimensions
- **compressed_size**: ASCII decimal number of compressed data bytes
- **binary_compressed_data**: Raw compressed stream (see below)

### 2. Raw Compressed Stream Format

Byte 0 of the stream is the format version. Three versions are emitted and
decoded by this implementation: v5 (current, C++ 3.5.0 and later), v4 and v3
(legacy). v3 and v4 differ only in the fixed header. v5 has the v4 header and
differs only in where a tile block stores a compressed min/dist header (see
[Header Compression](#header-compression)). The block-size table and all other
payloads are identical across versions. v2 (C++ 2.0.x) is also decoded: it is
v3 with a different row split between blocks (see [Task
Splitting](#task-splitting)).

#### Format v4/v5 Header (4 bytes)

| Offset | Size | Description                                          |
|--------|------|------------------------------------------------------|
| 0      | 1    | Format version = 4 or 5                              |
| 1      | 1    | Number of compressed blocks (chunks), 1-16           |
| 2      | 1    | Stream type: 0 = entropy-coded lossless, 1 = tile-based |
| 3      | 1    | Mode: 0 = Default, 1 = Fast, 2 = Dynamic             |

The block-size table starts at offset 4.

#### Format v3 Header (variable)

v3 has no stream-type or mode byte. The decoder discriminates on byte 2:

- **Entropy-coded lossless**: byte 2 is a `0x00` marker; the block-size table
  starts at offset 3.

  | Offset | Size | Description                         |
  |--------|------|-------------------------------------|
  | 0      | 1    | Format version = 3                  |
  | 1      | 1    | Number of compressed blocks (chunks) |
  | 2      | 1    | 0x00 (entropy-coded marker)         |

- **Tile-based**: the block-size table starts directly at offset 2, so byte 2
  is the least-significant byte of the first block size and **must be
  non-zero**. There is no quality byte in the header — the quality (error
  limit) lives in bits 0–6 of the first payload byte, and the mode in bit 7
  (1 = compressed tile headers = Default, 0 = Fast). Dynamic mode is not
  representable in v3.

  | Offset | Size | Description                         |
  |--------|------|-------------------------------------|
  | 0      | 1    | Format version = 3                  |
  | 1      | 1    | Number of compressed blocks (chunks) |
  | 2      | ...  | Block-size table (LSB must be ≠ 0)  |

  **Writer rule (padding)**: if a tile-based payload length is a multiple of
  256, the writer must append one zero pad byte and count it in the block
  size, so that the size's LSB stays non-zero. Decoders use block sizes only
  to locate the next block and read tile streams by need, so the pad byte is
  never read. In practice the current encoders always produce odd payload
  lengths (the entropy coder flushes 16-bit words and pixel data is an even
  byte count per tile), so the pad is defensive.

#### Block Size Table

Following the fixed header:

- Array of `uint32_t` values (4 bytes each, little-endian)
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

#### Header Compression

In Default and Dynamic mode, the min/dist streams are further compressed with the u8v1 entropy coder, as a
`(width/4) x (rows/4 * 2)` image with the mins above the dists (Dynamic mode uses the dynamic predictor). The
compressed header is used only when it saves more than the 4-byte field it needs (`compressed + 4 < numBlocks * 2`);
otherwise the block keeps the raw layout above with bit 7 clear. With bit 7 set:

```
v3/v4: [General Header: 1 byte][header_size: u32 LE][compressed header][Pixel data]
v5:    [General Header: 1 byte][header_offset: u32 LE][Pixel data][compressed header]
```

In v5, `header_offset` is counted from the start of the block (so it is at least 5) and the compressed header runs to
the end of the block. The encoder can then write the pixel stream before it knows the header size.

**Lossless header selection (v5 encoders):** lossless decoding adds the residuals to the block minimum and uses the
distance only for its bit width, so a block decodes the same with any `min` in `[max - (2^bits - 1), min]` and any
`dist` of the same width (0-6 bits for `dist < 64`, else 8). When the header is compressed at quality 0, the encoder
picks the values in those ranges nearest the block above (clamping the above block's `min` and `dist` into them) and
stores the residuals relative to the chosen `min`. This makes lossless default/dynamic streams 2-3% smaller and needs
no decoder change. Rows of blocks at the top of a chunk keep their own values.

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
| VERY_LOW  | 32    | 32          | Very low quality       |

## Constraints

1. Image dimensions must be non-zero multiples of 4
2. Only 8-bit grayscale images are supported
3. Maximum supported format version is 5; the decoder accepts versions 2 to 5
4. Multi-threaded compression splits image into horizontal strips

## Implementation Notes

### Bucket Size Algorithm (Format v2+)

The algorithm determines quantization buckets dynamically:

- Tests 1, 2, 4, 8, 16, 32, 64, or 256 buckets
- Selects minimum buckets where: ceil(range/buckets) <= error_limit + 1

### Task Splitting

The image is split into horizontal strips, one per block (chunk), each coded
on its own. The C++ reference writes one block per encoder thread; llic_rs
takes the count as a context setting (`LlicContext::new`'s `num_chunks`).
Format v3 and later:

- Base block size = (height / num_blocks) / 4 * 4
- Last block gets remainder pixels

Format v2 (C++ 2.0.x; otherwise identical to v3) rounds up instead:

- Block size = height / num_blocks, rounded up to a multiple of 4
- Later blocks shrink (to zero if needed) so the total does not exceed the
  height. This can leave rows unassigned (24 rows in 5 blocks covers 20), which
  is why v3 changed the split; such streams are invalid.

Formats v0 and v1 also use an older bucket-size algorithm and are not supported.

### Decoding Process

1. Read format version and validate
2. Read number of blocks
3. Determine stream type and mode: from header bytes 2 and 3 (v4/v5), or from
   the byte-2 discriminator and the first payload byte (v3)
4. Read block size table
5. For each block:
    - Decompress using appropriate algorithm
    - Write to correct position in output image

### Reconstruction (Tile-Based)

For each 4x4 block:

1. Read min, dist, and bits from streams
2. If bits > 0: Read and unpack quantized indices
3. Reconstruct: pixel = index * bucketSize + min + (bucketSize/2)
4. If bits = 0: All pixels = min

## Example File Analysis

A v3 entropy-coded lossless 8x8 image:

```
Header: 03 01 00  (version=3, blocks=1, entropy-coded marker)
Block size: 10 00 00 00  (16 bytes)
Compressed data: [entropy coded data]
```

A v3 tile-based 8x8 image with quality=4 (note: no quality byte in the
header — the block-size table follows the version and block count directly):

```
Header: 03 01  (version=3, blocks=1)
Block size: [size, LSB must be non-zero]
General header: 04  (error_limit=4, bit 7 clear = uncompressed header = Fast)
Min stream: [4 bytes for 4 tiles]
Dist stream: [4 bytes for 4 tiles]
Pixel stream: [variable packed data]
```

The same image in v4:

```
Header: 04 01 01 00  (version=4, blocks=1, tile-based, mode=Default)
Block size: [size]
General header + streams: [as above]
```