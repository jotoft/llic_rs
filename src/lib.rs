use thiserror::Error;

pub const API_VERSION_MAJOR: u32 = 3;
pub const API_VERSION_MINOR: u32 = 0;

/// Format version for compressed streams
pub const FORMAT_VERSION: u8 = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quality {
    /// Lossless compression (tile-based, error_limit=0)
    Lossless = 0,
    /// Near-lossless, error limit ±2
    VeryHigh = 2,
    /// High quality, error limit ±4
    High = 4,
    /// Medium quality, error limit ±8
    Medium = 8,
    /// Low quality, error limit ±16
    Low = 16,
    /// Very low quality, error limit ±32
    VeryLow = 32,
    /// Legacy entropy-coded lossless (for backward compatibility)
    LosslessEntropy = 64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Default = 0,
    Fast = 1,
    Dynamic = 2,
}

/// Container format version for compressed streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormatVersion {
    /// Legacy format without a mode byte:
    /// `[3][num_blocks][block-size table or 0x00 entropy marker]...`
    V3 = 3,
    /// `[4][num_blocks][tile_based][mode][block-size table]...`, readable by
    /// the C++ reference before 3.5.0 and llic_rs before 0.5.0.
    V4 = 4,
    /// Current format: the v4 header, but tile chunks with an entropy-coded
    /// header store it after the pixel stream (C++ reference 3.5.0 and later).
    /// Lossless tile streams in default and dynamic mode are also smaller.
    V5 = 5,
}

#[derive(Error, Debug)]
pub enum LlicError {
    #[error("Out of memory")]
    OutOfMemory,
    #[error("Invalid image dimensions (must be multiple of 4)")]
    ImageDimensions,
    #[error("Unsupported format")]
    UnsupportedFormat,
    #[error("Invalid argument")]
    InvalidArgument,
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid compressed data")]
    InvalidData,
    #[error("File I/O error")]
    FileIO,
}

pub type Result<T> = std::result::Result<T, LlicError>;

/// Most chunks a stream can hold, as in the C++ reference (`LLIC_MAX_CHUNKS`).
pub const MAX_CHUNKS: usize = 16;

pub struct LlicContext {
    width: u32,
    height: u32,
    bytes_per_line: u32,
    num_chunks: usize,
}

impl LlicContext {
    /// Creates a context for `width` x `height` images, `bytes_per_line` apart.
    ///
    /// `num_chunks` is how many horizontal strips the encoder splits each
    /// image into (default 1). Every chunk is coded on its own, with its own
    /// tile header, as the C++ reference writes one chunk per thread. Up to
    /// [`MAX_CHUNKS`], and at most one per four rows. Decoding takes the
    /// count from the stream, so it accepts any count.
    pub fn new(
        width: u32,
        height: u32,
        bytes_per_line: u32,
        num_chunks: Option<usize>,
    ) -> Result<Self> {
        if !width.is_multiple_of(4) || !height.is_multiple_of(4) {
            return Err(LlicError::ImageDimensions);
        }

        if width == 0 || height == 0 {
            return Err(LlicError::InvalidArgument);
        }

        let num_chunks = num_chunks.unwrap_or(1);
        if !(1..=MAX_CHUNKS).contains(&num_chunks) || num_chunks > height as usize / 4 {
            return Err(LlicError::InvalidArgument);
        }

        Ok(Self {
            width,
            height,
            bytes_per_line,
            num_chunks,
        })
    }

    /// The number of chunks the encoder splits each image into.
    pub fn num_chunks(&self) -> usize {
        self.num_chunks
    }

    /// Size of an output buffer that holds any stream `compress_gray8` and
    /// `compress_gray8_versioned` can write for this context, as the C++
    /// `llic_compressedBufferSize` computes it: the container header and, per
    /// chunk, its size entry and the larger of the entropy-coded worst case
    /// (13 bits per pixel plus the dynamic predictor bitmap, and 4 bytes the
    /// in-place encoder may write past it) and the tile-based one.
    pub fn compressed_buffer_size(&self) -> usize {
        let chunks = (0..self.num_chunks).map(|chunk| {
            #[allow(clippy::cast_possible_truncation)] // at most the height
            let rows =
                chunk_rows(FORMAT_VERSION, self.height as usize, self.num_chunks, chunk) as u32;
            let entropy = entropy_coder::max_compressed_size(self.width, rows, true);
            let tile = lossy::tile_block_scratch_size(self.width, rows);
            4 + entropy.max(tile)
        });
        V4_FIELDS_LEN + chunks.sum::<usize>()
    }

    pub fn decompress_gray8(
        &self,
        src_data: &[u8],
        dst_graymap: &mut [u8],
    ) -> Result<(Quality, Mode)> {
        if src_data.len() < 4 {
            return Err(LlicError::InvalidData);
        }

        // Parse header
        let version = src_data[0];
        let num_blocks = src_data[1];

        // Support format versions 2 to 5.
        //
        // v2 (C++ 2.0.x) is v3 with a different row split between chunks (see
        // `chunk_rows`). v0/v1 share the layout too but also use the pre-2.0
        // bucket-size algorithm (`findBucketsNeededOld`), which this port does
        // not implement, so we reject them rather than risk a silent misdecode.
        if !(2..=5).contains(&version) {
            return Err(LlicError::UnsupportedFormat);
        }

        // Parse format-specific header fields
        let (tile_based, mut mode, block_sizes_start) = if version >= 4 {
            // Format v4: [version][num_blocks][tile_based][mode][block_sizes...]
            let tile_based = src_data[2] != 0;
            let mode = match src_data[3] {
                0 => Mode::Default,
                1 => Mode::Fast,
                2 => Mode::Dynamic,
                _ => Mode::Default,
            };
            (tile_based, mode, 4usize)
        } else {
            // Format v3: [version][num_blocks][quality_or_block_size_byte...]
            // If byte 2 is 0, it's entropy-coded lossless (tile_based = false)
            // If byte 2 is non-0, it's tile-based (byte 2 is first byte of block sizes)
            let is_entropy_lossless = src_data[2] == 0;
            let tile_based = !is_entropy_lossless;
            let block_sizes_start = if is_entropy_lossless { 3 } else { 2 };
            (tile_based, Mode::Default, block_sizes_start)
        };

        // Read block size table (always little-endian)
        let mut pos = block_sizes_start;
        let mut block_sizes = Vec::with_capacity(num_blocks as usize);
        for _ in 0..num_blocks {
            if pos + 4 > src_data.len() {
                return Err(LlicError::InvalidData);
            }
            let size = u32::from_le_bytes([
                src_data[pos],
                src_data[pos + 1],
                src_data[pos + 2],
                src_data[pos + 3],
            ]);
            block_sizes.push(size);
            pos += 4;
        }

        // Determine quality from block data
        let quality = if tile_based {
            // For tile-based, quality (error_limit) is in the first byte of first block data
            if pos >= src_data.len() {
                return Err(LlicError::InvalidData);
            }
            let first_block_byte = src_data[pos];
            let error_limit = first_block_byte & 0x7f;

            // Legacy streams have no mode byte; tile-based encodes the mode in
            // bit 7 of the first payload byte (set = compressed tile headers =
            // Default, clear = Fast), matching the C++ reference decoder.
            if version < 4 {
                mode = if first_block_byte & 0x80 != 0 {
                    Mode::Default
                } else {
                    Mode::Fast
                };
            }

            match error_limit {
                0 => Quality::Lossless,
                2 => Quality::VeryHigh,
                4 => Quality::High,
                8 => Quality::Medium,
                16 => Quality::Low,
                32 => Quality::VeryLow,
                _ => Quality::Lossless, // Default fallback for unknown
            }
        } else {
            // Entropy-coded is always lossless
            Quality::LosslessEntropy
        };

        // Helper closure to decompress a block based on mode and tile_based flag
        let layout = lossy::TileHeaderLayout::for_format_version(version);
        let decompress_block =
            |data: &[u8], w: u32, h: u32, bpl: u32, dst: &mut [u8]| -> Result<()> {
                if tile_based {
                    lossy::decompress_tile_block_dynamic(
                        data,
                        w,
                        h,
                        bpl,
                        mode == Mode::Dynamic,
                        layout,
                        dst,
                    )
                } else if mode == Mode::Dynamic {
                    entropy_coder::decompress_dynamic(data, w, h, bpl, dst)
                } else {
                    entropy_coder::decompress(data, w, h, bpl, dst)
                }
            };

        // Each block (one per encoder thread) holds a horizontal strip of rows,
        // split as `chunk_rows` says; strips with no rows have empty blocks.
        // Blocks decode straight into their rows of the output.
        let (width, height) = (self.width as usize, self.height as usize);
        let stride = self.bytes_per_line as usize;
        if dst_graymap.len() < (height - 1) * stride + width {
            return Err(LlicError::InvalidArgument);
        }
        let mut row = 0usize;
        for (index, &size) in block_sizes.iter().enumerate() {
            let data_end = pos
                .checked_add(size as usize)
                .ok_or(LlicError::InvalidData)?;
            let data = src_data.get(pos..data_end).ok_or(LlicError::InvalidData)?;
            pos = data_end;
            let rows = chunk_rows(version, height, block_sizes.len(), index);
            if rows == 0 {
                continue;
            }
            if data.is_empty() || row + rows > height {
                return Err(LlicError::InvalidData);
            }
            let strip = &mut dst_graymap[row * stride..(row + rows - 1) * stride + width];
            #[allow(clippy::cast_possible_truncation)] // at most the height
            decompress_block(data, self.width, rows as u32, self.bytes_per_line, strip)?;
            row += rows;
        }
        if row != height {
            return Err(LlicError::InvalidData);
        }

        Ok((quality, mode))
    }

    /// Compress grayscale image data using the current format version (v5).
    ///
    /// Equivalent to calling [`compress_gray8_versioned`](Self::compress_gray8_versioned)
    /// with [`FormatVersion::V5`]. Use [`FormatVersion::V4`] for decoders older
    /// than C++ 3.5.0 or llic_rs 0.5.0.
    pub fn compress_gray8(
        &self,
        src_graymap: &[u8],
        quality: Quality,
        mode: Mode,
        dst_data: &mut [u8],
    ) -> Result<usize> {
        self.compress_gray8_versioned(src_graymap, quality, mode, FormatVersion::V5, dst_data)
    }

    /// Compress grayscale image data with an explicit format version.
    ///
    /// # Arguments
    /// * `src_graymap` - Source pixel data (row-major, 8-bit grayscale)
    /// * `quality` - Compression quality level
    /// * `mode` - Compression mode (Default, Fast, Dynamic)
    /// * `format_version` - LLIC container format version
    /// * `dst_data` - Output buffer for compressed data
    ///
    /// # Returns
    /// Number of bytes written to dst_data, or error.
    ///
    /// # Format
    /// The container header, one `u32 LE` size per chunk, then the chunks in
    /// row order (see [`LlicContext::new`] for the chunk count):
    /// - v5: as v4 with `version=5`; tile chunks with a compressed header put it
    ///   after the pixel stream (see [`lossy::TileHeaderLayout`])
    /// - v4: `[version=4][num_chunks][tile_based][mode][chunk_sizes][chunks]`
    /// - v3 entropy-coded: `[version=3][num_chunks][0x00][chunk_sizes][chunks]`
    /// - v3 tile-based: `[version=3][num_chunks][chunk_sizes][chunks]`
    ///
    /// In v3 the entropy-vs-tile discriminator is "byte 2 == 0", so for v3
    /// tile-based the least-significant byte of the first chunk size must be
    /// non-zero. When that chunk's length lands on a multiple of 256, one pad
    /// byte is appended (and counted in its size) to keep the LSB non-zero;
    /// decoders never read it.
    ///
    /// v3 has no mode byte. For tile-based streams the mode is carried in
    /// bit 7 of the first payload byte, so `Default` and `Fast` round-trip;
    /// for entropy-coded streams `Fast` is identical to `Default` and decodes
    /// as `Default`. `Dynamic` is not representable in v3 and returns
    /// `UnsupportedFormat`.
    pub fn compress_gray8_versioned(
        &self,
        src_graymap: &[u8],
        quality: Quality,
        mode: Mode,
        format_version: FormatVersion,
        dst_data: &mut [u8],
    ) -> Result<usize> {
        if format_version == FormatVersion::V3 && mode == Mode::Dynamic {
            return Err(LlicError::UnsupportedFormat);
        }

        let expected_size = self.height as usize * self.bytes_per_line as usize;
        if src_graymap.len() < expected_size {
            return Err(LlicError::InvalidArgument);
        }

        let use_entropy = quality == Quality::LosslessEntropy;
        let error_limit = match quality {
            Quality::Lossless | Quality::LosslessEntropy => 0,
            Quality::VeryHigh => 2,
            Quality::High => 4,
            Quality::Medium => 8,
            Quality::Low => 16,
            Quality::VeryLow => 32,
        };
        let dynamic = mode == Mode::Dynamic;
        let compress_header = mode != Mode::Fast;
        let layout = lossy::TileHeaderLayout::for_format_version(format_version as u8);
        let (width, bpl) = (self.width, self.bytes_per_line);

        // Chunk payloads are identical across versions, apart from where a tile
        // chunk keeps a compressed header (see `TileHeaderLayout`).
        let scratch_size = |rows: u32| {
            if use_entropy {
                entropy_coder::max_compressed_size(width, rows, dynamic)
            } else {
                lossy::tile_block_scratch_size(width, rows)
            }
        };
        let encode_chunk = |src: &[u8], rows: u32, out: &mut [u8]| {
            if !use_entropy {
                lossy::compress_tile_block_into(
                    src,
                    width,
                    rows,
                    bpl,
                    error_limit,
                    compress_header,
                    dynamic,
                    layout,
                    out,
                )
            } else if dynamic {
                entropy_coder::compress_dynamic_into(src, width, rows, bpl, out)
            } else {
                entropy_coder::compress_into(src, width, rows, bpl, out)
            }
        };

        let num_chunks = self.num_chunks;
        let fields_len =
            write_container_fields(dst_data, format_version, !use_entropy, mode, num_chunks)?;
        let mut pos = fields_len + 4 * num_chunks;
        if dst_data.len() < pos {
            return Err(LlicError::InvalidArgument);
        }
        let mut row = 0usize;
        for chunk in 0..num_chunks {
            let rows = chunk_rows(
                format_version as u8,
                self.height as usize,
                num_chunks,
                chunk,
            );
            let src = &src_graymap[row * bpl as usize..];
            #[allow(clippy::cast_possible_truncation)] // at most the height
            let rows32 = rows as u32;
            // Encode in place when the rest of the buffer has room for the
            // chunk's worst case (compressed_buffer_size always does), else
            // through a scratch buffer.
            let scratch = scratch_size(rows32);
            let mut size = if dst_data.len() - pos >= scratch {
                encode_chunk(src, rows32, &mut dst_data[pos..])?
            } else {
                let mut tmp = vec![0u8; scratch];
                let size = encode_chunk(src, rows32, &mut tmp)?;
                dst_data
                    .get_mut(pos..pos + size)
                    .ok_or(LlicError::InvalidArgument)?
                    .copy_from_slice(&tmp[..size]);
                size
            };
            if chunk == 0 && format_version == FormatVersion::V3 && !use_entropy {
                size = pad_v3_tile_chunk(dst_data, pos, size)?;
            }
            #[allow(clippy::cast_possible_truncation)] // chunks are far below 4 GiB
            let size_field = (size as u32).to_le_bytes();
            dst_data[fields_len + 4 * chunk..fields_len + 4 * chunk + 4]
                .copy_from_slice(&size_field);
            pos += size;
            row += rows;
        }
        Ok(pos)
    }
}

/// Rows in chunk `index` of `num_chunks`, split as the C++ reference does.
///
/// - v2 (`computeTaskSplitV1`): `height / num_chunks` rounded up to a multiple
///   of 4; the last chunks shrink (to zero if needed) so the total fits.
///   76 rows in 3 chunks: 28, 28, 20.
/// - v3+ (`computeTaskSplitV2`): rounded down; the last chunk takes the rest.
///   76 rows in 3 chunks: 24, 24, 28.
fn chunk_rows(version: u8, height: usize, num_chunks: usize, index: usize) -> usize {
    if version <= 2 {
        let lines = (height / num_chunks).next_multiple_of(4);
        lines.saturating_sub(((index + 1) * lines).saturating_sub(height))
    } else {
        let base = (height / num_chunks) / 4 * 4;
        if index + 1 == num_chunks {
            height - base * (num_chunks - 1)
        } else {
            base
        }
    }
}

/// Container header fields before the chunk size table in v4/v5:
/// `[version][num_chunks][tile_based][mode]`.
const V4_FIELDS_LEN: usize = 4;

/// Write the container header fields that precede the chunk size table and
/// return their length:
/// - v4/v5: `[version][num_chunks][tile_based][mode]`
/// - v3 entropy-coded: `[3][num_chunks][0x00]`
/// - v3 tile-based: `[3][num_chunks]`
fn write_container_fields(
    dst_data: &mut [u8],
    version: FormatVersion,
    tile_based: bool,
    mode: Mode,
    num_chunks: usize,
) -> Result<usize> {
    #[allow(clippy::cast_possible_truncation)] // at most MAX_CHUNKS
    let count = num_chunks as u8;
    let fields: &[u8] = match version {
        FormatVersion::V4 | FormatVersion::V5 => {
            &[version as u8, count, tile_based as u8, mode as u8]
        }
        FormatVersion::V3 if tile_based => &[3, count],
        FormatVersion::V3 => &[3, count, 0],
    };
    dst_data
        .get_mut(..fields.len())
        .ok_or(LlicError::InvalidArgument)?
        .copy_from_slice(fields);
    Ok(fields.len())
}

/// Pad the first chunk of a v3 tile-based stream, `size` bytes at `pos`.
///
/// The v3 decoder treats byte 2 == 0 as entropy-coded, and byte 2 is the LSB
/// of the first chunk's size, so it must be non-zero: when the size is a
/// multiple of 256, one zero pad byte is appended and counted in it. Returns
/// the size to record. Decoders use chunk sizes only to locate the next chunk
/// and read tile streams by need, so the pad byte is never read (both in this
/// crate and the C++ reference). The current tile encoder always produces odd
/// payload lengths (the bit writer flushes 16-bit words and pixel data is an
/// even byte count per tile), so the padding is defensive.
fn pad_v3_tile_chunk(dst_data: &mut [u8], pos: usize, size: usize) -> Result<usize> {
    if !size.is_multiple_of(256) {
        return Ok(size);
    }
    *dst_data
        .get_mut(pos + size)
        .ok_or(LlicError::InvalidArgument)? = 0;
    Ok(size + 1)
}

pub mod entropy_coder;
pub mod helpers;
pub mod lossy;
#[cfg(feature = "std")]
pub mod pgm;
#[cfg(feature = "wasm")]
pub mod wasm;

// Export aliases for convenience
pub use Mode as CompressionMode;
pub use Quality as CompressionQuality;

/// Compute prediction residual for grayscale image data.
///
/// Uses the same predictor as the lossless compression:
/// - First pixel: stored as-is (no prediction)
/// - First row: horizontal prediction (pixel - left)
/// - First column: vertical prediction (pixel - top)
/// - Other pixels: average prediction (pixel - (left + top) / 2)
///
/// Returns residuals mapped to 0-255 where 128 = zero residual.
pub fn compute_prediction_residual(data: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut residual = Vec::with_capacity(width * height);

    for y in 0..height {
        for x in 0..width {
            let pixel = data[y * width + x] as i16;
            let predicted = if y == 0 {
                if x == 0 {
                    0
                } else {
                    data[y * width + x - 1] as i16
                }
            } else if x == 0 {
                data[(y - 1) * width + x] as i16
            } else {
                let left = data[y * width + x - 1] as i16;
                let top = data[(y - 1) * width + x] as i16;
                (left + top) >> 1
            };
            let delta = pixel - predicted;
            // Map signed delta to unsigned: 128 = zero, <128 = negative, >128 = positive
            residual.push((delta + 128).clamp(0, 255) as u8);
        }
    }

    residual
}

/// LLSC file format magic header.
///
/// The file container format follows the original C++ implementation:
/// ```text
/// LLSC\n
/// <width> <height>\n
/// <compressed_size>\n
/// <binary_compressed_data>
/// ```
pub const LLSC_MAGIC: &[u8] = b"LLSC";

// File I/O convenience functions (only available with std feature)
#[cfg(feature = "std")]
pub fn decode_file(input_path: &str, output_path: &str) -> Result<()> {
    use std::fs::File;
    use std::io::Read;

    // Read the LLIC file
    let mut file = File::open(input_path).map_err(|_| LlicError::FileIO)?;
    let mut llic_data = Vec::new();
    file.read_to_end(&mut llic_data)
        .map_err(|_| LlicError::FileIO)?;

    // Parse the simple container format to extract dimensions and compressed data
    let (width, height, compressed_data) = parse_llic_container(&llic_data)?;

    // Create context and decompress
    let context = LlicContext::new(width, height, width, None)?;
    let mut image_data = vec![0u8; (width * height) as usize];
    context.decompress_gray8(&compressed_data, &mut image_data)?;

    // Save as PGM
    pgm::write(output_path, width, height, &image_data)?;

    Ok(())
}

#[cfg(feature = "std")]
pub fn encode_file(
    _input_path: &str,
    _output_path: &str,
    _quality: CompressionQuality,
    _mode: CompressionMode,
) -> Result<()> {
    todo!("Encoding not yet implemented")
}

/// Write compressed data to LLSC file format.
///
/// Format follows the original C++ implementation:
/// ```text
/// LLSC\n
/// <width> <height>\n
/// <compressed_size>\n
/// <binary_compressed_data>
/// ```
#[cfg(feature = "std")]
pub fn write_llic_file(path: &str, width: u32, height: u32, compressed_data: &[u8]) -> Result<()> {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(path).map_err(|_| LlicError::FileIO)?;

    // Write LLSC magic header
    writeln!(file, "LLSC").map_err(|_| LlicError::FileIO)?;
    writeln!(file, "{} {}", width, height).map_err(|_| LlicError::FileIO)?;
    writeln!(file, "{}", compressed_data.len()).map_err(|_| LlicError::FileIO)?;
    file.write_all(compressed_data)
        .map_err(|_| LlicError::FileIO)?;

    Ok(())
}

/// Read compressed data from LLSC file format.
///
/// Supports both formats:
/// - New format with "LLSC" magic header (from C++ llic_compress)
/// - Legacy format without magic header (width height on first line)
#[cfg(feature = "std")]
pub fn read_llic_file(path: &str) -> Result<(u32, u32, Vec<u8>)> {
    use std::fs::File;
    use std::io::Read;

    let mut file = File::open(path).map_err(|_| LlicError::FileIO)?;
    let mut data = Vec::new();
    file.read_to_end(&mut data).map_err(|_| LlicError::FileIO)?;

    parse_llic_container(&data)
}

#[cfg(feature = "std")]
fn parse_llic_container(data: &[u8]) -> Result<(u32, u32, Vec<u8>)> {
    use std::str;

    // Find first newline
    let first_newline = data
        .iter()
        .position(|&b| b == b'\n')
        .ok_or(LlicError::UnsupportedFormat)?;

    let first_line =
        str::from_utf8(&data[..first_newline]).map_err(|_| LlicError::UnsupportedFormat)?;

    // Check if this is the new format with LLSC magic header
    let (width, height, size_line_start) = if first_line == "LLSC" {
        // New format: LLSC\n<width> <height>\n<size>\n<data>
        let second_start = first_newline + 1;
        let second_newline = data[second_start..]
            .iter()
            .position(|&b| b == b'\n')
            .ok_or(LlicError::UnsupportedFormat)?
            + second_start;

        let dims_line = str::from_utf8(&data[second_start..second_newline])
            .map_err(|_| LlicError::UnsupportedFormat)?;

        let parts: Vec<&str> = dims_line.split_whitespace().collect();
        if parts.len() != 2 {
            return Err(LlicError::UnsupportedFormat);
        }

        let width: u32 = parts[0].parse().map_err(|_| LlicError::UnsupportedFormat)?;
        let height: u32 = parts[1].parse().map_err(|_| LlicError::UnsupportedFormat)?;

        (width, height, second_newline + 1)
    } else {
        // Legacy format: <width> <height>\n<size>\n<data>
        let parts: Vec<&str> = first_line.split_whitespace().collect();
        if parts.len() != 2 {
            return Err(LlicError::UnsupportedFormat);
        }

        let width: u32 = parts[0].parse().map_err(|_| LlicError::UnsupportedFormat)?;
        let height: u32 = parts[1].parse().map_err(|_| LlicError::UnsupportedFormat)?;

        (width, height, first_newline + 1)
    };

    // Parse compressed size
    let size_newline = data[size_line_start..]
        .iter()
        .position(|&b| b == b'\n')
        .ok_or(LlicError::UnsupportedFormat)?
        + size_line_start;

    let size_line = str::from_utf8(&data[size_line_start..size_newline])
        .map_err(|_| LlicError::UnsupportedFormat)?;

    let compressed_size: usize = size_line
        .trim()
        .parse()
        .map_err(|_| LlicError::UnsupportedFormat)?;

    // Extract compressed data
    let data_start = size_newline + 1;
    if data_start + compressed_size > data.len() {
        return Err(LlicError::UnsupportedFormat);
    }

    let compressed_data = data[data_start..data_start + compressed_size].to_vec();

    Ok((width, height, compressed_data))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_decompression_v3_entropy() {
        // Test with a known format v3 entropy-coded compressed file
        // This is a 4x4 all-zeros image compressed with old llic
        let compressed_data = vec![
            0x03, 0x01, 0x00, // Header: version=3, blocks=1, quality=0 (entropy-coded)
            0x04, 0x00, 0x00, 0x00, // Block size: 4 bytes
            0x00, 0x00, 0x00, 0x00, // Compressed data: 4 bytes of zeros
        ];

        let context = LlicContext::new(4, 4, 4, Some(1)).unwrap();
        let mut output = vec![0u8; 16];

        let (quality, mode) = context
            .decompress_gray8(&compressed_data, &mut output)
            .unwrap();

        // Format v3 entropy-coded returns LosslessEntropy
        assert_eq!(quality, Quality::LosslessEntropy);
        assert_eq!(mode, Mode::Default);

        // All pixels should be 0
        assert!(output.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_gradient_decompression_v3() {
        // Test decompression of format v3 files (created with old C++ llic)
        let compressed_path = "test_data/gradient_4x4_q0.llic";
        if std::path::Path::new(compressed_path).exists() {
            let file_content = std::fs::read(compressed_path).unwrap();

            // Skip the text header "4 4\n79\n"
            let header_end = file_content
                .windows(1)
                .enumerate()
                .filter(|(_, w)| w[0] == b'\n')
                .nth(1)
                .map(|(i, _)| i + 1)
                .unwrap();

            let compressed_data = &file_content[header_end..];

            let context = LlicContext::new(4, 4, 4, Some(1)).unwrap();
            let mut output = vec![0u8; 16];

            let (quality, mode) = context
                .decompress_gray8(compressed_data, &mut output)
                .unwrap();

            // Format v3 lossless returns LosslessEntropy
            assert_eq!(quality, Quality::LosslessEntropy);
            assert_eq!(mode, Mode::Default);

            // Check if we get the expected gradient
            let expected: Vec<u8> = (0..16).collect();
            if output != expected {
                println!("Expected: {:?}", expected);
                println!("Got:      {:?}", output);
                panic!("Gradient decompression failed!");
            }
        }
    }

    #[test]
    fn test_zeros_decompression_v3() {
        // Test decompressing format v3 all-zeros image
        let compressed_path = "test_data/zeros_4x4_q0.llic";
        if std::path::Path::new(compressed_path).exists() {
            let file_content = std::fs::read(compressed_path).unwrap();

            // Skip the text header
            let header_end = file_content
                .windows(1)
                .enumerate()
                .filter(|(_, w)| w[0] == b'\n')
                .nth(1)
                .map(|(i, _)| i + 1)
                .unwrap();

            let compressed_data = &file_content[header_end..];

            let context = LlicContext::new(4, 4, 4, Some(1)).unwrap();
            let mut output = vec![0u8; 16];

            let (quality, mode) = context
                .decompress_gray8(compressed_data, &mut output)
                .unwrap();

            // Format v3 lossless returns LosslessEntropy
            assert_eq!(quality, Quality::LosslessEntropy);
            assert_eq!(mode, Mode::Default);

            // All pixels should be 0
            assert!(
                output.iter().all(|&x| x == 0),
                "Expected all zeros, got: {:?}",
                output
            );
        }
    }

    #[test]
    fn test_roundtrip_tile_based_lossless() {
        // Test round-trip with new tile-based lossless (format v4)
        let original: Vec<u8> = (0..16).collect(); // 4x4 gradient

        let context = LlicContext::new(4, 4, 4, Some(1)).unwrap();
        let mut compressed = vec![0u8; context.compressed_buffer_size()];

        // Compress with tile-based lossless
        let compressed_size = context
            .compress_gray8(&original, Quality::Lossless, Mode::Default, &mut compressed)
            .unwrap();
        compressed.truncate(compressed_size);

        // Check format v5 header
        assert_eq!(compressed[0], 5); // version
        assert_eq!(compressed[2], 1); // tile_based = true

        // Decompress
        let mut output = vec![0u8; 16];
        let (quality, mode) = context.decompress_gray8(&compressed, &mut output).unwrap();

        assert_eq!(quality, Quality::Lossless);
        assert_eq!(mode, Mode::Default);
        assert_eq!(output, original);
    }

    #[test]
    fn test_roundtrip_entropy_lossless() {
        // Test round-trip with legacy entropy-coded lossless (format v5 with tile_based=0)
        let original: Vec<u8> = (0..16).collect(); // 4x4 gradient

        let context = LlicContext::new(4, 4, 4, Some(1)).unwrap();
        let mut compressed = vec![0u8; context.compressed_buffer_size()];

        // Compress with entropy-coded lossless (legacy)
        let compressed_size = context
            .compress_gray8(
                &original,
                Quality::LosslessEntropy,
                Mode::Default,
                &mut compressed,
            )
            .unwrap();
        compressed.truncate(compressed_size);

        // Check format v5 header
        assert_eq!(compressed[0], 5); // version
        assert_eq!(compressed[2], 0); // tile_based = false

        // Decompress
        let mut output = vec![0u8; 16];
        let (quality, mode) = context.decompress_gray8(&compressed, &mut output).unwrap();

        assert_eq!(quality, Quality::LosslessEntropy);
        assert_eq!(mode, Mode::Default);
        assert_eq!(output, original);
    }

    #[test]
    fn test_all_modes_roundtrip() {
        // Test all three modes with tile-based lossless
        let original: Vec<u8> = (0..64).collect(); // 8x8 gradient

        let context = LlicContext::new(8, 8, 8, Some(1)).unwrap();

        for mode in [Mode::Default, Mode::Fast, Mode::Dynamic] {
            let mut compressed = vec![0u8; context.compressed_buffer_size()];

            let compressed_size = context
                .compress_gray8(&original, Quality::Lossless, mode, &mut compressed)
                .unwrap();
            compressed.truncate(compressed_size);

            // Check format v5 header
            assert_eq!(compressed[0], 5, "version should be 5 for mode {mode:?}");
            assert_eq!(
                compressed[2], 1,
                "tile_based should be 1 for mode {:?}",
                mode
            );
            assert_eq!(compressed[3], mode as u8, "mode byte mismatch");

            // Decompress
            let mut output = vec![0u8; 64];
            let (quality, decoded_mode) =
                context.decompress_gray8(&compressed, &mut output).unwrap();

            assert_eq!(
                quality,
                Quality::Lossless,
                "quality mismatch for mode {:?}",
                mode
            );
            assert_eq!(decoded_mode, mode, "mode mismatch");
            assert_eq!(output, original, "pixel data mismatch for mode {:?}", mode);
        }
    }

    #[test]
    fn test_mode_compression_differences() {
        // Test that Fast mode produces larger output than Default/Dynamic (for some images)
        // and that Default/Dynamic compress headers
        let mut image = vec![0u8; 64]; // 8x8
                                       // Create a pattern with some structure (compressible headers)
        for (i, pixel) in image.iter_mut().enumerate().take(64) {
            *pixel = ((i / 4) * 17) as u8;
        }

        let context = LlicContext::new(8, 8, 8, Some(1)).unwrap();

        let mut compressed_default = vec![0u8; context.compressed_buffer_size()];
        let mut compressed_fast = vec![0u8; context.compressed_buffer_size()];

        let size_default = context
            .compress_gray8(
                &image,
                Quality::Lossless,
                Mode::Default,
                &mut compressed_default,
            )
            .unwrap();

        let size_fast = context
            .compress_gray8(&image, Quality::Lossless, Mode::Fast, &mut compressed_fast)
            .unwrap();

        // Fast mode should be >= Default mode (no header compression)
        // The difference may be small or zero for very simple images
        assert!(
            size_fast >= size_default,
            "Fast mode should produce >= size than Default (fast={}, default={})",
            size_fast,
            size_default
        );
    }

    #[test]
    fn test_dynamic_mode_with_edges() {
        // Test dynamic mode with an image that has sharp edges
        // where MED predictor should outperform average predictor
        let width = 64u32;
        let height = 64u32;
        let mut image = vec![0u8; (width * height) as usize];

        // Create vertical stripes - MED predictor should help here
        for y in 0..height {
            for x in 0..width {
                image[(y * width + x) as usize] = if x % 8 < 4 { 0 } else { 255 };
            }
        }

        let context = LlicContext::new(width, height, width, Some(1)).unwrap();

        // Test with all quality levels (the discriminant is the error limit)
        for quality in [
            Quality::Lossless,
            Quality::VeryHigh,
            Quality::High,
            Quality::Medium,
            Quality::Low,
        ] {
            let mut compressed = vec![0u8; context.compressed_buffer_size()];

            let compressed_size = context
                .compress_gray8(&image, quality, Mode::Dynamic, &mut compressed)
                .unwrap();
            compressed.truncate(compressed_size);

            // Decompress
            let mut output = vec![0u8; (width * height) as usize];
            let (decoded_quality, decoded_mode) =
                context.decompress_gray8(&compressed, &mut output).unwrap();

            assert_eq!(decoded_quality, quality);
            assert_eq!(decoded_mode, Mode::Dynamic);

            // Verify pixels are within error bounds (discriminant is the error limit)
            let error_limit = quality as u8;
            for i in 0..image.len() {
                let diff = (image[i] as i32 - output[i] as i32).abs();
                assert!(
                    diff <= error_limit as i32 + 1, // +1 for rounding
                    "Pixel {} error {} exceeds limit {} for quality {:?}",
                    i,
                    diff,
                    error_limit,
                    quality
                );
            }
        }
    }

    fn gradient_image(width: u32, height: u32) -> Vec<u8> {
        let mut img = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                img.push(((x * 7 + y * 13) & 0xFF) as u8);
            }
        }
        img
    }

    /// Wrap-free ramp (requires width + height <= 128): tile min/dist values
    /// are smooth, so tile-header compression engages, unlike with
    /// `gradient_image` whose `& 0xFF` wraparound makes them noisy.
    fn smooth_image(width: u32, height: u32) -> Vec<u8> {
        assert!(width + height <= 128);
        let mut img = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                img.push((x * 2 + y * 2) as u8);
            }
        }
        img
    }

    #[test]
    fn v3_entropy_roundtrip() {
        let width = 32;
        let height = 16;
        let src = gradient_image(width, height);
        let ctx = LlicContext::new(width, height, width, Some(1)).unwrap();
        let mut buf = vec![0u8; ctx.compressed_buffer_size()];
        let n = ctx
            .compress_gray8_versioned(
                &src,
                Quality::LosslessEntropy,
                Mode::Default,
                FormatVersion::V3,
                &mut buf,
            )
            .expect("v3 entropy compress");
        assert_eq!(buf[0], 3, "version byte");
        assert_eq!(buf[1], 1, "num_blocks");
        assert_eq!(buf[2], 0, "type byte for entropy-coded");

        let mut decoded = vec![0u8; (width * height) as usize];
        let (q, m) = ctx
            .decompress_gray8(&buf[..n], &mut decoded)
            .expect("v3 entropy decompress");
        assert_eq!(q, Quality::LosslessEntropy);
        assert_eq!(m, Mode::Default);
        assert_eq!(decoded, src);
    }

    #[test]
    fn v3_tile_lossless_roundtrip() {
        // Smooth enough that tile-header compression pays off: legacy streams
        // recover the mode from bit 7 of the first payload byte (compressed
        // header = Default), so an image whose header stays uncompressed
        // would faithfully decode as Fast — same as the C++ reference.
        let width = 64;
        let height = 64;
        let src = smooth_image(width, height);
        let ctx = LlicContext::new(width, height, width, Some(1)).unwrap();
        let mut buf = vec![0u8; ctx.compressed_buffer_size()];
        let n = ctx
            .compress_gray8_versioned(
                &src,
                Quality::Lossless,
                Mode::Default,
                FormatVersion::V3,
                &mut buf,
            )
            .expect("v3 tile-based compress");
        assert_eq!(buf[0], 3, "version byte");
        assert_eq!(buf[1], 1, "num_blocks");
        assert_ne!(buf[2], 0, "byte 2 must be non-zero (LSB of block_size)");

        let mut decoded = vec![0u8; (width * height) as usize];
        let (q, m) = ctx
            .decompress_gray8(&buf[..n], &mut decoded)
            .expect("v3 tile-based decompress");
        assert_eq!(q, Quality::Lossless);
        assert_eq!(m, Mode::Default);
        assert_eq!(decoded, src);
    }

    #[test]
    fn v3_tile_fast_roundtrip() {
        let width = 32;
        let height = 16;
        let src = gradient_image(width, height);
        let ctx = LlicContext::new(width, height, width, Some(1)).unwrap();
        let mut buf = vec![0u8; ctx.compressed_buffer_size()];
        let n = ctx
            .compress_gray8_versioned(
                &src,
                Quality::Lossless,
                Mode::Fast,
                FormatVersion::V3,
                &mut buf,
            )
            .expect("v3 fast tile-based compress");
        assert_eq!(buf[0], 3, "version byte");

        let mut decoded = vec![0u8; (width * height) as usize];
        let (q, m) = ctx
            .decompress_gray8(&buf[..n], &mut decoded)
            .expect("v3 fast tile-based decompress");
        assert_eq!(q, Quality::Lossless);
        assert_eq!(
            m,
            Mode::Fast,
            "mode recovered from bit 7 of first payload byte"
        );
        assert_eq!(decoded, src);
    }

    #[test]
    fn v3_rejects_dynamic() {
        let width = 16;
        let height = 16;
        let src = gradient_image(width, height);
        let ctx = LlicContext::new(width, height, width, Some(1)).unwrap();
        let mut buf = vec![0u8; ctx.compressed_buffer_size()];
        assert!(matches!(
            ctx.compress_gray8_versioned(
                &src,
                Quality::Lossless,
                Mode::Dynamic,
                FormatVersion::V3,
                &mut buf
            ),
            Err(LlicError::UnsupportedFormat)
        ));
        assert!(matches!(
            ctx.compress_gray8_versioned(
                &src,
                Quality::LosslessEntropy,
                Mode::Dynamic,
                FormatVersion::V3,
                &mut buf
            ),
            Err(LlicError::UnsupportedFormat)
        ));
    }

    #[test]
    fn v3_first_tile_chunk_is_padded_when_its_length_lsb_is_zero() {
        // The tile encoder currently always produces odd payload lengths, so
        // exercise the defensive pad path directly.
        let mut buf = vec![0xAAu8; 600];

        // 256 bytes: the size's LSB would be 0, so one pad byte is appended
        // and counted.
        assert_eq!(pad_v3_tile_chunk(&mut buf, 6, 256).unwrap(), 257);
        assert_eq!(buf[6 + 256], 0, "pad byte");
        assert_eq!(buf[6 + 255], 0xAA, "payload unchanged");

        // 255 bytes: no padding needed.
        let mut buf = vec![0xAAu8; 600];
        assert_eq!(pad_v3_tile_chunk(&mut buf, 6, 255).unwrap(), 255);
        assert!(buf.iter().all(|&b| b == 0xAA));

        // No room for the pad byte.
        let mut buf = vec![0xAAu8; 6 + 256];
        assert!(pad_v3_tile_chunk(&mut buf, 6, 256).is_err());
    }

    #[test]
    fn v3_tile_decoder_ignores_pad_byte() {
        // Decoders locate blocks by offset and read tile streams by need, so a
        // padded stream (block_size + 1, trailing zero byte) must decode
        // identically — this is the premise the pad path relies on.
        let width = 64;
        let height = 64;
        let src = smooth_image(width, height);
        let ctx = LlicContext::new(width, height, width, Some(1)).unwrap();
        let mut buf = vec![0u8; ctx.compressed_buffer_size()];
        let n = ctx
            .compress_gray8_versioned(
                &src,
                Quality::Lossless,
                Mode::Default,
                FormatVersion::V3,
                &mut buf,
            )
            .expect("v3 tile-based compress");

        let mut padded = buf[..n].to_vec();
        padded.push(0);
        let block_size = u32::from_le_bytes([padded[2], padded[3], padded[4], padded[5]]) + 1;
        padded[2..6].copy_from_slice(&block_size.to_le_bytes());

        let mut decoded = vec![0u8; (width * height) as usize];
        let (q, m) = ctx
            .decompress_gray8(&padded, &mut decoded)
            .expect("padded v3 tile-based decompress");
        assert_eq!(q, Quality::Lossless);
        assert_eq!(m, Mode::Default);
        assert_eq!(decoded, src);
    }

    #[test]
    fn compress_gray8_defaults_to_v5() {
        let width = 16;
        let height = 16;
        let src = gradient_image(width, height);
        let ctx = LlicContext::new(width, height, width, Some(1)).unwrap();
        let mut buf = vec![0u8; ctx.compressed_buffer_size()];
        let n = ctx
            .compress_gray8(&src, Quality::LosslessEntropy, Mode::Default, &mut buf)
            .expect("v5 compress");
        assert_eq!(buf[0], 5, "compress_gray8 must emit the current format, v5");
        // Sanity round-trip
        let mut decoded = vec![0u8; (width * height) as usize];
        ctx.decompress_gray8(&buf[..n], &mut decoded).unwrap();
        assert_eq!(decoded, src);
    }

    #[test]
    fn chunk_rows_match_cpp_task_splitters() {
        let split = |version, height, chunks| -> Vec<usize> {
            (0..chunks)
                .map(|i| chunk_rows(version, height, chunks, i))
                .collect()
        };
        assert_eq!(split(2, 76, 3), [28, 28, 20]);
        assert_eq!(split(3, 76, 3), [24, 24, 28]);
        assert_eq!(split(5, 76, 3), [24, 24, 28]);
        // v2 rounds up, so later chunks can shrink to nothing.
        assert_eq!(split(2, 8, 5), [4, 4, 0, 0, 0]);
        assert_eq!(split(2, 44, 4), [12, 12, 12, 8]);
        // v2 can also leave rows unassigned, which is why C++ 2.1 replaced it:
        // 24 rows in 5 chunks covers only 20. Such streams fail the row check.
        assert_eq!(split(2, 24, 5), [4, 4, 4, 4, 4]);
        for height in (4..=200).step_by(4) {
            // Encoders give every chunk at least one row of blocks.
            for chunks in 1..=(height / 4).min(16) {
                let v2: usize = split(2, height, chunks).iter().sum();
                let v3: usize = split(3, height, chunks).iter().sum();
                assert!(v2 <= height, "v2 {height} rows in {chunks}");
                assert_eq!(v3, height, "v3 {height} rows in {chunks}");
            }
        }
    }

    /// The in-place encoder (buffer with room for the worst case) and the
    /// fallback (buffer barely larger than the stream) must agree exactly.
    #[test]
    fn in_place_and_fallback_encodes_match() {
        let (width, height) = (72u32, 40u32);
        let src: Vec<u8> = (0..width * height)
            .map(|i| ((i * 7919) % 251 + (i % width)) as u8)
            .collect();
        for (version, chunks) in [FormatVersion::V3, FormatVersion::V4, FormatVersion::V5]
            .into_iter()
            .flat_map(|version| [1, 3].map(|chunks| (version, chunks)))
        {
            let ctx = LlicContext::new(width, height, width, Some(chunks)).unwrap();
            for quality in [Quality::Lossless, Quality::Medium, Quality::LosslessEntropy] {
                for mode in [Mode::Default, Mode::Fast, Mode::Dynamic] {
                    if version == FormatVersion::V3 && mode == Mode::Dynamic {
                        continue;
                    }
                    let mut roomy = vec![0u8; ctx.compressed_buffer_size()];
                    let size = ctx
                        .compress_gray8_versioned(&src, quality, mode, version, &mut roomy)
                        .unwrap();
                    let mut tight = vec![0u8; size];
                    let tight_size = ctx
                        .compress_gray8_versioned(&src, quality, mode, version, &mut tight)
                        .unwrap();
                    let case = format!("{version:?} {quality:?} {mode:?} {chunks} chunks");
                    assert_eq!(tight_size, size, "{case}");
                    assert_eq!(tight, roomy[..size], "{case}");
                    let mut too_small = vec![0u8; size - 1];
                    assert!(
                        ctx.compress_gray8_versioned(&src, quality, mode, version, &mut too_small)
                            .is_err(),
                        "{case}"
                    );
                }
            }
        }
    }

    /// Random noise is the entropy coder's worst case (13-bit escapes); every
    /// quality and mode must fit in `compressed_buffer_size`.
    #[test]
    fn compressed_buffer_size_fits_noise() {
        // Large enough that a fixed margin cannot hide a per-pixel shortfall.
        let (width, height) = (512u32, 128u32);
        let mut state = 0x1234_5678_u32;
        let noise: Vec<u8> = (0..width * height)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state >> 24) as u8
            })
            .collect();
        for (version, chunks) in [FormatVersion::V3, FormatVersion::V4, FormatVersion::V5]
            .into_iter()
            .flat_map(|version| [1, 3, MAX_CHUNKS].map(|chunks| (version, chunks)))
        {
            let ctx = LlicContext::new(width, height, width, Some(chunks)).unwrap();
            let mut buf = vec![0u8; ctx.compressed_buffer_size()];
            for quality in [
                Quality::Lossless,
                Quality::VeryLow,
                Quality::LosslessEntropy,
            ] {
                for mode in [Mode::Default, Mode::Fast, Mode::Dynamic] {
                    if version == FormatVersion::V3 && mode == Mode::Dynamic {
                        continue;
                    }
                    ctx.compress_gray8_versioned(&noise, quality, mode, version, &mut buf)
                        .unwrap_or_else(|e| {
                            panic!("{version:?} {quality:?} {mode:?} {chunks} chunks: {e}")
                        });
                }
            }
        }
    }

    #[test]
    fn new_checks_the_chunk_count() {
        assert_eq!(LlicContext::new(64, 64, 64, None).unwrap().num_chunks(), 1);
        assert_eq!(
            LlicContext::new(64, 64, 64, Some(16)).unwrap().num_chunks(),
            16
        );
        // At most one chunk per four rows.
        assert_eq!(
            LlicContext::new(64, 8, 64, Some(2)).unwrap().num_chunks(),
            2
        );
        for (height, chunks) in [(64, 0), (128, MAX_CHUNKS + 1), (8, 3)] {
            assert!(
                matches!(
                    LlicContext::new(64, height, 64, Some(chunks)),
                    Err(LlicError::InvalidArgument)
                ),
                "{chunks} chunks of {height} rows"
            );
        }
    }

    /// Each chunk is coded as its strip of rows would be on its own (the C++
    /// layout: one encoder per chunk, rows split by `chunk_rows`), and the
    /// stream decodes to the strips' decoded rows.
    #[test]
    fn chunks_are_coded_independently() {
        // 76 rows split 24, 24, 28 in three chunks; stride wider than a row.
        let (width, height, stride) = (72u32, 76u32, 80u32);
        let src: Vec<u8> = (0..stride * height)
            .map(|i| ((i * 7919) % 251 + (i % stride) * 2) as u8)
            .collect();
        let stride_len = stride as usize;
        for version in [FormatVersion::V3, FormatVersion::V4, FormatVersion::V5] {
            let fields = |quality| match version {
                FormatVersion::V4 | FormatVersion::V5 => 4,
                FormatVersion::V3 if quality == Quality::LosslessEntropy => 3,
                FormatVersion::V3 => 2,
            };
            for quality in [Quality::Lossless, Quality::High, Quality::LosslessEntropy] {
                for mode in [Mode::Default, Mode::Fast, Mode::Dynamic] {
                    if version == FormatVersion::V3 && mode == Mode::Dynamic {
                        continue;
                    }
                    for chunks in [2usize, 3, 5, 19] {
                        let case = format!("{version:?} {quality:?} {mode:?} {chunks} chunks");
                        let ctx =
                            LlicContext::new(width, height, stride, Some(chunks.min(16))).unwrap();
                        let chunks = ctx.num_chunks();
                        let mut buf = vec![0u8; ctx.compressed_buffer_size()];
                        let size = ctx
                            .compress_gray8_versioned(&src, quality, mode, version, &mut buf)
                            .unwrap();
                        let stream = &buf[..size];
                        assert_eq!(stream[1] as usize, chunks, "{case}");

                        let table = fields(quality);
                        let mut pos = table + 4 * chunks;
                        let mut row = 0usize;
                        let mut expected = vec![0u8; src.len()];
                        for chunk in 0..chunks {
                            let rows = chunk_rows(version as u8, height as usize, chunks, chunk);
                            let at = table + 4 * chunk;
                            let chunk_size =
                                u32::from_le_bytes(stream[at..at + 4].try_into().unwrap()) as usize;

                            let strip = &src[row * stride_len..];
                            let alone =
                                LlicContext::new(width, rows as u32, stride, Some(1)).unwrap();
                            let mut alone_buf = vec![0u8; alone.compressed_buffer_size()];
                            let alone_size = alone
                                .compress_gray8_versioned(
                                    strip,
                                    quality,
                                    mode,
                                    version,
                                    &mut alone_buf,
                                )
                                .unwrap();
                            let alone_stream = &alone_buf[..alone_size];
                            assert_eq!(
                                stream[pos..pos + chunk_size],
                                alone_stream[table + 4..],
                                "{case}: chunk {chunk}"
                            );
                            alone
                                .decompress_gray8(alone_stream, &mut expected[row * stride_len..])
                                .unwrap();
                            pos += chunk_size;
                            row += rows;
                        }
                        assert_eq!(pos, size, "{case}");
                        assert_eq!(row, height as usize, "{case}");

                        let mut decoded = vec![0u8; src.len()];
                        let (decoded_quality, _) =
                            ctx.decompress_gray8(stream, &mut decoded).unwrap();
                        assert_eq!(decoded_quality, quality, "{case}");
                        for y in 0..height as usize {
                            let row = y * stride_len..y * stride_len + width as usize;
                            assert_eq!(
                                decoded[row.clone()],
                                expected[row.clone()],
                                "{case}: row {y}"
                            );
                            if quality != Quality::High {
                                assert_eq!(decoded[row.clone()], src[row], "{case}: row {y}");
                            }
                        }
                    }
                }
            }
        }
    }
}
