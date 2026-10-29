use super::tables::{get_compress_table_entry, DECODE_TABLE_13, DECOMPRESS_TABLE};
use crate::{LlicError, Result};

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wavefront_simd128;
#[cfg(target_arch = "x86_64")]
mod wavefront_sse;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use wavefront_simd128 as wavefront;
/// The wavefront for this target (same interface in both modules).
#[cfg(target_arch = "x86_64")]
use wavefront_sse as wavefront;

/// MED (Median Edge Detector) predictor used in JPEG-LS and Dynamic mode.
///
/// `left + top - clamp(top_left, min, max)` gives all three cases without
/// branches, which mispredict on noisy rows (the SIMD code computes the same).
#[inline(always)]
fn med_predictor(left: u8, top: u8, top_left: u8) -> u8 {
    let (min_ab, max_ab) = (left.min(top), left.max(top));
    left.wrapping_add(top)
        .wrapping_sub(top_left.max(min_ab).min(max_ab))
}

/// Align a value up to the next multiple of 4.
#[inline]
fn align_to_4(value: usize) -> usize {
    (value + 3) & !3
}

/// Decompresses data encoded with the u8v1 entropy coder.
pub fn decompress(
    src_data: &[u8],
    width: u32,
    height: u32,
    bytes_per_line: u32,
    dst_image: &mut [u8],
) -> Result<()> {
    decode_image(src_data, None, width, height, bytes_per_line, dst_image)
}

/// Decompresses data encoded with the u8v1 entropy coder with dynamic predictor.
///
/// Dynamic mode uses a per-row predictor bitmap to choose between average and MED predictors.
///
/// # Arguments
/// * `src_data` - Compressed data with predictor bitmap at the start
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `bytes_per_line` - Stride between rows in output
/// * `dst_image` - Output buffer for decompressed pixels
pub fn decompress_dynamic(
    src_data: &[u8],
    width: u32,
    height: u32,
    bytes_per_line: u32,
    dst_image: &mut [u8],
) -> Result<()> {
    let predictor_storage_bytes = align_to_4((height as usize).div_ceil(8));
    if src_data.len() < predictor_storage_bytes {
        return Err(LlicError::InvalidData);
    }
    let (predictor_bits, stream) = src_data.split_at(predictor_storage_bytes);
    decode_image(
        stream,
        Some(predictor_bits),
        width,
        height,
        bytes_per_line,
        dst_image,
    )
}

/// Decompresses two independent streams of the same width at once, one into
/// each of `dst_images`: [`decompress`] (or [`decompress_dynamic`]) of each,
/// with the two symbol decoders interleaved so that their table lookups
/// overlap instead of each waiting on the one before.
pub fn decompress_pair(
    src_data: [&[u8]; 2],
    dynamic: bool,
    width: u32,
    heights: [u32; 2],
    bytes_per_line: u32,
    dst_images: [&mut [u8]; 2],
) -> Result<()> {
    let mut images = Vec::with_capacity(2);
    for (src, height) in src_data.into_iter().zip(heights) {
        let (predictor_bits, stream) = if dynamic {
            let predictor_storage_bytes = align_to_4((height as usize).div_ceil(8));
            if src.len() < predictor_storage_bytes {
                return Err(LlicError::InvalidData);
            }
            let (bits, stream) = src.split_at(predictor_storage_bytes);
            (Some(bits), stream)
        } else {
            (None, src)
        };
        images.push(ImageDecoder::new(stream, predictor_bits, width, height, bytes_per_line)?);
    }
    let [a, b]: [ImageDecoder; 2] = images.try_into().ok().expect("two images");
    let [dst_a, dst_b] = dst_images;
    a.check_dst(dst_a)?;
    b.check_dst(dst_b)?;
    decode_image_pair([a, b], [dst_a, dst_b])
}

/// Whether row `y` uses the MED predictor (dynamic mode) instead of the average.
#[inline]
fn row_uses_med(predictor_bits: Option<&[u8]>, y: usize) -> bool {
    predictor_bits.is_some_and(|bits| (bits[y / 8] >> (y % 8)) & 1 != 0)
}

/// Reconstructs `row` from `residuals`: pixel 0 predicts from `top` only, the
/// rest from the left and top pixels (rounded-down average, or MED).
#[inline(always)]
fn reconstruct_row(top: &[u8], residuals: &[u8], row: &mut [u8], width: usize, use_med: bool) {
    let (top, residuals, row) = (&top[..width], &residuals[..width], &mut row[..width]);
    row[0] = residuals[0].wrapping_add(top[0]);
    if use_med {
        for x in 1..width {
            row[x] = residuals[x].wrapping_add(med_predictor(row[x - 1], top[x], top[x - 1]));
        }
    } else {
        for x in 1..width {
            let average = ((u16::from(row[x - 1]) + u16::from(top[x])) >> 1) as u8;
            row[x] = residuals[x].wrapping_add(average);
        }
    }
}

/// Like [`reconstruct_row`] for pixels `[x0, x1)`, with `row` holding the
/// residuals on entry (pixel `x` is replaced after pixel `x - 1`).
#[cfg(any(
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[inline(always)]
fn reconstruct_row_in_place(top: &[u8], row: &mut [u8], x0: usize, x1: usize, use_med: bool) {
    let mut x = x0;
    if x == 0 && x < x1 {
        row[0] = row[0].wrapping_add(top[0]);
        x = 1;
    }
    if use_med {
        for x in x..x1 {
            row[x] = row[x].wrapping_add(med_predictor(row[x - 1], top[x], top[x - 1]));
        }
    } else {
        for x in x..x1 {
            let average = ((u16::from(row[x - 1]) + u16::from(top[x])) >> 1) as u8;
            row[x] = row[x].wrapping_add(average);
        }
    }
}

#[cfg(any(
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
/// Reconstructs pixels `span(k)` of each wavefront row k in order, in place;
/// `rows` holds the rows `width` apart and `prev` is the row above them.
#[inline(always)]
fn reconstruct_parts(
    prev: &[u8],
    rows: &mut [u8],
    width: usize,
    uses_med: impl Fn(usize) -> bool,
    span: impl Fn(usize) -> (usize, usize),
) {
    for k in 0..WAVE_ROWS {
        let (x0, x1) = span(k);
        if k == 0 {
            reconstruct_row_in_place(prev, &mut rows[..width], x0, x1, uses_med(0));
        } else {
            let (above, row) = rows[(k - 1) * width..(k + 1) * width].split_at_mut(width);
            reconstruct_row_in_place(above, row, x0, x1, uses_med(k));
        }
    }
}

/// Decodes an image: the first row by horizontal delta, the others from the
/// left and top pixels (dynamic mode chooses per row from `predictor_bits`).
fn decode_image(
    stream: &[u8],
    predictor_bits: Option<&[u8]>,
    width: u32,
    height: u32,
    bytes_per_line: u32,
    dst_image: &mut [u8],
) -> Result<()> {
    let mut image = ImageDecoder::new(stream, predictor_bits, width, height, bytes_per_line)?;
    image.check_dst(dst_image)?;
    while !image.done() {
        image.step(dst_image)?;
    }
    Ok(())
}

/// Decodes two images in lockstep, interleaving their symbol decoders, and
/// finishes the taller one alone.
fn decode_image_pair(images: [ImageDecoder<'_>; 2], dst_images: [&mut [u8]; 2]) -> Result<()> {
    let [mut a, mut b] = images;
    let [dst_a, dst_b] = dst_images;
    while !a.done() && !b.done() {
        let (count_a, count_b) = (a.next_count(), b.next_count());
        let (elem_a, elem_b) = (a.carry_over(), b.carry_over());
        let (elem_a, elem_b) = SymbolDecoder::decode_fast_pair(
            [&mut a.decoder, &mut b.decoder],
            [&mut a.residuals, &mut b.residuals],
            [elem_a, elem_b],
            [count_a, count_b],
        );
        a.filled = a.decoder.finish(&mut a.residuals, elem_a, count_a)?;
        b.filled = b.decoder.finish(&mut b.residuals, elem_b, count_b)?;
        a.reconstruct(dst_a, count_a);
        b.reconstruct(dst_b, count_b);
    }
    while !a.done() {
        a.step(dst_a)?;
    }
    while !b.done() {
        b.step(dst_b)?;
    }
    Ok(())
}

/// One image being decoded: its symbol decoder, the residuals of the rows in
/// flight, and the next row to reconstruct. Each step decodes the first row,
/// a wavefront of [`WAVE_ROWS`] rows, or (after the last wavefront) one row.
struct ImageDecoder<'a> {
    decoder: SymbolDecoder<'a>,
    predictor_bits: Option<&'a [u8]>,
    width: usize,
    height: usize,
    stride: usize,
    wave: bool,
    /// Residuals for a wavefront of rows, plus room for a symbol pair beyond.
    residuals: Vec<u8>,
    /// Symbols in `residuals`, and how many of them the last step used.
    filled: usize,
    used: usize,
    /// The next row to reconstruct.
    y: usize,
}

impl<'a> ImageDecoder<'a> {
    fn new(
        stream: &'a [u8],
        predictor_bits: Option<&'a [u8]>,
        width: u32,
        height: u32,
        bytes_per_line: u32,
    ) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(LlicError::InvalidArgument);
        }
        let (width, height, stride) = (width as usize, height as usize, bytes_per_line as usize);
        if stride < width {
            return Err(LlicError::InvalidArgument);
        }

        #[cfg(any(
            target_arch = "x86_64",
            all(target_arch = "wasm32", target_feature = "simd128")
        ))]
        let wave = width >= 2 * WAVE_ROWS && wavefront::available();
        #[cfg(not(any(
            target_arch = "x86_64",
            all(target_arch = "wasm32", target_feature = "simd128")
        )))]
        let wave = false;
        Ok(Self {
            decoder: SymbolDecoder::new(stream),
            predictor_bits,
            width,
            height,
            stride,
            wave,
            residuals: vec![0u8; if wave { WAVE_ROWS } else { 1 } * width + 256],
            filled: 0,
            used: 0,
            y: 0,
        })
    }

    fn check_dst(&self, dst_image: &[u8]) -> Result<()> {
        if dst_image.len() < (self.height - 1) * self.stride + self.width {
            return Err(LlicError::InvalidArgument);
        }
        Ok(())
    }

    #[inline(always)]
    fn done(&self) -> bool {
        self.y >= self.height
    }

    /// Residuals the next step needs.
    #[inline(always)]
    fn next_count(&self) -> usize {
        if self.y > 0 && self.wave && self.y + WAVE_ROWS <= self.height {
            WAVE_ROWS * self.width
        } else {
            self.width
        }
    }

    /// Moves the symbols decoded past the last step's residuals to the front
    /// and returns how many there are.
    #[inline(always)]
    fn carry_over(&mut self) -> usize {
        if self.filled > self.used {
            self.residuals.copy_within(self.used..self.filled, 0);
        }
        self.filled - self.used
    }

    /// Decodes and reconstructs the next step's rows.
    #[inline(always)]
    fn step(&mut self, dst_image: &mut [u8]) -> Result<()> {
        let count = self.next_count();
        let elem = self.carry_over();
        self.filled = self.decoder.finish(&mut self.residuals, elem, count)?;
        self.reconstruct(dst_image, count);
        Ok(())
    }

    /// Reconstructs the step's rows from the first `count` residuals.
    #[inline(always)]
    fn reconstruct(&mut self, dst_image: &mut [u8], count: usize) {
        let (width, stride, y) = (self.width, self.stride, self.y);
        let residuals = &mut self.residuals;
        if y == 0 {
            dst_image[0] = residuals[0];
            for x in 1..width {
                dst_image[x] = residuals[x].wrapping_add(dst_image[x - 1]);
            }
        } else if count > width {
            #[cfg(any(
                target_arch = "x86_64",
                all(target_arch = "wasm32", target_feature = "simd128")
            ))]
            {
                let med_rows = (0..WAVE_ROWS)
                    .filter(|&k| row_uses_med(self.predictor_bits, y + k))
                    .fold(0u32, |rows, k| rows | (1 << k));
                let (done, rest) = dst_image.split_at_mut(y * stride);
                let prev = &done[(y - 1) * stride..(y - 1) * stride + width];
                // SAFETY: the CPU supports the wavefront's instructions (`available`).
                unsafe { wavefront::reconstruct(prev, &mut residuals[..count], width, med_rows) };
                // Sixteen interleaved row streams write back slowly, so the rows are
                // reconstructed in the buffer and copied one after another.
                for (k, row) in residuals[..count].chunks_exact(width).enumerate() {
                    rest[k * stride..k * stride + width].copy_from_slice(row);
                }
            }
        } else {
            let (done, rest) = dst_image.split_at_mut(y * stride);
            let top = &done[(y - 1) * stride..];
            reconstruct_row(top, residuals, rest, width, row_uses_med(self.predictor_bits, y));
        }
        self.used = count;
        self.y += count / width;
    }
}

/// Rows reconstructed together by the wavefront; lane k holds row k.
const WAVE_ROWS: usize = 16;

/// Loads 64 stream bits: four 16-bit little-endian words, the first one most
/// significant.
#[inline(always)]
fn stream_bits(bytes: [u8; 8]) -> u64 {
    let words = u64::from_le_bytes(bytes).rotate_left(32);
    ((words << 16) & 0xFFFF_0000_FFFF_0000) | ((words >> 16) & 0x0000_FFFF_0000_FFFF)
}

/// `container << (entry & 63)`. On x86-64 a bare `SHL`, which masks the count
/// itself: LLVM shares the `entry & 63` with the bit count update and puts the
/// AND on the dependency chain from one table lookup to the next (about 7% of
/// entropy decoding).
#[inline(always)]
fn shift_by_entry(container: u64, entry: u32) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        let mut container = container;
        // SAFETY: only shifts a register.
        unsafe {
            std::arch::asm!(
                "shl {container}, cl",
                container = inout(reg) container,
                in("ecx") entry,
                options(pure, nomem, nostack),
            );
        }
        container
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        container << (entry & 63)
    }
}

/// Symbol decoder after the C++ 4.1 `decompressFast`: a 64-bit container
/// refilled with whole 16-bit words, and three lookups per refill in
/// [`DECODE_TABLE_13`], each decoding up to three symbols.
struct SymbolDecoder<'a> {
    src: &'a [u8],
    /// Bytes of `src` loaded into the container.
    pos: usize,
    /// Stream bits, first bit in bit 63. `num_bits` are loaded; the bits
    /// below are zero or bits a later refill ORs in again unchanged.
    container: u64,
    num_bits: u32,
}

impl<'a> SymbolDecoder<'a> {
    fn new(src: &'a [u8]) -> Self {
        Self {
            src,
            pos: 0,
            container: 0,
            num_bits: 0,
        }
    }

    /// Decodes into `buf` from `elem` while three full lookups fit before
    /// `count` and eight stream bytes remain. Returns the new `elem`.
    #[inline(always)]
    fn decode_fast(&mut self, buf: &mut [u8], mut elem: usize, count: usize) -> usize {
        assert!(buf.len() >= count + 16);
        let mut state = FastState::load(self);
        while elem + 9 <= count && state.has_word(self.src) {
            state.refill(self.src);
            for _ in 0..3 {
                // SAFETY: elem + 9 <= count before the three lookups, each
                // adds at most 3, and buf holds count + 16 bytes.
                elem = unsafe { state.lookup(buf, elem) };
            }
        }
        state.store(self);
        elem
    }

    /// [`decode_fast`](Self::decode_fast) for two decoders at once: each
    /// iteration refills both and alternates their lookups, so the two
    /// dependency chains overlap. Stops when either one would stop.
    #[inline(always)]
    fn decode_fast_pair(
        decoders: [&mut Self; 2],
        bufs: [&mut [u8]; 2],
        elems: [usize; 2],
        counts: [usize; 2],
    ) -> (usize, usize) {
        let [a, b] = decoders;
        let [buf_a, buf_b] = bufs;
        let ([mut elem_a, mut elem_b], [count_a, count_b]) = (elems, counts);
        assert!(buf_a.len() >= count_a + 16 && buf_b.len() >= count_b + 16);
        let (mut state_a, mut state_b) = (FastState::load(a), FastState::load(b));
        while elem_a + 9 <= count_a
            && elem_b + 9 <= count_b
            && state_a.has_word(a.src)
            && state_b.has_word(b.src)
        {
            state_a.refill(a.src);
            state_b.refill(b.src);
            for _ in 0..3 {
                // SAFETY: as in `decode_fast`, for each buffer.
                unsafe {
                    elem_a = state_a.lookup(buf_a, elem_a);
                    elem_b = state_b.lookup(buf_b, elem_b);
                }
            }
        }
        state_a.store(a);
        state_b.store(b);
        (elem_a, elem_b)
    }

    /// Finishes decoding into `buf` until it holds at least `count` symbols,
    /// from `elem`, and returns how many it holds (an entry can end up to two
    /// past `count`).
    #[inline(always)]
    fn finish(&mut self, buf: &mut [u8], mut elem: usize, count: usize) -> Result<usize> {
        elem = self.decode_fast(buf, elem, count);
        while elem < count {
            elem += self.decode_checked(buf, elem)?;
        }
        Ok(elem)
    }

    /// Decodes one table entry near the end of the stream, checking that its
    /// bits are there; when only the first symbol's are, decodes that alone.
    /// Returns the symbols decoded, or `InvalidData` for a truncated stream.
    #[inline(always)]
    fn decode_checked(&mut self, buf: &mut [u8], elem: usize) -> Result<usize> {
        if self.num_bits < 16 {
            let left = self.src.len() - self.pos;
            if left >= 4 {
                let bytes: [u8; 4] = self.src[self.pos..self.pos + 4]
                    .try_into()
                    .expect("4 bytes");
                let word = u32::from_le_bytes(bytes).rotate_left(16);
                self.container |= u64::from(word) << (32 - self.num_bits);
                (self.num_bits, self.pos) = (self.num_bits + 32, self.pos + 4);
            } else if left >= 2 {
                let word = u16::from_le_bytes([self.src[self.pos], self.src[self.pos + 1]]);
                self.container |= u64::from(word) << (48 - self.num_bits);
                (self.num_bits, self.pos) = (self.num_bits + 16, self.pos + 2);
            }
        }
        let entry = DECODE_TABLE_13.0[(self.container >> 51) as usize];
        let (bits, count) = (entry & 63, (entry >> 6) & 3);
        if bits <= self.num_bits {
            self.container <<= bits;
            self.num_bits -= bits;
            buf[elem..elem + 4].copy_from_slice(&(entry >> 8).to_le_bytes());
            return Ok(count as usize);
        }
        // Escapes are always alone; otherwise the first symbol has a short
        // code, whose length the pair table gives.
        #[allow(clippy::cast_possible_truncation)]
        let symbol = (entry >> 8) as u8;
        let first_bits = if count == 1 {
            bits
        } else {
            (get_compress_table_entry(usize::from(symbol)) & 0x3F) - 2
        };
        if first_bits > self.num_bits {
            return Err(LlicError::InvalidData);
        }
        self.container <<= first_bits;
        self.num_bits -= first_bits;
        buf[elem] = symbol;
        Ok(1)
    }
}

/// The fast loop's decoder state, kept in registers: the stream position,
/// the 64-bit container and its bit count.
#[derive(Clone, Copy)]
struct FastState {
    pos: usize,
    container: u64,
    num_bits: u32,
    /// Only the low six bits count: whole entries are subtracted.
    shift: u32,
}

impl FastState {
    #[inline(always)]
    fn load(decoder: &SymbolDecoder) -> Self {
        Self {
            pos: decoder.pos,
            container: decoder.container,
            num_bits: decoder.num_bits,
            shift: decoder.num_bits,
        }
    }

    #[inline(always)]
    fn store(self, decoder: &mut SymbolDecoder) {
        (decoder.pos, decoder.container, decoder.num_bits) =
            (self.pos, self.container, self.num_bits);
    }

    /// Whether eight stream bytes remain for [`refill`](Self::refill).
    #[inline(always)]
    fn has_word(&self, src: &[u8]) -> bool {
        src.len() - self.pos >= 8
    }

    /// Loads the whole words that fit: at least 48 bits are loaded
    /// afterwards, enough for three 13-bit lookups.
    #[inline(always)]
    fn refill(&mut self, src: &[u8]) {
        let bytes: [u8; 8] = src[self.pos..self.pos + 8].try_into().expect("8 bytes");
        self.container |= stream_bits(bytes) >> (self.shift & 63);
        let words = (63 - self.num_bits) >> 4;
        self.pos += (words * 2) as usize;
        self.num_bits += words * 16;
        self.shift = self.num_bits;
    }

    /// Decodes one table entry (up to three symbols) into `buf` at `elem` and
    /// returns the new `elem`.
    ///
    /// # Safety
    /// `buf` must hold `elem + 4` bytes.
    #[inline(always)]
    unsafe fn lookup(&mut self, buf: &mut [u8], elem: usize) -> usize {
        let entry = DECODE_TABLE_13.0[(self.container >> 51) as usize];
        self.container = shift_by_entry(self.container, entry);
        self.shift = self.shift.wrapping_sub(entry);
        self.num_bits -= entry & 63;
        buf.as_mut_ptr()
            .add(elem)
            .cast::<[u8; 4]>()
            .write_unaligned((entry >> 8).to_le_bytes());
        elem + ((entry >> 6) & 3) as usize
    }
}

pub fn decompress_with_debug(
    src_data: &[u8],
    width: u32,
    height: u32,
    bytes_per_line: u32,
    dst_image: &mut [u8],
    debug: bool,
) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(LlicError::InvalidArgument);
    }

    let mut decoder = Decoder::new(src_data);
    decoder.debug = debug;
    let mut row_buffer = vec![0u8; width as usize + 256];

    // CRITICAL: Start with 0 elements, not width!
    // The C++ code initializes numFilledElements to inWidth, but that's for the first call
    // We need to decode the first row from scratch
    let mut num_filled_elements = 0;

    if debug {
        println!("\n=== Starting decompression ===");
        println!(
            "Width: {}, Height: {}, Bytes per line: {}",
            width, height, bytes_per_line
        );
    }

    // Decompress first row
    num_filled_elements = decoder.decompress_row_with_debug(
        &mut row_buffer,
        num_filled_elements,
        width as usize,
        debug,
        0,
    )?;

    if debug {
        println!("\nRow 0 raw symbols: {:?}", &row_buffer[0..width as usize]);
        println!("After row 0: {} elements in buffer", num_filled_elements);
    }

    // First row: Horizontal delta only
    dst_image[0] = row_buffer[0];
    for x in 1..width {
        dst_image[x as usize] = row_buffer[x as usize].wrapping_add(dst_image[x as usize - 1]);
    }

    if debug {
        println!("Row 0 after delta: {:?}", &dst_image[0..width as usize]);
        println!(
            "Row 0 final bytes: {:02X?}",
            &dst_image[0..width.min(20) as usize]
        );
    }

    // Decompress remaining rows (combined predictor)
    for y in 1..height {
        let row_offset = (y * bytes_per_line) as usize;
        let prev_row_offset = ((y - 1) * bytes_per_line) as usize;

        if debug && (2..=5).contains(&y) {
            println!("\n--- Processing row {} ---", y);
            println!(
                "Carryover buffer state before decode: {} elements",
                num_filled_elements
            );
            if num_filled_elements > width as usize {
                println!(
                    "Carryover symbols: {:?}",
                    &row_buffer[width as usize..num_filled_elements]
                );
            }
        }

        // Decompress row with carryover
        num_filled_elements = decoder.decompress_row_with_debug(
            &mut row_buffer,
            num_filled_elements,
            width as usize,
            debug && (2..=5).contains(&y),
            y,
        )?;

        if debug && (2..=5).contains(&y) {
            println!(
                "Row {} raw symbols: {:?}",
                y,
                &row_buffer[0..width.min(20) as usize]
            );
            println!(
                "After row {}: {} elements in buffer",
                y, num_filled_elements
            );
        }

        // Apply prediction and reconstruction
        dst_image[row_offset] = row_buffer[0].wrapping_add(dst_image[prev_row_offset]);

        if debug && (2..=5).contains(&y) {
            println!(
                "First pixel: delta={}, top={}, result={}",
                row_buffer[0], dst_image[prev_row_offset], dst_image[row_offset]
            );
        }

        for x in 1..width {
            let left = dst_image[row_offset + x as usize - 1];
            let top = dst_image[prev_row_offset + x as usize];
            let avg = ((left as i32 + top as i32) / 2) as u8;
            let delta = row_buffer[x as usize];
            dst_image[row_offset + x as usize] = delta.wrapping_add(avg);

            if debug && (2..=5).contains(&y) && x < 20 {
                println!(
                    "  Pixel[{},{}]: left={}, top={}, avg={}, delta={}, result={}",
                    y,
                    x,
                    left,
                    top,
                    avg,
                    delta,
                    dst_image[row_offset + x as usize]
                );
            }
        }

        if debug && (2..=5).contains(&y) {
            println!(
                "Row {} reconstructed: {:02X?}",
                y,
                &dst_image[row_offset..row_offset + width.min(20) as usize]
            );

            // Check for corruption at byte 271 (row 4, column 15)
            if y == 4 {
                let byte_pos = row_offset + 15;
                println!("\nByte 271 check (row 4, col 15):");
                println!("  Absolute position: {}", byte_pos);
                println!("  Value: 0x{:02X}", dst_image[byte_pos]);
                if byte_pos > 0 {
                    println!("  Previous byte (270): 0x{:02X}", dst_image[byte_pos - 1]);
                }
            }
        }
    }

    Ok(())
}

struct Decoder<'a> {
    data: &'a [u8],
    pos: usize,
    bit_container: u32,
    bits_available: u32,
    debug: bool,
}

impl<'a> Decoder<'a> {
    fn new(data: &'a [u8]) -> Self {
        let mut decoder = Self {
            data,
            pos: 0,
            bit_container: 0,
            bits_available: 0,
            debug: false,
        };
        // Pre-fill the bit container
        decoder.refill();
        decoder
    }

    fn refill(&mut self) {
        // The C++ version reads 16-bit values
        while self.bits_available < 16 && self.pos + 1 < self.data.len() {
            // Read 16 bits in little-endian order
            let word = u16::from_le_bytes([self.data[self.pos], self.data[self.pos + 1]]);
            self.bit_container |= (word as u32) << (16 - self.bits_available);
            self.bits_available += 16;
            self.pos += 2;

            if self.debug {
                println!("        refill: read word 0x{:04X} at pos {}, bit_container=0x{:08X}, bits_available={}", 
                    word, self.pos - 2, self.bit_container, self.bits_available);
            }
        }

        if self.debug && self.pos + 1 >= self.data.len() {
            println!(
                "        refill: reached end of data at pos {}, data.len()={}",
                self.pos,
                self.data.len()
            );
        }
    }

    fn consume_bits(&mut self, bits: u32) {
        self.bit_container <<= bits;
        self.bits_available = self.bits_available.saturating_sub(bits);
        self.refill();
    }

    fn decode_sequence(&mut self, out: &mut [u8]) -> Result<usize> {
        self.refill();

        // Debug: show bit container state
        if self.debug {
            println!(
                "      decode_sequence: bit_container=0x{:08X}, bits_available={}",
                self.bit_container, self.bits_available
            );
        }

        // Get top 12 bits for table lookup
        let index = (self.bit_container >> 20) as usize;

        if self.debug {
            println!("      Table lookup index: 0x{:03X} ({})", index, index);
        }

        // Special case: 13-bit codes
        if index >= 0xF80 {
            let symbol = ((self.bit_container >> 19) & 0xFF) as u8;
            if self.debug {
                println!("      13-bit code: symbol={}", symbol);
            }
            out[0] = symbol;
            self.consume_bits(13);
            return Ok(1);
        }

        // Normal case: use lookup table
        if index >= DECOMPRESS_TABLE.len() {
            return Err(LlicError::InvalidData);
        }

        let entry = DECOMPRESS_TABLE[index];

        if self.debug {
            println!(
                "      Table entry: bits={}, num_symbols={}, symbols=[{}, {}]",
                entry.bits, entry.num_symbols, entry.symbols[0], entry.symbols[1]
            );
        }

        self.consume_bits(entry.bits as u32);

        // Copy decoded symbols (always write both, even if num_symbols == 1)
        out[0] = entry.symbols[0];
        out[1] = entry.symbols[1];

        Ok(entry.num_symbols as usize)
    }

    #[allow(dead_code)]
    fn decompress_row(&mut self, buf: &mut [u8], num_elems: usize, width: usize) -> Result<usize> {
        self.decompress_row_with_debug(buf, num_elems, width, false, 0)
    }

    fn decompress_row_with_debug(
        &mut self,
        buf: &mut [u8],
        num_elems: usize,
        width: usize,
        debug: bool,
        _row_num: u32,
    ) -> Result<usize> {
        let mut elem = num_elems;

        if debug {
            println!(
                "  decompress_row: start with {} elements, width={}",
                num_elems, width
            );
        }

        // Handle carryover from previous row
        if elem > width {
            // Copy the extra elements to the beginning of the buffer
            if debug {
                println!(
                    "  Copying {} carryover elements from position {} to 0",
                    elem - width,
                    width
                );
                println!("  Carryover data: {:?}", &buf[width..elem]);
            }
            buf.copy_within(width..elem, 0);
        }
        elem = elem.saturating_sub(width);

        if debug {
            println!("  After carryover handling: elem={}", elem);
        }

        // Decode symbols until we have enough for this row
        let mut decode_count = 0;
        while elem < width {
            if debug {
                println!(
                    "  Decoding at position {}, need {} more symbols",
                    elem,
                    width - elem
                );
            }
            let symbols = self.decode_sequence(&mut buf[elem..])?;
            if debug {
                println!(
                    "    Decoded {} symbols: {:?}",
                    symbols,
                    &buf[elem..elem + symbols]
                );
            }
            elem += symbols;
            decode_count += 1;
        }

        if debug {
            println!(
                "  Total decode operations: {}, final elem count: {}",
                decode_count, elem
            );
        }

        Ok(elem)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manual_gradient_decode() {
        // Let's manually trace through what should happen
        // For gradient 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

        // Expected first row deltas: 0, 1, 1, 1
        // Row 0 after reconstruction: 0, 1, 2, 3

        // Row 1 deltas:
        // pixel[1,0] = 4, top = 0, delta = 4
        // pixel[1,1] = 5, left = 4, top = 1, avg = 2, delta = 3
        // pixel[1,2] = 6, left = 5, top = 2, avg = 3, delta = 3
        // pixel[1,3] = 7, left = 6, top = 3, avg = 4, delta = 3

        // Let's create a minimal decoder test
        let mut output = [0u8; 16];

        // First row
        output[0] = 0;
        output[1] = 1;
        output[2] = 1 + 1;
        output[3] = 2 + 1;

        // Second row
        output[4] = 4; // top + delta
        output[5] = 4_u8.div_ceil(2) + 3; // avg + delta = 2 + 3 = 5
        output[6] = ((5 + 2) / 2) + 3; // avg + delta = 3 + 3 = 6
        output[7] = ((6 + 3) / 2) + 3; // avg + delta = 4 + 3 = 7

        println!("Manual calculation first 2 rows: {:?}", &output[0..8]);
        assert_eq!(&output[0..8], &[0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_gradient_decode_with_debug() {
        // Use the actual compressed file we have
        let compressed_path = "test_data/gradient_4x4_q0.llic";
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

            // Parse LLIC header to find where entropy data starts
            let num_blocks = compressed_data[1];
            let header_size = 3 + (num_blocks as usize * 4); // version + blocks + quality + block_sizes
            let entropy_data = &compressed_data[header_size..];

            let mut output = vec![0u8; 16];
            let result = decompress_with_debug(entropy_data, 4, 4, 4, &mut output, true);

            println!("\nFinal output: {:?}", output);
            println!("Expected:     {:?}", (0..16).collect::<Vec<u8>>());

            assert!(result.is_ok());

            // Check if we get the expected gradient
            let expected: Vec<u8> = (0..16).collect();
            assert_eq!(
                output, expected,
                "Gradient decompression should produce 0..16"
            );
        }
    }

    #[test]
    fn test_patterns_decompression_debug() {
        use crate::pgm::Pgm;
        use crate::LlicContext;

        // Test specifically for the patterns_64x64.pgm file
        let compressed_path = "test_data/patterns_64x64_q0.llic";
        let original_path = "test_data/patterns_64x64.pgm";

        if std::path::Path::new(compressed_path).exists()
            && std::path::Path::new(original_path).exists()
        {
            println!("\n=== Testing patterns_64x64.pgm decompression ===");

            // Load original for comparison
            let original = Pgm::open(original_path).unwrap();

            // Load compressed file
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

            // Use the proper decompression function that handles multi-block files
            let context = LlicContext::new(64, 64, 64, None).unwrap();
            let mut output = vec![0u8; 64 * 64];

            println!("Using LlicContext to decompress (handles multi-block properly)");
            let result = context.decompress_gray8(compressed_data, &mut output);

            assert!(result.is_ok(), "Decompression should succeed");

            // Check specific byte at position 271 (row 4, column 15)
            println!("\n=== Checking corruption point ===");
            println!(
                "Byte 270: original=0x{:02X}, decompressed=0x{:02X}",
                original.data()[270],
                output[270]
            );
            println!(
                "Byte 271: original=0x{:02X}, decompressed=0x{:02X}",
                original.data()[271],
                output[271]
            );
            println!(
                "Byte 272: original=0x{:02X}, decompressed=0x{:02X}",
                original.data()[272],
                output[272]
            );

            // Check first few rows
            let mut first_diff = None;
            for (i, (orig, dec)) in original.data().iter().zip(output.iter()).enumerate() {
                if orig != dec && first_diff.is_none() {
                    first_diff = Some(i);
                    println!(
                        "\nFirst difference at byte {}: original=0x{:02X}, decompressed=0x{:02X}",
                        i, orig, dec
                    );
                    println!("Row: {}, Column: {}", i / 64, i % 64);
                }
            }

            if let Some(pos) = first_diff {
                // Let's also test with debug mode on a single block to understand the issue
                println!("\n=== Debug mode: Testing individual blocks ===");

                // Parse LLIC header
                let num_blocks = compressed_data[1];
                let header_size = 3 + (num_blocks as usize * 4);

                println!("Number of blocks: {}", num_blocks);
                println!("Header size: {}", header_size);

                // Print all block sizes
                let mut block_sizes = Vec::new();
                for i in 0..num_blocks as usize {
                    let size_offset = 3 + i * 4;
                    let block_size = u32::from_le_bytes([
                        compressed_data[size_offset],
                        compressed_data[size_offset + 1],
                        compressed_data[size_offset + 2],
                        compressed_data[size_offset + 3],
                    ]);
                    block_sizes.push(block_size);
                    println!("Block {} size in header: {}", i, block_size);
                }

                // Find the last non-zero block (which should contain the full image)
                let mut last_nonzero_block = None;
                let mut block_start = header_size;
                for (i, &block_size) in block_sizes.iter().enumerate() {
                    if block_size > 0 {
                        last_nonzero_block = Some((i, block_start, block_size));
                    }
                    block_start += block_size as usize;
                }

                if let Some((block_idx, offset, size)) = last_nonzero_block {
                    println!(
                        "\nLast non-zero block is {} at offset {} with size {}",
                        block_idx, offset, size
                    );
                    let entropy_data = &compressed_data[offset..offset + size as usize];

                    let mut debug_output = vec![0u8; 64 * 64];
                    let result =
                        decompress_with_debug(entropy_data, 64, 64, 64, &mut debug_output, true);

                    if result.is_ok() {
                        println!("\nLast block decompression succeeded!");
                    } else {
                        println!("\nLast block decompression failed: {:?}", result);
                    }
                }

                panic!("Decompression mismatch starting at byte {}", pos);
            } else {
                println!("\nDecompression successful - all bytes match!");
            }
        } else {
            println!("Skipping patterns test - files not found");
        }
    }

    #[cfg(any(
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))]
    fn test_image(width: u32, height: u32, stride: u32) -> Vec<u8> {
        let mut state = 0x9E37_79B9_u32 ^ width;
        (0..stride * height)
            .map(|i| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let (x, y) = (i % stride, i / stride);
                let amplitude = 1 << ((x / 8 + y / 3) % 9);
                #[allow(clippy::cast_possible_truncation)]
                let value = (x * 3 + y * 5 + (state % amplitude)) as u8;
                value
            })
            .collect()
    }

    /// The wavefront must reproduce the per-row decode for every width
    /// remainder and height, both predictors and mixed per-row selection.
    #[test]
    #[cfg(any(
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))]
    fn wavefront_matches_per_row_decode() {
        if !wavefront::available() {
            return;
        }
        for (width, height, stride) in [
            (32u32, 17u32, 32u32),
            (48, 40, 52),
            (68, 33, 68),
            (100, 50, 104),
            (200, 16, 200),
        ] {
            let image = test_image(width, height, stride);
            for dynamic in [false, true] {
                let compressed = if dynamic {
                    crate::entropy_coder::compress_dynamic(&image, width, height, stride)
                } else {
                    crate::entropy_coder::compress(&image, width, height, stride)
                }
                .expect("compress");
                let decode = || {
                    let mut out = vec![0u8; image.len()];
                    if dynamic {
                        decompress_dynamic(&compressed, width, height, stride, &mut out)
                    } else {
                        decompress(&compressed, width, height, stride, &mut out)
                    }
                    .expect("decompress");
                    out
                };
                let wave = decode();
                wavefront::tests::DISABLED.with(|d| d.set(true));
                let per_row = decode();
                wavefront::tests::DISABLED.with(|d| d.set(false));
                let case = format!("{width}x{height} stride {stride} dynamic {dynamic}");
                assert_eq!(wave, per_row, "{case}");
                for y in 0..height as usize {
                    let row = y * stride as usize..y * stride as usize + width as usize;
                    assert_eq!(wave[row.clone()], image[row], "{case} row {y}");
                }
            }
        }
    }

    /// A truncated entropy stream is an error, not zeros decoded past its end.
    #[test]
    fn truncated_entropy_stream_is_an_error() {
        let (width, height) = (64u32, 48u32);
        let image: Vec<u8> = (0..width * height).map(|i| (i * 7 % 251) as u8).collect();
        let compressed =
            crate::entropy_coder::compress(&image, width, height, width).expect("compress");
        let mut out = vec![0u8; image.len()];
        decompress(&compressed, width, height, width, &mut out).expect("full stream");
        assert_eq!(out, image);
        for len in [0, 4, compressed.len() / 2, compressed.len() - 8] {
            assert!(
                matches!(
                    decompress(&compressed[..len], width, height, width, &mut out),
                    Err(LlicError::InvalidData)
                ),
                "truncated to {len} of {}",
                compressed.len()
            );
        }
    }

    /// Decoding two streams together gives each one's own decode, for images
    /// of different heights (one finishes alone) and both predictors, with and
    /// without the wavefront; a truncated stream on either side is an error.
    #[test]
    fn pair_decode_matches_single_decodes() {
        for (width, stride) in [(8u32, 8u32), (64, 72)] {
            for heights in [[24u32, 24], [40, 56], [20, 4]] {
                for dynamic in [false, true] {
                    let case = format!("{width}x{heights:?} stride {stride} dynamic {dynamic}");
                    let images = heights.map(|height| test_image(width, height + 1, stride));
                    let streams: Vec<Vec<u8>> = images
                        .iter()
                        .zip(heights)
                        .map(|(image, height)| {
                            let compress = if dynamic {
                                crate::entropy_coder::compress_dynamic
                            } else {
                                crate::entropy_coder::compress
                            };
                            compress(image, width, height, stride).expect("compress")
                        })
                        .collect();
                    let decode_one = |k: usize, stream: &[u8], out: &mut [u8]| {
                        let (h, w, s) = (heights[k], width, stride);
                        if dynamic {
                            decompress_dynamic(stream, w, h, s, out)
                        } else {
                            decompress(stream, w, h, s, out)
                        }
                    };
                    let sizes = heights.map(|height| (stride * height) as usize);
                    let expected: Vec<Vec<u8>> = (0..2)
                        .map(|k| {
                            let mut out = vec![0u8; sizes[k]];
                            decode_one(k, &streams[k], &mut out).expect("single decode");
                            out
                        })
                        .collect();
                    let (mut out_a, mut out_b) = (vec![0u8; sizes[0]], vec![0u8; sizes[1]]);
                    decompress_pair(
                        [&streams[0], &streams[1]],
                        dynamic,
                        width,
                        heights,
                        stride,
                        [&mut out_a, &mut out_b],
                    )
                    .unwrap_or_else(|e| panic!("{case}: {e}"));
                    assert_eq!(out_a, expected[0], "{case}");
                    assert_eq!(out_b, expected[1], "{case}");

                    for k in 0..2 {
                        let len = streams[k].len();
                        for cut in [0, 4, len / 2, len.saturating_sub(8)] {
                            let mut cut_streams = [&streams[0][..], &streams[1][..]];
                            cut_streams[k] = &streams[k][..cut];
                            assert!(
                                decompress_pair(
                                    cut_streams,
                                    dynamic,
                                    width,
                                    heights,
                                    stride,
                                    [&mut out_a, &mut out_b],
                                )
                                .is_err(),
                                "{case}: stream {k} cut to {cut} of {len}"
                            );
                        }
                    }
                }
            }
        }
    }
}
