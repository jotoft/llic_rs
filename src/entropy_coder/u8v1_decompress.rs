use crate::Result;

mod decompress_table;
use decompress_table::U8V1_DECOMPRESS_TABLE_2X;

#[derive(Debug)]
struct DecompressStream<'a> {
    src_ptr: &'a [u8],
    pos: usize,
    bit_container: u32,
    num_bits: u32,
}

impl<'a> DecompressStream<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            src_ptr: data,
            pos: 0,
            bit_container: 0,
            num_bits: 0,
        }
    }

    #[inline]
    fn get_more_bytes_if_possible(&mut self) {
        if self.num_bits < 16 && self.pos + 1 < self.src_ptr.len() {
            // Read 16 bits (2 bytes) as little-endian
            let tmp = u16::from_le_bytes([
                self.src_ptr[self.pos],
                self.src_ptr[self.pos + 1],
            ]) as u32;
            self.pos += 2;

            self.bit_container |= tmp << (16 - self.num_bits);
            self.num_bits += 16;
        }
    }

    #[inline]
    fn decompress_sequence(&mut self, out: &mut [u8]) -> usize {
        self.get_more_bytes_if_possible();
        let p = self.bit_container;

        // It is very likely that we have a symbol with <= 9 bits.
        if p < 0xF800_0000 {
            let t = &U8V1_DECOMPRESS_TABLE_2X[(p >> 20) as usize];

            self.bit_container = p << t.bits;
            self.num_bits -= t.bits as u32;
            out[0] = t.symbol[0];
            out[1] = t.symbol[1]; // Always written, even if num_symbols == 1.

            return t.num_symbols as usize;
        }

        // This is the case with 13-bit codes: The actual symbol is given by the least significant 8 bits of the code.
        self.bit_container = p << 13;
        self.num_bits -= 13;
        out[0] = ((p >> 19) & 0xFF) as u8;
        1
    }

    #[inline]
    fn decompress_row(&mut self, buf: &mut [u8], num_elems: usize, width: usize) -> usize {
        let mut elem = num_elems;
        
        // We are quite likely to have decoded more elements than the previous row.
        if elem > width {
            // Copy the last elements to the beginning of the buffer.
            buf.copy_within((width..elem), 0);
        }
        elem = elem.saturating_sub(width);

        while elem < width {
            let count = self.decompress_sequence(&mut buf[elem..]);
            elem += count;
        }

        elem
    }
}

/// Decompresses u8v1 entropy-coded data
/// 
/// # Arguments
/// * `compressed_data` - The compressed input data
/// * `width` - Image width
/// * `height` - Image height  
/// * `bytes_per_line` - Number of bytes per line in the output image
/// * `row_buffer` - Working buffer, must be at least width + 256 bytes
/// * `out_image` - Output image buffer
///
/// # Returns
/// Ok(()) on success, or an error
pub fn decompress(
    compressed_data: &[u8],
    width: usize,
    height: usize,
    bytes_per_line: usize,
    row_buffer: &mut [u8],
    out_image: &mut [u8],
) -> Result<()> {
    // Nothing to do? This is a valid case, and protects against invalid memory access later.
    if width == 0 || height == 0 {
        return Ok(());
    }

    let mut stream = DecompressStream::new(compressed_data);

    // "Fake" that we have used the correct number of symbols.
    let mut num_filled_elements = width;
    num_filled_elements = stream.decompress_row(row_buffer, num_filled_elements, width);

    // First row: Horizontal delta only.
    out_image[0] = row_buffer[0];
    for x in 1..width {
        out_image[x] = row_buffer[x].wrapping_add(out_image[x - 1]);
    }

    // Rows 2 to N: Horizontal + vertical delta.
    for y in 1..height {
        num_filled_elements = stream.decompress_row(row_buffer, num_filled_elements, width);

        let p0_start = (y - 1) * bytes_per_line;
        let p1_start = y * bytes_per_line;

        out_image[p1_start] = row_buffer[0].wrapping_add(out_image[p0_start]);
        
        for x in 1..width {
            let pred = (out_image[p1_start + x - 1] as i32 + out_image[p0_start + x] as i32) / 2;
            out_image[p1_start + x] = row_buffer[x].wrapping_add(pred as u8);
        }
    }

    Ok(())
}