use crate::{Result, LlicError};
use super::tables::DECOMPRESS_TABLE;

/// Decompresses data encoded with the u8v1 entropy coder
pub fn decompress(
    src_data: &[u8],
    width: u32,
    height: u32,
    bytes_per_line: u32,
    dst_image: &mut [u8],
) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(LlicError::InvalidArgument);
    }
    
    let mut decoder = Decoder::new(src_data);
    let mut row_buffer = vec![0u8; width as usize + 256];
    
    // Decompress first row (horizontal delta only)
    for x in 0..width {
        row_buffer[x as usize] = decoder.decode_symbol()?;
    }
    
    // Apply horizontal delta decoding to first row
    dst_image[0] = row_buffer[0];
    for x in 1..width {
        dst_image[x as usize] = row_buffer[x as usize].wrapping_add(dst_image[x as usize - 1]);
    }
    
    // Decompress remaining rows (combined predictor)
    for y in 1..height {
        let row_offset = (y * bytes_per_line) as usize;
        let prev_row_offset = ((y - 1) * bytes_per_line) as usize;
        
        // Decode symbols for this row
        for x in 0..width {
            row_buffer[x as usize] = decoder.decode_symbol()?;
        }
        
        // First pixel of row - just add to pixel above
        dst_image[row_offset] = row_buffer[0].wrapping_add(dst_image[prev_row_offset]);
        
        // Rest of row - use average of left and top
        for x in 1..width {
            let left = dst_image[row_offset + x as usize - 1];
            let top = dst_image[prev_row_offset + x as usize];
            let avg = ((left as i32 + top as i32) / 2) as u8;
            dst_image[row_offset + x as usize] = row_buffer[x as usize].wrapping_add(avg);
        }
    }
    
    Ok(())
}

struct Decoder<'a> {
    data: &'a [u8],
    pos: usize,
    bit_container: u32,
    bits_available: u32,
}

impl<'a> Decoder<'a> {
    fn new(data: &'a [u8]) -> Self {
        let mut decoder = Self {
            data,
            pos: 0,
            bit_container: 0,
            bits_available: 0,
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
        }
    }
    
    fn consume_bits(&mut self, bits: u32) {
        self.bit_container <<= bits;
        self.bits_available -= bits;
        self.refill();
    }
    
    fn decode_symbol(&mut self) -> Result<u8> {
        // Get top 12 bits for table lookup
        let index = (self.bit_container >> 20) as usize;
        
        // Special case: 13-bit codes
        if index >= 0xF80 {
            let symbol = ((self.bit_container >> 19) & 0xFF) as u8;
            self.consume_bits(13);
            return Ok(symbol);
        }
        
        // Normal case: use lookup table
        if index >= DECOMPRESS_TABLE.len() {
            return Err(LlicError::InvalidData);
        }
        
        let entry = DECOMPRESS_TABLE[index];
        self.consume_bits(entry.bits as u32);
        
        // For single symbol decoding, return the first symbol
        Ok(entry.symbols[0])
    }
    
    fn decode_symbols(&mut self, symbols: &mut [u8]) -> Result<usize> {
        // Get top 12 bits for table lookup
        let index = (self.bit_container >> 20) as usize;
        
        // Special case: 13-bit codes
        if index >= 0xF80 {
            symbols[0] = ((self.bit_container >> 19) & 0xFF) as u8;
            self.consume_bits(13);
            return Ok(1);
        }
        
        // Normal case: use lookup table
        if index >= DECOMPRESS_TABLE.len() {
            return Err(LlicError::InvalidData);
        }
        
        let entry = DECOMPRESS_TABLE[index];
        self.consume_bits(entry.bits as u32);
        
        // Copy decoded symbols
        let num_symbols = entry.num_symbols as usize;
        symbols[..num_symbols].copy_from_slice(&entry.symbols[..num_symbols]);
        
        Ok(num_symbols)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_decoder_creation() {
        let data = vec![0xFF, 0xAA, 0x55, 0x00];
        let decoder = Decoder::new(&data);
        assert_eq!(decoder.pos, 4);
        assert_eq!(decoder.bits_available, 32);
    }
}