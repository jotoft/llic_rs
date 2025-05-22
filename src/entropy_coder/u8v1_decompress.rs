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
    
    // Decompress first row (horizontal delta only)
    let mut prev = 0u8;
    for x in 0..width {
        let delta = decoder.decode_symbol()?;
        prev = prev.wrapping_add(delta);
        dst_image[x as usize] = prev;
    }
    
    // Decompress remaining rows (combined predictor)
    for y in 1..height {
        let row_offset = (y * bytes_per_line) as usize;
        let prev_row_offset = ((y - 1) * bytes_per_line) as usize;
        
        // First pixel of row - use top pixel only (left is implicitly 0 at start of row)
        let top = dst_image[prev_row_offset];
        let avg = top / 2;  // (0 + top) / 2
        let delta = decoder.decode_symbol()?;
        dst_image[row_offset] = avg.wrapping_add(delta);
        
        // Rest of row
        for x in 1..width {
            let left = dst_image[row_offset + x as usize - 1];
            let top = dst_image[prev_row_offset + x as usize];
            let avg = ((left as u16 + top as u16) / 2) as u8;
            let delta = decoder.decode_symbol()?;
            dst_image[row_offset + x as usize] = avg.wrapping_add(delta);
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
        while self.bits_available <= 24 && self.pos < self.data.len() {
            self.bit_container |= (self.data[self.pos] as u32) << (24 - self.bits_available);
            self.bits_available += 8;
            self.pos += 1;
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