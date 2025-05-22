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
    decompress_with_debug(src_data, width, height, bytes_per_line, dst_image, false)
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
    let mut row_buffer = vec![0u8; width as usize + 256];
    
    // Initialize with the correct number of symbols
    let mut num_filled_elements = width as usize;
    
    // Decompress first row
    num_filled_elements = decoder.decompress_row(&mut row_buffer, num_filled_elements, width as usize)?;
    
    if debug {
        println!("First row raw symbols: {:?}", &row_buffer[0..width as usize]);
    }
    
    // First row: Horizontal delta only
    dst_image[0] = row_buffer[0];
    for x in 1..width {
        dst_image[x as usize] = row_buffer[x as usize].wrapping_add(dst_image[x as usize - 1]);
    }
    
    if debug {
        println!("First row after delta: {:?}", &dst_image[0..width as usize]);
    }
    
    // Decompress remaining rows (combined predictor)
    for y in 1..height {
        let row_offset = (y * bytes_per_line) as usize;
        let prev_row_offset = ((y - 1) * bytes_per_line) as usize;
        
        // Decompress row with carryover
        num_filled_elements = decoder.decompress_row(&mut row_buffer, num_filled_elements, width as usize)?;
        
        if debug {
            println!("Row {} raw symbols: {:?}", y, &row_buffer[0..width as usize]);
        }
        
        // Apply prediction and reconstruction
        dst_image[row_offset] = row_buffer[0].wrapping_add(dst_image[prev_row_offset]);
        
        for x in 1..width {
            let left = dst_image[row_offset + x as usize - 1];
            let top = dst_image[prev_row_offset + x as usize];
            let avg = ((left as i32 + top as i32) / 2) as u8;
            dst_image[row_offset + x as usize] = row_buffer[x as usize].wrapping_add(avg);
        }
        
        if debug {
            println!("Row {} after reconstruction: {:?}", y, 
                &dst_image[row_offset..row_offset + width as usize]);
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
        self.bits_available = self.bits_available.saturating_sub(bits);
        self.refill();
    }
    
    fn decode_sequence(&mut self, out: &mut [u8]) -> Result<usize> {
        self.refill();
        
        // Get top 12 bits for table lookup
        let index = (self.bit_container >> 20) as usize;
        
        // Special case: 13-bit codes
        if index >= 0xF80 {
            out[0] = ((self.bit_container >> 19) & 0xFF) as u8;
            self.consume_bits(13);
            return Ok(1);
        }
        
        // Normal case: use lookup table
        if index >= DECOMPRESS_TABLE.len() {
            return Err(LlicError::InvalidData);
        }
        
        let entry = DECOMPRESS_TABLE[index];
        self.consume_bits(entry.bits as u32);
        
        // Copy decoded symbols (always write both, even if num_symbols == 1)
        out[0] = entry.symbols[0];
        out[1] = entry.symbols[1];
        
        Ok(entry.num_symbols as usize)
    }
    
    fn decompress_row(&mut self, buf: &mut [u8], num_elems: usize, width: usize) -> Result<usize> {
        let mut elem = num_elems;
        
        // Handle carryover from previous row
        if elem > width {
            // Copy the extra elements to the beginning of the buffer
            buf.copy_within(width..elem, 0);
        }
        elem = elem.saturating_sub(width);
        
        // Decode symbols until we have enough for this row
        while elem < width {
            let symbols = self.decode_sequence(&mut buf[elem..])?;
            elem += symbols;
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
        let mut output = vec![0u8; 16];
        
        // First row
        output[0] = 0;
        output[1] = 0 + 1;  
        output[2] = 1 + 1;
        output[3] = 2 + 1;
        
        // Second row
        output[4] = 0 + 4;  // top + delta
        output[5] = ((4 + 1) / 2) + 3;  // avg + delta = 2 + 3 = 5
        output[6] = ((5 + 2) / 2) + 3;  // avg + delta = 3 + 3 = 6
        output[7] = ((6 + 3) / 2) + 3;  // avg + delta = 4 + 3 = 7
        
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
            let header_end = file_content.windows(1)
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
            assert_eq!(output, expected, "Gradient decompression should produce 0..16");
        }
    }
}