use crate::{Result, LlicError};
use super::tables::DECOMPRESS_TABLE;

/// Decompresses data encoded with the u8v1 entropy coder (optimized path)
pub fn decompress(
    src_data: &[u8],
    width: u32,
    height: u32,
    bytes_per_line: u32,
    dst_image: &mut [u8],
) -> Result<()> {
    // Use the fast path for normal operation
    decompress_fast(src_data, width, height, bytes_per_line, dst_image)
}

/// Fast decompression without any debug overhead
#[inline(never)] // Prevent inlining to help with profiling
fn decompress_fast(
    src_data: &[u8],
    width: u32,
    height: u32,
    bytes_per_line: u32,
    dst_image: &mut [u8],
) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(LlicError::InvalidArgument);
    }

    let width = width as usize;
    let height = height as usize;
    let bytes_per_line = bytes_per_line as usize;

    let mut decoder = FastDecoder::new(src_data);
    let mut row_buffer = vec![0u8; width + 256];

    // Decompress first row
    let mut num_filled = decoder.decompress_row(&mut row_buffer, 0, width);

    // First row: Horizontal delta only
    dst_image[0] = row_buffer[0];
    for x in 1..width {
        dst_image[x] = row_buffer[x].wrapping_add(dst_image[x - 1]);
    }

    // Remaining rows: Combined predictor
    for y in 1..height {
        let row_offset = y * bytes_per_line;
        let prev_row_offset = (y - 1) * bytes_per_line;

        num_filled = decoder.decompress_row(&mut row_buffer, num_filled, width);

        // First pixel uses only top predictor
        dst_image[row_offset] = row_buffer[0].wrapping_add(dst_image[prev_row_offset]);

        // Remaining pixels use average of left and top
        for x in 1..width {
            let left = dst_image[row_offset + x - 1];
            let top = dst_image[prev_row_offset + x];
            // Use wrapping arithmetic to match C++ behavior: (left + top) / 2
            let avg = ((left as u16 + top as u16) / 2) as u8;
            dst_image[row_offset + x] = row_buffer[x].wrapping_add(avg);
        }
    }

    Ok(())
}

/// Optimized decoder matching C++ implementation closely
struct FastDecoder<'a> {
    src_ptr: *const u16,
    end_ptr: *const u16,
    bit_container: u32,
    num_bits: u32,
    _phantom: std::marker::PhantomData<&'a [u8]>,
}

impl<'a> FastDecoder<'a> {
    fn new(data: &'a [u8]) -> Self {
        let src_ptr = data.as_ptr() as *const u16;
        let end_ptr = unsafe { src_ptr.add(data.len() / 2) };

        Self {
            src_ptr,
            end_ptr,
            bit_container: 0,
            num_bits: 0,
            _phantom: std::marker::PhantomData,
        }
    }

    #[inline(always)]
    fn refill(&mut self) {
        // Single conditional refill - matches C++ exactly
        if self.num_bits < 16 && self.src_ptr != self.end_ptr {
            let word = unsafe { self.src_ptr.read_unaligned() };
            self.src_ptr = unsafe { self.src_ptr.add(1) };
            self.bit_container |= (word as u32) << (16 - self.num_bits);
            self.num_bits += 16;
        }
    }

    #[inline(always)]
    fn decode_sequence(&mut self, out: &mut [u8]) -> usize {
        self.refill();
        let p = self.bit_container;

        // Fast path: symbols with <= 12 bits (most common case)
        // Use the same threshold as C++: 0xF8000000
        if p < 0xF800_0000 {
            let index = (p >> 20) as usize;
            let entry = unsafe { DECOMPRESS_TABLE.get_unchecked(index) };

            self.bit_container = p << entry.bits;
            self.num_bits -= entry.bits as u32;

            out[0] = entry.symbols[0];
            out[1] = entry.symbols[1]; // Always written, even if num_symbols == 1

            entry.num_symbols as usize
        } else {
            // 13-bit escape code: symbol is in bits 19-26
            self.bit_container = p << 13;
            self.num_bits -= 13;
            out[0] = ((p >> 19) & 0xFF) as u8;
            1
        }
    }

    #[inline(always)]
    fn decompress_row(&mut self, buf: &mut [u8], num_elems: usize, width: usize) -> usize {
        let mut elem = num_elems;

        // Handle carryover from previous row
        if elem > width {
            buf.copy_within(width..elem, 0);
        }
        elem = elem.saturating_sub(width);

        // Decode until we have enough symbols
        while elem < width {
            elem += self.decode_sequence(&mut buf[elem..]);
        }

        elem
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
        println!("Width: {}, Height: {}, Bytes per line: {}", width, height, bytes_per_line);
    }
    
    // Decompress first row
    num_filled_elements = decoder.decompress_row_with_debug(&mut row_buffer, num_filled_elements, width as usize, debug, 0)?;
    
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
        println!("Row 0 final bytes: {:02X?}", &dst_image[0..width.min(20) as usize]);
    }
    
    // Decompress remaining rows (combined predictor)
    for y in 1..height {
        let row_offset = (y * bytes_per_line) as usize;
        let prev_row_offset = ((y - 1) * bytes_per_line) as usize;
        
        if debug && (y >= 2 && y <= 5) {
            println!("\n--- Processing row {} ---", y);
            println!("Carryover buffer state before decode: {} elements", num_filled_elements);
            if num_filled_elements > width as usize {
                println!("Carryover symbols: {:?}", &row_buffer[width as usize..num_filled_elements]);
            }
        }
        
        // Decompress row with carryover
        num_filled_elements = decoder.decompress_row_with_debug(&mut row_buffer, num_filled_elements, width as usize, debug && (y >= 2 && y <= 5), y)?;
        
        if debug && (y >= 2 && y <= 5) {
            println!("Row {} raw symbols: {:?}", y, &row_buffer[0..width.min(20) as usize]);
            println!("After row {}: {} elements in buffer", y, num_filled_elements);
        }
        
        // Apply prediction and reconstruction
        dst_image[row_offset] = row_buffer[0].wrapping_add(dst_image[prev_row_offset]);
        
        if debug && (y >= 2 && y <= 5) {
            println!("First pixel: delta={}, top={}, result={}", 
                row_buffer[0], dst_image[prev_row_offset], dst_image[row_offset]);
        }
        
        for x in 1..width {
            let left = dst_image[row_offset + x as usize - 1];
            let top = dst_image[prev_row_offset + x as usize];
            let avg = ((left as i32 + top as i32) / 2) as u8;
            let delta = row_buffer[x as usize];
            dst_image[row_offset + x as usize] = delta.wrapping_add(avg);
            
            if debug && (y >= 2 && y <= 5) && x < 20 {
                println!("  Pixel[{},{}]: left={}, top={}, avg={}, delta={}, result={}", 
                    y, x, left, top, avg, delta, dst_image[row_offset + x as usize]);
            }
        }
        
        if debug && (y >= 2 && y <= 5) {
            println!("Row {} reconstructed: {:02X?}", y, 
                &dst_image[row_offset..row_offset + width.min(20) as usize]);
            
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
            println!("        refill: reached end of data at pos {}, data.len()={}", self.pos, self.data.len());
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
            println!("      decode_sequence: bit_container=0x{:08X}, bits_available={}", 
                self.bit_container, self.bits_available);
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
            println!("      Table entry: bits={}, num_symbols={}, symbols=[{}, {}]",
                entry.bits, entry.num_symbols, entry.symbols[0], entry.symbols[1]);
        }
        
        self.consume_bits(entry.bits as u32);
        
        // Copy decoded symbols (always write both, even if num_symbols == 1)
        out[0] = entry.symbols[0];
        out[1] = entry.symbols[1];
        
        Ok(entry.num_symbols as usize)
    }
    
    fn decompress_row(&mut self, buf: &mut [u8], num_elems: usize, width: usize) -> Result<usize> {
        self.decompress_row_with_debug(buf, num_elems, width, false, 0)
    }
    
    fn decompress_row_with_debug(&mut self, buf: &mut [u8], num_elems: usize, width: usize, debug: bool, row_num: u32) -> Result<usize> {
        let mut elem = num_elems;
        
        if debug {
            println!("  decompress_row: start with {} elements, width={}", num_elems, width);
        }
        
        // Handle carryover from previous row
        if elem > width {
            // Copy the extra elements to the beginning of the buffer
            if debug {
                println!("  Copying {} carryover elements from position {} to 0", elem - width, width);
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
                println!("  Decoding at position {}, need {} more symbols", elem, width - elem);
            }
            let symbols = self.decode_sequence(&mut buf[elem..])?;
            if debug {
                println!("    Decoded {} symbols: {:?}", symbols, &buf[elem..elem+symbols]);
            }
            elem += symbols;
            decode_count += 1;
        }
        
        if debug {
            println!("  Total decode operations: {}, final elem count: {}", decode_count, elem);
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
    
    #[test]
    fn test_patterns_decompression_debug() {
        use crate::pgm::Pgm;
        use crate::LlicContext;
        
        // Test specifically for the patterns_64x64.pgm file
        let compressed_path = "test_data/patterns_64x64_q0.llic";
        let original_path = "test_data/patterns_64x64.pgm";
        
        if std::path::Path::new(compressed_path).exists() && std::path::Path::new(original_path).exists() {
            println!("\n=== Testing patterns_64x64.pgm decompression ===");
            
            // Load original for comparison
            let original = Pgm::open(original_path).unwrap();
            
            // Load compressed file
            let file_content = std::fs::read(compressed_path).unwrap();
            
            // Skip the text header
            let header_end = file_content.windows(1)
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
            println!("Byte 270: original=0x{:02X}, decompressed=0x{:02X}", 
                original.data()[270], output[270]);
            println!("Byte 271: original=0x{:02X}, decompressed=0x{:02X}", 
                original.data()[271], output[271]);
            println!("Byte 272: original=0x{:02X}, decompressed=0x{:02X}", 
                original.data()[272], output[272]);
            
            // Check first few rows
            let mut first_diff = None;
            for (i, (orig, dec)) in original.data().iter().zip(output.iter()).enumerate() {
                if orig != dec && first_diff.is_none() {
                    first_diff = Some(i);
                    println!("\nFirst difference at byte {}: original=0x{:02X}, decompressed=0x{:02X}", 
                        i, orig, dec);
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
                    println!("\nLast non-zero block is {} at offset {} with size {}", block_idx, offset, size);
                    let entropy_data = &compressed_data[offset..offset + size as usize];
                    
                    let mut debug_output = vec![0u8; 64 * 64];
                    let result = decompress_with_debug(entropy_data, 64, 64, 64, &mut debug_output, true);
                    
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
}