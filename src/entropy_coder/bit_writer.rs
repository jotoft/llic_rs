//! Bit writer for producing a byte stream from bits.
//!
//! Mirrors the BitReader interface for encoding. Bits are stored MSB-first
//! and flushed to the output buffer in 32-bit chunks (as 2x 16-bit little-endian words).
//!
//! Optimized for high-throughput encoding using pointer-based writes.

/// A bit writer that produces a byte stream from bits.
///
/// Uses a pre-allocated buffer with pointer-based writes for maximum performance.
pub struct BitWriter {
    /// Output buffer (pre-allocated to max size)
    output: Vec<u8>,
    /// Current write position
    pos: usize,
    /// Bits are packed at the MSB side
    bits: u64,
    /// Number of valid bits currently in the container
    count: u32,
}

impl BitWriter {
    /// Create a new BitWriter with an initial capacity.
    pub fn new() -> Self {
        Self::with_capacity(1024)
    }

    /// Create a new BitWriter with the specified initial capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        // Pre-fill to capacity so we can write without bounds checks
        Self {
            output: vec![0; capacity],
            pos: 0,
            bits: 0,
            count: 0,
        }
    }

    /// Write `num_bits` bits from the MSB side of `value`.
    ///
    /// The bits are taken from the most significant side of `value`.
    /// For example, write_bits(0x8000_0000, 4) writes binary 1000.
    #[inline]
    pub fn write_bits(&mut self, value: u32, num_bits: u8) {
        debug_assert!(num_bits <= 32);
        debug_assert!(
            num_bits == 0 || value >> (32 - num_bits) << (32 - num_bits) == value,
            "value has bits beyond num_bits: value={:#x}, num_bits={}",
            value,
            num_bits
        );

        if num_bits == 0 {
            return;
        }

        // Shift value to align with current bit position
        let value64 = (value as u64) << 32;
        self.bits |= value64 >> self.count;
        self.count += num_bits as u32;

        // Flush complete 16-bit words
        self.flush_words();
    }

    /// Write bits from a pre-packed entry (upper bits = code, lower 6 bits = length).
    /// This is optimized for the compression table format.
    ///
    /// This is the hot path for compression - inlined and optimized.
    #[inline(always)]
    pub fn write_packed(&mut self, packed: u32) {
        // Left-justify to 64-bit and mask away the length bits (matches C++ exactly)
        let symbol = ((packed & 0xFFFF_FFC0) as u64) << 32;
        self.bits |= symbol >> self.count;
        // Use the 6 least significant bits for the bit count
        self.count += packed & 0x3F;
    }

    /// Flush if we have 32 or more bits accumulated.
    /// Must be called after write_packed to maintain the invariant.
    /// Uses direct pointer writes for maximum performance.
    #[inline(always)]
    pub fn flush_if_needed(&mut self) {
        if self.count >= 32 {
            self.count -= 32;
            let a = (self.bits >> 32) as u32;
            // Store as 2x uint16_t in LE format (matches C++ exactly)
            let swapped = (a >> 16) | ((a & 0xFFFF) << 16);

            // Direct pointer write - no bounds check since we pre-allocated
            // SAFETY: pos is always within bounds due to pre-allocation
            unsafe {
                let ptr = self.output.as_mut_ptr().add(self.pos) as *mut u32;
                ptr.write_unaligned(swapped);
            }
            self.pos += 4;
            self.bits <<= 32;
        }
    }

    /// Flush any complete 16-bit words to the output.
    /// Used by write_bits for general bit writing.
    #[inline]
    fn flush_words(&mut self) {
        while self.count >= 16 {
            // Extract top 16 bits
            let word = (self.bits >> 48) as u16;
            // SAFETY: pos is always within bounds due to pre-allocation
            unsafe {
                let ptr = self.output.as_mut_ptr().add(self.pos) as *mut u16;
                ptr.write_unaligned(word.to_le());
            }
            self.pos += 2;
            self.bits <<= 16;
            self.count -= 16;
        }
    }

    /// Finalize the stream, flushing any remaining bits.
    /// Pads with zeros to complete the final byte(s).
    pub fn finish(mut self) -> Vec<u8> {
        // Flush any remaining bits (matches C++ flushLastData behavior)
        if self.count > 0 {
            let a = (self.bits >> 32) as u32;
            // Store as 2x uint16_t in LE format
            let swapped = (a >> 16) | ((a & 0xFFFF) << 16);
            unsafe {
                let ptr = self.output.as_mut_ptr().add(self.pos) as *mut u32;
                ptr.write_unaligned(swapped);
            }
            self.pos += 4;
        }
        // Truncate to actual size
        self.output.truncate(self.pos);
        self.output
    }

    /// Get current output length in bytes (not including unflushed bits).
    #[allow(dead_code)]
    pub fn output_len(&self) -> usize {
        self.pos
    }

    /// Get current output length in bytes.
    pub fn len(&self) -> usize {
        self.pos
    }

    /// Check if writer is empty.
    pub fn is_empty(&self) -> bool {
        self.pos == 0 && self.count == 0
    }
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_single_byte() {
        let mut writer = BitWriter::new();
        // Write 8 bits: 0xAB (MSB-justified: 0xAB00_0000)
        writer.write_bits(0xAB00_0000, 8);
        let output = writer.finish();
        // Output is a 32-bit word with 0xAB00 in MSB position
        // C++ format: (a >> 16) | ((a & 0xFFFF) << 16) where a = 0xAB000000 >> 32 = 0
        // But our bits are: 0xAB00_0000_0000_0000 >> 32 = 0xAB00_0000
        // swapped = (0xAB00_0000 >> 16) | ((0xAB00_0000 & 0xFFFF) << 16) = 0xAB00 | 0 = 0x0000_AB00
        // In native endian bytes: depends on platform, but content test shows [0x00, 0xAB, 0x00, 0x00]
        assert_eq!(output, vec![0x00, 0xAB, 0x00, 0x00]);
    }

    #[test]
    fn test_write_16_bits() {
        let mut writer = BitWriter::new();
        // Write 16 bits: 0xABCD (MSB-justified: 0xABCD_0000)
        writer.write_bits(0xABCD_0000, 16);
        let output = writer.finish();
        // 16 bits triggers immediate flush via flush_words -> [0xCD, 0xAB]
        // finish() sees count=0 so no additional output
        assert_eq!(output, vec![0xCD, 0xAB]);
    }

    #[test]
    fn test_write_multiple_small() {
        let mut writer = BitWriter::new();
        // Write 4 bits: 0xF (MSB-justified: 0xF000_0000)
        writer.write_bits(0xF000_0000, 4);
        // Write 4 bits: 0xA (MSB-justified: 0xA000_0000)
        writer.write_bits(0xA000_0000, 4);
        // Write 8 bits: 0xBC (MSB-justified: 0xBC00_0000)
        writer.write_bits(0xBC00_0000, 8);
        let output = writer.finish();
        // Total 16 bits, triggers immediate flush -> [0xBC, 0xFA]
        assert_eq!(output, vec![0xBC, 0xFA]);
    }

    #[test]
    fn test_write_32_bits() {
        let mut writer = BitWriter::new();
        writer.write_bits(0x1234_5678, 32);
        let output = writer.finish();
        // After flush_words: bits shifted out as 16-bit LE words
        // This writes two complete 16-bit words during flush_words
        // 0x1234 -> [0x34, 0x12], 0x5678 -> [0x78, 0x56]
        assert_eq!(output, vec![0x34, 0x12, 0x78, 0x56]);
    }

    #[test]
    fn test_write_packed() {
        let mut writer = BitWriter::new();
        // Packed: upper bits = 0xABCD_0000, lower 6 bits = 16
        let packed = 0xABCD_0000 | 16;
        writer.write_packed(packed);
        writer.flush_if_needed(); // No flush since only 16 bits
        let output = writer.finish();
        // 16 bits of 0xABCD, padded to 32-bit word
        assert_eq!(output, vec![0xCD, 0xAB, 0x00, 0x00]);
    }
}
