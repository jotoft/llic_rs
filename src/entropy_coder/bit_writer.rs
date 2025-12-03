//! Bit writer for producing a byte stream from bits.
//!
//! Mirrors the BitReader interface for encoding. Bits are stored MSB-first
//! and flushed to the output buffer in 16-bit little-endian words.

/// A bit writer that produces a byte stream from bits.
///
/// Bits are accumulated MSB-first in a 64-bit container and flushed
/// as 16-bit little-endian words when full.
pub struct BitWriter {
    /// Output buffer
    output: Vec<u8>,
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
        Self {
            output: Vec::with_capacity(capacity),
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
        debug_assert!(num_bits == 0 || value >> (32 - num_bits) << (32 - num_bits) == value,
            "value has bits beyond num_bits: value={:#x}, num_bits={}", value, num_bits);

        if num_bits == 0 {
            return;
        }

        // Shift value to align with current bit position
        // value is already MSB-justified (32-bit), extend to 64-bit and position
        let value64 = (value as u64) << 32;
        self.bits |= value64 >> self.count;
        self.count += num_bits as u32;

        // Flush complete 16-bit words
        self.flush_words();
    }

    /// Write bits from a pre-packed entry (upper bits = code, lower 6 bits = length).
    /// This is optimized for the compression table format.
    #[inline]
    pub fn write_packed(&mut self, packed: u32) {
        let num_bits = (packed & 0x3F) as u8;
        let code = packed & !0x3F;  // Clear the length bits, keep code MSB-justified
        self.write_bits(code, num_bits);
    }

    /// Flush any complete 16-bit words to the output.
    #[inline]
    fn flush_words(&mut self) {
        while self.count >= 16 {
            // Extract top 16 bits
            let word = (self.bits >> 48) as u16;
            self.output.extend_from_slice(&word.to_le_bytes());
            self.bits <<= 16;
            self.count -= 16;
        }
    }

    /// Finalize the stream, flushing any remaining bits.
    /// Pads with zeros to complete the final byte(s).
    pub fn finish(mut self) -> Vec<u8> {
        // Flush any remaining bits (pad to 16-bit boundary)
        if self.count > 0 {
            // Pad to 16 bits
            let word = (self.bits >> 48) as u16;
            self.output.extend_from_slice(&word.to_le_bytes());
        }
        self.output
    }

    /// Get current output length in bytes.
    pub fn len(&self) -> usize {
        self.output.len()
    }

    /// Check if writer is empty.
    pub fn is_empty(&self) -> bool {
        self.output.is_empty() && self.count == 0
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
        // Output should be 0xAB padded to 16 bits: [0xAB, 0x00] in LE
        assert_eq!(output, vec![0x00, 0xAB]);
    }

    #[test]
    fn test_write_16_bits() {
        let mut writer = BitWriter::new();
        // Write 16 bits: 0xABCD (MSB-justified: 0xABCD_0000)
        writer.write_bits(0xABCD_0000, 16);
        let output = writer.finish();
        // Output: LE 16-bit word 0xABCD -> [0xCD, 0xAB]
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
        // Combined: 0xFABC -> LE: [0xBC, 0xFA]
        assert_eq!(output, vec![0xBC, 0xFA]);
    }

    #[test]
    fn test_write_32_bits() {
        let mut writer = BitWriter::new();
        writer.write_bits(0x1234_5678, 32);
        let output = writer.finish();
        // Two 16-bit words: 0x1234, 0x5678 -> LE: [0x34, 0x12, 0x78, 0x56]
        assert_eq!(output, vec![0x34, 0x12, 0x78, 0x56]);
    }

    #[test]
    fn test_write_packed() {
        let mut writer = BitWriter::new();
        // Packed: upper bits = 0xABCD_0000, lower 6 bits = 16
        let packed = 0xABCD_0000 | 16;
        writer.write_packed(packed);
        let output = writer.finish();
        assert_eq!(output, vec![0xCD, 0xAB]);
    }
}
