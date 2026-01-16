//! Bit reader for consuming bits from a byte stream.
//!
//! Provides generic implementations for different container sizes (32-bit, 64-bit).

use std::ops::{BitOrAssign, Shl, ShlAssign};

/// Trait for types that can be used as bit containers.
pub trait BitContainer:
    Copy + Default + Ord + BitOrAssign + Shl<u32, Output = Self> + ShlAssign<u32> + From<u16>
{
    /// Number of bits in the container
    const BITS: u32;

    /// Threshold for the 13-bit escape code check (0xF800... at MSB)
    const ESCAPE_THRESHOLD: Self;

    /// Extract the top 32 bits as u32 for table lookup
    fn top32(&self) -> u32;

    /// Right shift by n bits
    fn shr(self, n: u32) -> Self;
}

impl BitContainer for u32 {
    const BITS: u32 = 32;
    const ESCAPE_THRESHOLD: Self = 0xF800_0000;

    #[inline(always)]
    fn top32(&self) -> u32 {
        *self
    }

    #[inline(always)]
    fn shr(self, n: u32) -> Self {
        self >> n
    }
}

impl BitContainer for u64 {
    const BITS: u32 = 64;
    const ESCAPE_THRESHOLD: Self = 0xF800_0000_0000_0000;

    #[inline(always)]
    fn top32(&self) -> u32 {
        (*self >> 32) as u32
    }

    #[inline(always)]
    fn shr(self, n: u32) -> Self {
        self >> n
    }
}

impl BitContainer for u128 {
    const BITS: u32 = 128;
    const ESCAPE_THRESHOLD: Self = 0xF800_0000_0000_0000_0000_0000_0000_0000;

    #[inline(always)]
    fn top32(&self) -> u32 {
        (*self >> 96) as u32
    }

    #[inline(always)]
    fn shr(self, n: u32) -> Self {
        self >> n
    }
}

/// 256-bit container using two u128s
#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct U256 {
    /// High 128 bits (MSB side)
    hi: u128,
    /// Low 128 bits
    lo: u128,
}

impl U256 {
    pub const ZERO: Self = Self { hi: 0, lo: 0 };
}

impl BitOrAssign for U256 {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        self.hi |= rhs.hi;
        self.lo |= rhs.lo;
    }
}

impl Shl<u32> for U256 {
    type Output = Self;

    #[inline(always)]
    fn shl(self, n: u32) -> Self {
        if n == 0 {
            self
        } else if n >= 256 {
            Self::ZERO
        } else if n >= 128 {
            Self {
                hi: self.lo << (n - 128),
                lo: 0,
            }
        } else {
            Self {
                hi: (self.hi << n) | (self.lo >> (128 - n)),
                lo: self.lo << n,
            }
        }
    }
}

impl ShlAssign<u32> for U256 {
    #[inline(always)]
    fn shl_assign(&mut self, n: u32) {
        *self = *self << n;
    }
}

impl From<u16> for U256 {
    #[inline(always)]
    fn from(v: u16) -> Self {
        Self {
            hi: 0,
            lo: v as u128,
        }
    }
}

impl BitContainer for U256 {
    const BITS: u32 = 256;
    const ESCAPE_THRESHOLD: Self = Self {
        hi: 0xF800_0000_0000_0000_0000_0000_0000_0000,
        lo: 0,
    };

    #[inline(always)]
    fn top32(&self) -> u32 {
        (self.hi >> 96) as u32
    }

    #[inline(always)]
    fn shr(self, n: u32) -> Self {
        if n == 0 {
            self
        } else if n >= 256 {
            Self::ZERO
        } else if n >= 128 {
            Self {
                hi: 0,
                lo: self.hi >> (n - 128),
            }
        } else {
            Self {
                hi: self.hi >> n,
                lo: (self.lo >> n) | (self.hi << (128 - n)),
            }
        }
    }
}

/// A bit reader that consumes bits from a byte stream.
///
/// Bits are stored MSB-first in the container. New bits are added
/// after existing bits, and consumption shifts bits out from the MSB side.
///
/// # Type Parameter
/// - `T`: The container type (u32 or u64)
pub struct BitReader<'a, T: BitContainer> {
    src: &'a [u8],
    pos: usize,
    /// Bits are packed at the MSB side
    bits: T,
    /// Number of valid bits currently in the container
    count: u32,
}

impl<'a, T: BitContainer> BitReader<'a, T> {
    /// Create a new BitReader from a byte slice.
    #[inline(always)]
    pub fn new(src: &'a [u8]) -> Self {
        let mut reader = Self {
            src,
            pos: 0,
            bits: T::default(),
            count: 0,
        };
        reader.refill();
        reader
    }

    /// Refill the bit container if we have room and data available.
    /// Adds 16 bits at a time, positioning them after existing bits.
    #[inline(always)]
    pub fn refill(&mut self) {
        // For 32-bit: refill when < 16 bits (can add one 16-bit word)
        // For 64-bit: refill when <= 48 bits (can add up to 3 words)
        let threshold = T::BITS - 16;

        while self.count <= threshold && self.pos + 2 <= self.src.len() {
            let word = unsafe {
                u16::from_le_bytes([
                    *self.src.get_unchecked(self.pos),
                    *self.src.get_unchecked(self.pos + 1),
                ])
            };
            // Position new bits after existing bits
            // For 32-bit with 0 bits: shift by 16, word goes to bits 16-31
            // For 64-bit with 0 bits: shift by 48, word goes to bits 48-63
            let shift = T::BITS - 16 - self.count;
            self.bits |= T::from(word) << shift;
            self.count += 16;
            self.pos += 2;
        }
    }

    /// Peek at the top 32 bits for table lookup, without consuming.
    #[inline(always)]
    pub fn peek(&self) -> u32 {
        self.bits.top32()
    }

    /// Check if the current bits indicate an escape code.
    #[inline(always)]
    pub fn is_escape(&self) -> bool {
        self.bits >= T::ESCAPE_THRESHOLD
    }

    /// Consume n bits, shifting them out of the container.
    #[inline(always)]
    pub fn consume(&mut self, n: u8) {
        self.bits <<= n as u32;
        self.count = self.count.saturating_sub(n as u32);
    }

    /// Refill and return the top 32 bits for decoding.
    #[inline(always)]
    pub fn peek_refilled(&mut self) -> u32 {
        self.refill();
        self.bits.top32()
    }

    /// Get the number of bits currently available.
    #[inline(always)]
    pub fn available(&self) -> u32 {
        self.count
    }

    /// Get the current position in the source buffer.
    #[inline(always)]
    pub fn position(&self) -> usize {
        self.pos
    }
}

/// Type alias for 32-bit bit reader (matches C++ implementation)
pub type BitReader32<'a> = BitReader<'a, u32>;

/// Type alias for 64-bit bit reader (fewer refills than 32-bit)
pub type BitReader64<'a> = BitReader<'a, u64>;

/// Specialized implementation for 64-bit reader with bulk loading
impl<'a> BitReader<'a, u64> {
    /// Bulk refill: load 64 bits at once when container is nearly empty.
    /// Falls back to 16-bit loads when not enough data available.
    #[inline(always)]
    pub fn refill_bulk(&mut self) {
        // Only do bulk load when we have room for 64 bits and enough data
        if self.count == 0 && self.pos + 8 <= self.src.len() {
            // Load 8 bytes at once
            let bytes: [u8; 8] = unsafe {
                *self
                    .src
                    .get_unchecked(self.pos..self.pos + 8)
                    .as_ptr()
                    .cast()
            };

            // Transform from LE memory layout to MSB-first bit container
            // Memory: [b0, b1, b2, b3, b4, b5, b6, b7]
            // LE u64: 0x_b7_b6_b5_b4_b3_b2_b1_b0
            // Want:   0x_b1_b0_b3_b2_b5_b4_b7_b6 (LE words in MSB-first order)
            let v = u64::from_ne_bytes(bytes).swap_bytes();
            // Swap adjacent bytes within each 16-bit pair
            let v = ((v & 0xFF00FF00FF00FF00) >> 8) | ((v & 0x00FF00FF00FF00FF) << 8);

            self.bits = v;
            self.count = 64;
            self.pos += 8;
        } else if self.count <= 48 && self.pos + 2 <= self.src.len() {
            // Fall back to regular refill for partial loads
            self.refill();
        }
    }

    /// Refill (bulk) and return the top 32 bits for decoding.
    #[inline(always)]
    pub fn peek_refilled_bulk(&mut self) -> u32 {
        self.refill_bulk();
        self.bits.top32()
    }
}

/// Type alias for 128-bit bit reader (even fewer refills)
pub type BitReader128<'a> = BitReader<'a, u128>;

/// Type alias for 256-bit bit reader (minimal refills)
pub type BitReader256<'a> = BitReader<'a, U256>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitreader32_empty() {
        let data: &[u8] = &[];
        let reader = BitReader32::new(data);
        assert_eq!(reader.available(), 0);
        assert_eq!(reader.position(), 0);
    }

    #[test]
    fn test_bitreader32_single_word() {
        // Little-endian: 0x01, 0x02 -> word = 0x0201
        // Placed at bits 16-31: bits = 0x0201_0000
        let data: &[u8] = &[0x01, 0x02];
        let reader = BitReader32::new(data);
        assert_eq!(reader.available(), 16);
        assert_eq!(reader.peek(), 0x0201_0000);
    }

    #[test]
    fn test_bitreader32_two_words() {
        // word1 = 0x0201, word2 = 0x0403
        // After loading both: bits = 0x0201_0403
        let data: &[u8] = &[0x01, 0x02, 0x03, 0x04];
        let reader = BitReader32::new(data);
        assert_eq!(reader.available(), 32);
        assert_eq!(reader.peek(), 0x0201_0403);
    }

    #[test]
    fn test_bitreader32_consume() {
        let data: &[u8] = &[0xFF, 0x00, 0xAA, 0x55];
        let mut reader = BitReader32::new(data);

        // Initial: 0x00FF_55AA
        assert_eq!(reader.peek(), 0x00FF_55AA);

        // Consume 8 bits: shift left by 8
        reader.consume(8);
        assert_eq!(reader.peek(), 0xFF55_AA00);
        assert_eq!(reader.available(), 24);

        // Consume 4 more bits
        reader.consume(4);
        assert_eq!(reader.peek(), 0xF55A_A000);
        assert_eq!(reader.available(), 20);
    }

    #[test]
    fn test_bitreader32_refill_after_consume() {
        let data: &[u8] = &[0x01, 0x02, 0x03, 0x04, 0x05, 0x06];
        let mut reader = BitReader32::new(data);

        // Initial: 32 bits loaded (0x0201_0403)
        assert_eq!(reader.available(), 32);
        assert_eq!(reader.position(), 4);

        // Consume 20 bits -> 12 bits remain
        reader.consume(20);
        assert_eq!(reader.available(), 12);

        // Refill should add 16 more bits (word 0x0605)
        reader.refill();
        assert_eq!(reader.available(), 28);
        assert_eq!(reader.position(), 6);
    }

    #[test]
    fn test_bitreader32_escape_detection() {
        // 0xF8 at MSB means escape code
        let data: &[u8] = &[0x00, 0xF8, 0x00, 0x00];
        let reader = BitReader32::new(data);
        assert!(reader.is_escape());

        // 0xF7 is not escape
        let data: &[u8] = &[0x00, 0xF7, 0x00, 0x00];
        let reader = BitReader32::new(data);
        assert!(!reader.is_escape());
    }

    #[test]
    fn test_bitreader64_single_word() {
        // word = 0x0201, placed at bits 48-63
        let data: &[u8] = &[0x01, 0x02];
        let reader = BitReader64::new(data);
        assert_eq!(reader.available(), 16);
        // top32() extracts bits 32-63, so word is at bits 16-31 of that u32
        assert_eq!(reader.peek(), 0x0201_0000);
    }

    #[test]
    fn test_bitreader64_four_words() {
        // Load up to 64 bits
        let data: &[u8] = &[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        let reader = BitReader64::new(data);
        assert_eq!(reader.available(), 64);
        // Top 32 bits should be first two words: 0x0201_0403
        assert_eq!(reader.peek(), 0x0201_0403);
    }

    #[test]
    fn test_bitreader64_consume_and_refill() {
        let data: &[u8] = &[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A];
        let mut reader = BitReader64::new(data);

        // Initial: 64 bits loaded
        assert_eq!(reader.available(), 64);
        assert_eq!(reader.position(), 8);

        // Consume 32 bits
        reader.consume(32);
        assert_eq!(reader.available(), 32);

        // Now top32 should be the second pair of words: 0x0605_0807
        assert_eq!(reader.peek(), 0x0605_0807);

        // Refill should add the last word
        reader.refill();
        assert_eq!(reader.available(), 48);
    }

    #[test]
    fn test_bitreader64_escape_detection() {
        // For 64-bit, escape is when top bits are >= 0xF800...
        // We need 0xF8 to appear at bit position 63-56
        let data: &[u8] = &[0x00, 0xF8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let reader = BitReader64::new(data);
        assert!(reader.is_escape());
    }

    #[test]
    fn test_bitreader32_matches_bitreader64_initial() {
        // Both should produce the same initial peek() value
        let data: &[u8] = &[0xAB, 0xCD, 0xEF, 0x12, 0x34, 0x56, 0x78, 0x9A];

        let r32 = BitReader32::new(data);
        let r64 = BitReader64::new(data);

        // Initial peek should match (both have loaded at least 32 bits)
        assert_eq!(r32.peek(), r64.peek());
    }

    #[test]
    fn test_bitreader_consume_exact_bits() {
        // Test that consuming specific bits produces expected results
        let data: &[u8] = &[0x12, 0x34, 0x56, 0x78];
        let mut reader = BitReader32::new(data);

        // Initial: 0x3412_7856
        let initial = reader.peek();
        assert_eq!(initial, 0x3412_7856);

        // Consume top 12 bits (0x341), remaining should start with 0x2...
        reader.consume(12);
        let after_12 = reader.peek();
        // After shifting left by 12: 0x27856000
        assert_eq!(after_12, 0x2785_6000);
    }

    // === 128-bit tests ===

    #[test]
    fn test_bitreader128_single_word() {
        let data: &[u8] = &[0x01, 0x02];
        let reader = BitReader128::new(data);
        assert_eq!(reader.available(), 16);
        // top32() extracts bits 96-127, word at bits 112-127 appears at bits 16-31 of u32
        assert_eq!(reader.peek(), 0x0201_0000);
    }

    #[test]
    fn test_bitreader128_full_load() {
        // 16 bytes = 128 bits
        let data: &[u8] = &[
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E,
            0x0F, 0x10,
        ];
        let reader = BitReader128::new(data);
        assert_eq!(reader.available(), 128);
        // Top 32 bits should be first two words: 0x0201_0403
        assert_eq!(reader.peek(), 0x0201_0403);
    }

    #[test]
    fn test_bitreader128_matches_others_initial() {
        let data: &[u8] = &[0xAB, 0xCD, 0xEF, 0x12, 0x34, 0x56, 0x78, 0x9A];

        let r32 = BitReader32::new(data);
        let r64 = BitReader64::new(data);
        let r128 = BitReader128::new(data);

        // All should produce the same initial peek
        assert_eq!(r32.peek(), r64.peek());
        assert_eq!(r64.peek(), r128.peek());
    }

    // === 256-bit tests ===

    #[test]
    fn test_u256_shift_left() {
        let v = U256 { hi: 0, lo: 1 };

        // Shift by 0
        let shifted = v << 0;
        assert_eq!(shifted.hi, 0);
        assert_eq!(shifted.lo, 1);

        // Shift by 64
        let shifted = v << 64;
        assert_eq!(shifted.hi, 0);
        assert_eq!(shifted.lo, 1 << 64);

        // Shift by 128 (crosses boundary)
        let shifted = v << 128;
        assert_eq!(shifted.hi, 1);
        assert_eq!(shifted.lo, 0);

        // Shift by 192
        let shifted = v << 192;
        assert_eq!(shifted.hi, 1 << 64);
        assert_eq!(shifted.lo, 0);
    }

    #[test]
    fn test_bitreader256_single_word() {
        let data: &[u8] = &[0x01, 0x02];
        let reader = BitReader256::new(data);
        assert_eq!(reader.available(), 16);
        assert_eq!(reader.peek(), 0x0201_0000);
    }

    #[test]
    fn test_bitreader256_full_load() {
        // 32 bytes = 256 bits
        let data: &[u8] = &[
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E,
            0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C,
            0x1D, 0x1E, 0x1F, 0x20,
        ];
        let reader = BitReader256::new(data);
        assert_eq!(reader.available(), 256);
        // Top 32 bits: first two words 0x0201_0403
        assert_eq!(reader.peek(), 0x0201_0403);
    }

    #[test]
    fn test_bitreader256_matches_others_initial() {
        let data: &[u8] = &[
            0xAB, 0xCD, 0xEF, 0x12, 0x34, 0x56, 0x78, 0x9A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00,
        ];

        let r32 = BitReader32::new(data);
        let r64 = BitReader64::new(data);
        let r128 = BitReader128::new(data);
        let r256 = BitReader256::new(data);

        // All should produce the same initial peek
        assert_eq!(r32.peek(), r64.peek());
        assert_eq!(r64.peek(), r128.peek());
        assert_eq!(r128.peek(), r256.peek());
    }

    #[test]
    fn test_bitreader256_consume_and_refill() {
        let data: &[u8] = &[
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E,
            0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C,
            0x1D, 0x1E, 0x1F, 0x20, 0x21, 0x22, 0x23, 0x24, // Extra data for refill
        ];
        let mut reader = BitReader256::new(data);

        // Initial: 256 bits loaded
        assert_eq!(reader.available(), 256);
        assert_eq!(reader.position(), 32);

        // Consume 128 bits
        reader.consume(128);
        assert_eq!(reader.available(), 128);

        // Refill should add more
        reader.refill();
        assert!(reader.available() > 128);
    }
}
