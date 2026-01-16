/// Decompression table entry structure
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DecompressEntry {
    pub bits: u8,
    pub num_symbols: u8,
    pub symbols: [u8; 2],
}

/// Decompression lookup table with 4096 entries
/// Indexed by the first 12 bits of compressed data
pub static DECOMPRESS_TABLE: &[DecompressEntry; 4096] = unsafe {
    &*(include_bytes!("../../tables/u8v1_decompress_table_2x.bin").as_ptr()
        as *const [DecompressEntry; 4096])
};

/// Compression lookup table with 65536 entries
/// Indexed by two consecutive bytes: byte1 | (byte2 << 8)
/// Each entry is a 32-bit value where:
/// - Upper 26 bits: Variable-length codes (left-justified)
/// - Lower 6 bits: Total number of bits for both codes
pub static COMPRESS_TABLE_BYTES: &[u8] = include_bytes!("../../tables/u8v1_compress_table_2x.bin");

pub fn get_compress_table_entry(index: usize) -> u32 {
    let offset = index * 4;
    u32::from_le_bytes([
        COMPRESS_TABLE_BYTES[offset],
        COMPRESS_TABLE_BYTES[offset + 1],
        COMPRESS_TABLE_BYTES[offset + 2],
        COMPRESS_TABLE_BYTES[offset + 3],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_table_sizes() {
        assert_eq!(std::mem::size_of::<DecompressEntry>(), 4);
        assert_eq!(std::mem::size_of_val(DECOMPRESS_TABLE), 16384);
        assert_eq!(COMPRESS_TABLE_BYTES.len(), 262144);
    }

    #[test]
    fn test_decompress_table_access() {
        // Test that we can access the table without panicking
        let entry = DECOMPRESS_TABLE[0];
        assert!(entry.bits > 0);
        assert!(entry.num_symbols > 0 && entry.num_symbols <= 2);
    }

    #[test]
    fn test_compress_table_access() {
        // Test that we can access the table without panicking
        let entry = get_compress_table_entry(0);
        let num_bits = entry & 0x3F;
        assert!((4..=32).contains(&num_bits));
    }
}
