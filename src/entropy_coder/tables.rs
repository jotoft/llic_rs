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

/// Compression lookup table raw bytes (65536 * 4 = 262144 bytes)
/// Indexed by two consecutive bytes: byte1 | (byte2 << 8)
/// Each entry is a 32-bit value where:
/// - Upper 26 bits: Variable-length codes (left-justified)
/// - Lower 6 bits: Total number of bits for both codes
static COMPRESS_TABLE_BYTES: &[u8; 262144] =
    include_bytes!("../../tables/u8v1_compress_table_2x.bin");

/// Get an entry from the compression table.
/// Uses unchecked access for performance (index is always valid from u16).
#[inline(always)]
pub fn get_compress_table_entry(index: usize) -> u32 {
    // SAFETY: index comes from u16 value (< 65536), offset is always < 262144
    let offset = index * 4;
    unsafe {
        let ptr = COMPRESS_TABLE_BYTES.as_ptr().add(offset) as *const u32;
        ptr.read_unaligned()
    }
}

/// Code of a single residual symbol, right-aligned, and its length in bits
/// (2 to 13), read from the pair table entry `(symbol, 0)` minus the 2-bit code
/// of the zero.
const fn symbol_code(symbol: usize) -> (u32, u32) {
    let b = COMPRESS_TABLE_BYTES;
    let i = symbol * 4;
    let entry = u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
    let len = (entry & 0x3F) - 2;
    (entry >> (32 - len), len)
}

/// The 13-bit lookahead decode table (the C++ `U8V1_DECOMPRESS_TABLE`),
/// aligned to a cache line.
#[repr(C, align(64))]
pub struct DecodeTable13(pub [u32; 1 << 13]);

/// Decode table indexed by the next 13 stream bits. Each entry decodes one to
/// three complete codes: bits 0-5 hold their total length, bits 6-7 the symbol
/// count, and bits 8-15, 16-23 and 24-31 the symbols in stream order. An escape
/// code (13 bits) is always decoded alone. Built at compile time by greedy
/// prefix decoding of the codes in the compress table, as the C++ table is.
pub static DECODE_TABLE_13: DecodeTable13 = DecodeTable13(build_decode_table_13());

/// First window of the 13-bit escape codes `11111 + literal`.
const ESCAPE_WINDOWS: usize = 0b11111 << 8;

const fn build_decode_table_13() -> [u32; 1 << 13] {
    const BITS: u32 = 13;
    const MASK: usize = (1 << BITS) - 1;
    // For each 13-bit window, the symbol whose code prefixes it and its length.
    // Escapes (11111 + literal) can code any byte, also those with a shorter
    // code, so all 256 escape windows decode; short codes never start 11111.
    let mut first = [0u16; 1 << BITS];
    let mut literal = 0;
    while literal < 256 {
        first[ESCAPE_WINDOWS + literal] = (literal as u16) | ((BITS as u16) << 8);
        literal += 1;
    }
    let mut symbol = 0;
    while symbol < 256 {
        let (code, len) = symbol_code(symbol);
        let start = (code << (BITS - len)) as usize;
        let end = ((code + 1) << (BITS - len)) as usize;
        let mut i = start;
        while i < end {
            first[i] = (symbol as u16) | ((len as u16) << 8);
            i += 1;
        }
        symbol += 1;
    }
    let mut table = [0u32; 1 << BITS];
    let mut window = 0;
    while window <= MASK {
        let (mut used, mut count, mut symbols) = (0u32, 0u32, 0u32);
        while count < 3 {
            // Bits past `used` are zeros here, so a match only counts if its
            // whole code lies inside the window.
            let next = first[(window << used) & MASK];
            let len = (next >> 8) as u32;
            if len == 0 || len > BITS - used {
                break;
            }
            symbols |= ((next & 0xFF) as u32) << (8 * count);
            count += 1;
            used += len;
        }
        // Every window starts with a code (see `every_window_starts_a_code`).
        assert!(count > 0);
        table[window] = used | (count << 6) | (symbols << 8);
        window += 1;
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The generated table equals the C++ reference's, when the submodule is
    /// checked out.
    #[test]
    fn decode_table_13_matches_cpp() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/llic/src/llic/entropycoder/u8v1_decompress_table.cpp"
        );
        let Ok(source) = std::fs::read_to_string(path) else {
            return;
        };
        let body = source
            .split("U8V1_DECOMPRESS_TABLE[1 << 13] = {")
            .nth(1)
            .expect("table");
        let cpp: Vec<u32> = body
            .split(['{', '}', ',', ';'])
            .filter_map(|v| v.trim().strip_prefix("0x"))
            .map(|v| u32::from_str_radix(v.trim_end_matches('u'), 16).expect("hex"))
            .collect();
        assert_eq!(cpp.len(), 1 << 13);
        for (i, (&ours, &theirs)) in DECODE_TABLE_13.0.iter().zip(&cpp).enumerate() {
            assert_eq!(ours, theirs, "entry {i:#x}");
        }
    }

    /// The short codes plus the 256 escape windows cover every 13-bit window
    /// exactly once, so each table entry decodes at least one symbol.
    #[test]
    fn every_window_starts_a_code() {
        let short: u64 = (0..256)
            .map(|s| symbol_code(s).1)
            .filter(|&len| len < 13)
            .map(|len| 1u64 << (13 - len))
            .sum();
        assert_eq!(short + 256, 1 << 13);
        assert!((0..256).all(|s| symbol_code(s).1 == 13
            || (symbol_code(s).0 << (13 - symbol_code(s).1)) < ESCAPE_WINDOWS as u32));
    }

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
