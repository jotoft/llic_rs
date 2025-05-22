#!/usr/bin/env python3
import re

# Read the C++ file
with open('llic/src/llic/entropycoder/u8v1_decompress_table.cpp', 'r') as f:
    content = f.read()

# Extract all table entries
pattern = r'\{(\d+)u, (\d+)u, \{(\d+)u, (\d+)u\}\}'
matches = re.findall(pattern, content)

# Convert to Rust format
rust_entries = []
for match in matches:
    bits, num_symbols, sym0, sym1 = match
    rust_entries.append(f'    DecompressTable2x {{ bits: {bits}, num_symbols: {num_symbols}, symbol: [{sym0}, {sym1}] }}')

# Create Rust file content
rust_content = '''//-----------------------------------------------------------------------------
// Copyright (C) 2024 Peter Rundberg
//
// This software is provided 'as-is', without any express or implied
// warranty.  In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. If you use this software
//    in a product, an acknowledgment in the product documentation would be
//    appreciated but is not required.
// 2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.
//-----------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct DecompressTable2x {
    pub bits: u8,
    pub num_symbols: u8,
    pub symbol: [u8; 2],
}

pub const U8V1_DECOMPRESS_TABLE_2X: [DecompressTable2x; 4096] = [
'''

# Add entries in groups of 1 per line for readability
for entry in rust_entries:
    rust_content += entry + ',\n'

rust_content += '];\n'

# Write the Rust file
with open('src/entropy_coder/decompress_table.rs', 'w') as f:
    f.write(rust_content)

print(f"Converted {len(rust_entries)} entries to Rust format")