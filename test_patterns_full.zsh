#!/usr/bin/env zsh

echo "Decoding patterns file and comparing full output..."

# Decode the file
./target/release/llic_rs2 decode test_data/patterns_64x64_q0.llic /tmp/patterns_decoded.pgm

# Compare specific rows
echo -e "\nOriginal Row 3:"
xxd -s 205 -l 64 test_data/patterns_64x64.pgm

echo -e "\nDecoded Row 3:"
xxd -s 205 -l 64 /tmp/patterns_decoded.pgm

echo -e "\nOriginal Row 4:"
xxd -s 269 -l 64 test_data/patterns_64x64.pgm

echo -e "\nDecoded Row 4:"
xxd -s 269 -l 64 /tmp/patterns_decoded.pgm

# Find first difference
echo -e "\nFinding first difference..."
cmp -l test_data/patterns_64x64.pgm /tmp/patterns_decoded.pgm | head -5