#!/usr/bin/env zsh

# Test with the single-threaded version first
echo "Testing single-threaded patterns file..."
llic/build/src/llic_compress/llic_compress c test_data/patterns_64x64.pgm /tmp/patterns_single.llic 0
./target/release/llic_rs2 decode /tmp/patterns_single.llic /tmp/patterns_single_decoded.pgm

echo "Comparing single-threaded result..."
if diff test_data/patterns_64x64.pgm /tmp/patterns_single_decoded.pgm; then
    echo "Single-threaded: OK"
else
    echo "Single-threaded: FAILED"
fi

echo ""
echo "Original multi-threaded file info:"
xxd -l 100 test_data/patterns_64x64_q0.llic | head -10