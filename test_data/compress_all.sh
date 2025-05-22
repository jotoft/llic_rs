#!/bin/bash

LLIC_COMPRESS="/home/jtoft/kod/llic_rs2/llic/build/src/llic_compress/llic_compress"

# Test if llic_compress exists
if [ ! -x "$LLIC_COMPRESS" ]; then
    echo "Error: llic_compress not found at $LLIC_COMPRESS"
    exit 1
fi

# Compress each PGM file with quality levels 0-4
for pgm in *.pgm; do
    if [ -f "$pgm" ]; then
        base="${pgm%.pgm}"
        echo "Compressing $pgm..."
        
        for quality in 0 1 2 3 4; do
            output="${base}_q${quality}.llic"
            echo "  Quality $quality -> $output"
            "$LLIC_COMPRESS" c "$pgm" "$output" "$quality"
        done
    fi
done

echo "Compression complete!"