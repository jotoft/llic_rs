#!/usr/bin/env zsh

# Create output directory
mkdir -p test_data/out

# Build the Rust project in release mode for better performance
echo "Building llic_rs2..."
cargo build --release

# Check if build succeeded
if [[ $? -ne 0 ]]; then
    echo "Error: Failed to build the project"
    exit 1
fi

# Find all .llic files in test_data
echo "Decoding LLIC files..."
for llic_file in test_data/*.llic; do
    # Skip if no files found
    [[ -f "$llic_file" ]] || continue
    
    # Get base filename without extension
    base_name=$(basename "$llic_file" .llic)
    
    # Output PGM file path
    output_file="test_data/out/${base_name}.pgm"
    
    echo "Decoding: $llic_file -> $output_file"
    
    # Run the decoder
    ./target/release/llic_rs2 decode "$llic_file" "$output_file"
    
    # Check if decode succeeded
    if [[ $? -eq 0 ]]; then
        echo "  ✓ Success"
    else
        echo "  ✗ Failed"
    fi
done

echo ""
echo "Decoding complete. Check test_data/out/ for results."
echo "You can compare with original PGM files using:"
echo "  diff test_data/gradient_8x8.pgm test_data/out/gradient_8x8_lossless.pgm"