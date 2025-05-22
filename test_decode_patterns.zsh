#!/usr/bin/env zsh

# Let's test different sized images to find where the issue starts
echo "Testing different image sizes..."

# 32x32 test
echo -n "Creating 32x32 gradient... "
cat > /tmp/gradient_32x32.pgm << 'EOF'
P5
32 32
255
EOF
# Generate 32x32 gradient data
perl -e 'for($i=0;$i<1024;$i++){print chr($i%256)}' >> /tmp/gradient_32x32.pgm
echo "done"

echo -n "Compressing 32x32... "
llic/build/src/llic_compress/llic_compress c /tmp/gradient_32x32.pgm /tmp/gradient_32x32.llic 0
echo "done"

echo -n "Decompressing 32x32... "
./target/release/llic_rs2 decode /tmp/gradient_32x32.llic /tmp/gradient_32x32_out.pgm
echo "done"

echo -n "Comparing 32x32... "
if diff /tmp/gradient_32x32.pgm /tmp/gradient_32x32_out.pgm > /dev/null 2>&1; then
    echo "OK"
else
    echo "FAILED"
    echo "First few differences:"
    xxd /tmp/gradient_32x32.pgm | head -20 > /tmp/orig.hex
    xxd /tmp/gradient_32x32_out.pgm | head -20 > /tmp/out.hex
    diff /tmp/orig.hex /tmp/out.hex | head -10
fi