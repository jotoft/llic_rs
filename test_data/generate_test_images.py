#!/usr/bin/env python3
import numpy as np
import os

def write_pgm(filename, image):
    """Write a grayscale image to PGM format."""
    height, width = image.shape
    max_val = np.max(image)
    
    with open(filename, 'wb') as f:
        # Write PGM header
        f.write(b'P5\n')
        f.write(f'{width} {height}\n'.encode())
        f.write(f'{max_val}\n'.encode())
        # Write pixel data
        f.write(image.astype(np.uint8).tobytes())

# 1. 8x8 gradient image
gradient_8x8 = np.zeros((8, 8), dtype=np.uint8)
for i in range(8):
    for j in range(8):
        gradient_8x8[i, j] = int((i * 8 + j) * 255 / 63)
write_pgm('gradient_8x8.pgm', gradient_8x8)

# 2. 16x16 checkerboard pattern
checkerboard_16x16 = np.zeros((16, 16), dtype=np.uint8)
for i in range(16):
    for j in range(16):
        if (i // 4 + j // 4) % 2 == 0:
            checkerboard_16x16[i, j] = 255
        else:
            checkerboard_16x16[i, j] = 0
write_pgm('checkerboard_16x16.pgm', checkerboard_16x16)

# 3. 64x64 image with various patterns
patterns_64x64 = np.zeros((64, 64), dtype=np.uint8)
# Quadrant 1: Vertical stripes
for i in range(32):
    for j in range(32):
        patterns_64x64[i, j] = 255 if j % 4 < 2 else 0
# Quadrant 2: Horizontal stripes
for i in range(32):
    for j in range(32, 64):
        patterns_64x64[i, j] = 255 if i % 4 < 2 else 0
# Quadrant 3: Diagonal gradient
for i in range(32, 64):
    for j in range(32):
        patterns_64x64[i, j] = int(((i-32) + j) * 255 / 63)
# Quadrant 4: Random noise
np.random.seed(42)
patterns_64x64[32:64, 32:64] = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
write_pgm('patterns_64x64.pgm', patterns_64x64)

print("Test PGM images generated successfully!")