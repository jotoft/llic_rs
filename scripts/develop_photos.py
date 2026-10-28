# /// script
# requires-python = ">=3.10"
# dependencies = ["rawpy==0.27.1", "pillow==13.0.0", "numpy==2.5.4"]
# ///
"""Develop the camera originals in test_data/raw into the test photos.

Each RAF is demosaiced by LibRaw with the camera white balance into 8-bit
sRGB, center-cropped to 3:2, resized with Lanczos, and saved as lossless
WebP. See test_data/photos/README.md for where the originals come from.

    uv run scripts/develop_photos.py                         # committed sizes
    uv run scripts/develop_photos.py --sizes full --out test_data/out
"""

import argparse
import os
from pathlib import Path

# LibRaw's parallel X-Trans demosaicing is not deterministic (a few pixels
# differ by 1-3 between runs); one thread reproduces the files exactly. Must be
# set before LibRaw loads its OpenMP runtime.
os.environ["OMP_NUM_THREADS"] = "1"

import rawpy  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Output name -> original file name in test_data/raw (and B2).
PHOTOS = {
    "rattan": "20240121_143210_00001977.RAF",
    "rock": "20200721_130702_00005159.RAF",
}


def develop(raf: Path) -> Image.Image:
    with rawpy.imread(str(raf)) as raw:
        rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
    return Image.fromarray(rgb)


def crop_3_2(img: Image.Image) -> Image.Image:
    """Center-crop to 3:2 with both sides a multiple of 12, so that every
    3:2 size with sides divisible by 4 is reachable."""
    unit = min(img.width // 12, img.height // 8)
    width, height = unit * 12, unit * 8
    left, top = (img.width - width) // 2, (img.height - height) // 2
    return img.crop((left, top, left + width, top + height))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", nargs="+", default=["1920", "960"],
                        help="output widths in pixels, or 'full' (default: 1920 960)")
    parser.add_argument("--out", type=Path, default=ROOT / "test_data/photos")
    parser.add_argument("--raw", type=Path, default=ROOT / "test_data/raw")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for name, raf_name in PHOTOS.items():
        img = crop_3_2(develop(args.raw / raf_name))
        for size in args.sizes:
            if size == "full":
                out = img
            else:
                width = int(size)
                if width % 12:
                    parser.error(f"width {width} must be a multiple of 12 for a 3:2 image "
                                 "with sides divisible by 4")
                out = img.resize((width, width * 2 // 3), Image.LANCZOS)
            path = args.out / f"{name}_{out.width}x{out.height}.webp"
            out.save(path, lossless=True, quality=100, method=6)
            print(f"{path}  {path.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
