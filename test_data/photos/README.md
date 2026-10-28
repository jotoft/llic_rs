# Test photos

Color photos developed from camera raw files, for benchmarks, color helper
tests and the web demo. Developed straight from the raw data instead of from
JPEGs, so they carry real sensor noise and no JPEG block artifacts. Grayscale
versions are derived in code (BT.601 luma, as the C++ `llic_compress` does).

| File | Original | Taken | Camera and settings |
|---|---|---|---|
| `rattan_*.webp` ("Rattan Weave" in the demo) | `DSCF4309.RAF` | 2024-01-21 14:32:10 | Fujifilm X-T2, XF18-55mm at 55 mm, f/4, 1/15 s, ISO 1600 |
| `rock_*.webp` ("Rock Formation" in the demo) | `DSCF0233.RAF` | 2020-07-21 15:07:02 | Fujifilm X-T2, XF18-55mm at 50.5 mm, f/11, 1/400 s, ISO 200 |

All files are lossless WebP, 8-bit sRGB, center-cropped to 3:2. The X-T2 has an
X-Trans sensor (6x6 color filter pattern), not Bayer, so Bayer and RGB-IR test
mosaics have to be made by re-mosaicing these images.

## Regenerating

The originals are not in git. Keep them in `test_data/raw/` (gitignored), then:

```bash
uv run scripts/develop_photos.py                                  # these files
uv run scripts/develop_photos.py --sizes full --out test_data/out # full size, not committed
```

The script pins rawpy 0.27.1 (LibRaw 0.22.1), Pillow and numpy, and runs LibRaw
single-threaded, so it reproduces these files byte for byte.

## Finding the originals

They are in PhotoDB (bucket `photodbjt` on Backblaze B2, region
`eu-central-003`), under these object keys:

| Original | B2 object key | SHA-256 |
|---|---|---|
| `DSCF4309.RAF` | `2024/01/20240121_143210_00001977.RAF` | `1d1fa10d8e483e9aba1ada6655a6dad2dd5442d29c57f0d36012998931b347f9` |
| `DSCF0233.RAF` | `2020/07/20200721_130702_00005159.RAF` | `becb49b996103f2ba88a66fa86c62d30a7e254adffcf1c04d6a1612679e1b6e6` |

Save them in `test_data/raw/` under the key's file name (for example
`20240121_143210_00001977.RAF`). Any S3 client works against
`https://s3.eu-central-003.backblazeb2.com`, with the keys in
`~/.config/photodb/.env`.

If the keys change, find the files again with PhotoDB's semantic search
(`GET /api/search?q=...` on the PhotoDB server, with the CLI's API token). The
queries "close-up of woven rattan texture" and "rock formation landscape"
return them at or near the top. Check the capture time and exposure in the
EXIF against the table above.
