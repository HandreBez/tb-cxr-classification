"""
One-time housekeeping script: pre-resize source images to 224x224 grayscale
and cache them to disk, so training doesn't pay the decode/resize cost fresh
every epoch.

Run this once:
    python cache_images.py

After running, data.py's collect_images() calls should point at the
data/cache/... folders below instead of the original data/Shenzhen /
data/Montgomery folders. Update the two path lines at the bottom of data.py:

    shenzhen_images = collect_images(Path("data/cache/Shenzhen"))
    montgomery_images = collect_images(Path("data/cache/Montgomery"))

Cached images are saved as PNG (lossless), already resized to 224x224 and
converted to grayscale ("L" mode) — matching what get_transforms() would
have produced right before ToTensor()/Normalize() anyway, so nothing about
the augmentation pipeline (RandomCrop, ColorJitter, etc., which still run
per-epoch in TBXrayDataset) changes. Only the expensive decode-of-huge-file +
initial resize step is done once instead of every epoch.
"""

from pathlib import Path
from PIL import Image

SOURCE_DIRS = {
    "Shenzhen": Path("data/Shenzhen/images/images"),
    "Montgomery": Path("data/Montgomery/images/images"),
}

CACHE_ROOT = Path("data/cache")
TARGET_SIZE = (224, 224)


def cache_folder(name, source_dir):
    dest_dir = CACHE_ROOT / name
    dest_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for file_path in source_dir.glob("*.png"):
        dest_path = dest_dir / file_path.name
        if dest_path.exists():
            continue  # already cached, skip

        image = Image.open(file_path).convert("L")
        image = image.resize(TARGET_SIZE, Image.BILINEAR)
        image.save(dest_path)
        count += 1

    print(f"{name}: cached {count} new images -> {dest_dir}")


if __name__ == "__main__":
    for name, source_dir in SOURCE_DIRS.items():
        cache_folder(name, source_dir)
    print("Done.")