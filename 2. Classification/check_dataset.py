"""
Validate a YOLO image-classification dataset before training.

Classification datasets are folder-based (no data.yaml, no label files).
Each split contains one subfolder per class, and the images live inside:

    dataset/
    ├── train/
    │   ├── class_a/   (images)
    │   └── class_b/   (images)
    ├── valid/
    │   ├── class_a/
    │   └── class_b/
    └── test/
        ├── class_a/
        └── class_b/

Checks:
  - the dataset root exists
  - train/valid/test splits exist
  - each split has the same set of class subfolders
  - every class folder actually contains images
  - reports image counts per class per split

Usage:
    python check_dataset.py
    python check_dataset.py --data path/to/dataset

Exit code is 0 if the dataset looks usable, 1 if a blocking problem was found.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Anchor default paths to this script's folder so it works regardless of the
# current working directory (terminal, VS Code "Run" button, etc.).
SCRIPT_DIR = Path(__file__).resolve().parent

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "valid", "test")


class Reporter:
    """Collects errors (blocking) and warnings (non-blocking)."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)
        print(f"  [ERROR] {msg}")

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"  [WARN]  {msg}")

    def ok(self, msg: str) -> None:
        print(f"  [OK]    {msg}")


def class_dirs(split_dir: Path) -> list[Path]:
    """Return the class subfolders inside a split directory."""
    return sorted(p for p in split_dir.iterdir() if p.is_dir())


def count_images(class_dir: Path) -> int:
    return sum(
        1 for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )


def check_split(split_dir: Path, name: str, rep: Reporter) -> dict[str, int]:
    """Validate one split folder. Returns {class_name: image_count}."""
    print(f"\n[{name}]")
    counts: dict[str, int] = {}

    if not split_dir.exists():
        rep.warn(f"split '{name}' not found at {split_dir} (skipping).")
        return counts

    classes = class_dirs(split_dir)
    if not classes:
        rep.error(f"{name}: no class subfolders found in {split_dir}")
        return counts

    for class_dir in classes:
        n = count_images(class_dir)
        counts[class_dir.name] = n
        if n == 0:
            rep.warn(f"{name}/{class_dir.name}: no images found.")

    total = sum(counts.values())
    rep.ok(f"{name}: {len(classes)} class(es), {total} image(s)")
    for cname, n in counts.items():
        print(f"           - {cname}: {n} image(s)")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a YOLO classification dataset."
    )
    parser.add_argument(
        "--data",
        default=str(SCRIPT_DIR / "dataset"),
        help="Path to the dataset root (default: <script dir>/dataset)",
    )
    args = parser.parse_args()

    rep = Reporter()
    data_root = Path(args.data)

    print("=== Dataset validation ===")
    if not data_root.exists():
        rep.error(
            f"Dataset root not found at {data_root.resolve()}. "
            "Export your dataset from Roboflow as 'Folder Structure' "
            "(classification) and unzip it here."
        )
        return _summary(rep)
    rep.ok(f"Found dataset root: {data_root}")

    per_split: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        per_split[split] = check_split(data_root / split, split, rep)

    # Cross-split class consistency: train is the reference.
    train_classes = set(per_split.get("train", {}))
    if not train_classes:
        rep.error("train split has no classes; cannot train.")
        return _summary(rep)

    print("\n[class consistency]")
    for split in ("valid", "test"):
        split_classes = set(per_split.get(split, {}))
        if not split_classes:
            continue
        missing = train_classes - split_classes
        extra = split_classes - train_classes
        if missing:
            rep.warn(f"{split} is missing class(es) present in train: {sorted(missing)}")
        if extra:
            rep.warn(f"{split} has class(es) not in train: {sorted(extra)}")
        if not missing and not extra:
            rep.ok(f"{split} classes match train")

    print(f"\nClasses ({len(train_classes)}): {sorted(train_classes)}")
    return _summary(rep)


def _summary(rep: Reporter) -> int:
    print("\n=== Summary ===")
    print(f"  errors:   {len(rep.errors)}")
    print(f"  warnings: {len(rep.warnings)}")
    if rep.errors:
        print("\nDataset has blocking problems. Fix the errors above before training.")
        return 1
    if rep.warnings:
        print("\nDataset is usable but has warnings worth reviewing.")
    else:
        print("\nDataset looks good. Ready to train.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
