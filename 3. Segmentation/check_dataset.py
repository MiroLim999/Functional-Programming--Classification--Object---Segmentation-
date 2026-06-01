"""
Validate a YOLO instance-segmentation dataset exported from Roboflow.

Segmentation datasets use the same layout as detection (data.yaml + per-image
label files), but each label line is a polygon:

    class x1 y1 x2 y2 x3 y3 ...   (an odd number of values: 1 class id + pairs)

Checks:
  - data.yaml exists and is readable
  - the train/valid/test splits referenced by data.yaml exist
  - each split has matching images/ and labels/ folders
  - every image has a corresponding label file (and flags empties/orphans)
  - label files are well-formed polygons (class id + >=3 coordinate pairs,
    class id in range, coords in 0..1)
  - reports class counts per split

Usage:
    python check_dataset.py
    python check_dataset.py --data path/to/data.yaml

Exit code is 0 if the dataset looks usable, 1 if a blocking problem was found.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml

# Anchor default paths to this script's folder so it works regardless of the
# current working directory (terminal, VS Code "Run" button, etc.).
SCRIPT_DIR = Path(__file__).resolve().parent

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


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


def load_data_yaml(data_path: Path, rep: Reporter) -> dict | None:
    if not data_path.exists():
        rep.error(
            f"data.yaml not found at {data_path.resolve()}. "
            "Export your dataset from Roboflow as 'YOLOv11' "
            "(instance segmentation) and unzip it here."
        )
        return None
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:  # noqa: BLE001
        rep.error(f"Could not parse {data_path}: {e}")
        return None
    rep.ok(f"Loaded {data_path}")
    return cfg


def get_class_names(cfg: dict, rep: Reporter) -> list[str]:
    names = cfg.get("names")
    if names is None:
        rep.error("data.yaml has no 'names' entry (class list).")
        return []
    if isinstance(names, dict):  # {0: 'a', 1: 'b'}
        names = [names[k] for k in sorted(names)]
    if not isinstance(names, list) or not names:
        rep.error("'names' in data.yaml is empty or malformed.")
        return []
    rep.ok(f"Found {len(names)} class(es): {names}")
    return list(names)


def resolve_split_dir(data_path: Path, cfg: dict, key: str) -> Path | None:
    """Resolve a split path from data.yaml across common Roboflow layouts."""
    raw = cfg.get(key)
    if not raw:
        return None

    raw_path = Path(raw)
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        # Normal case: relative to data.yaml directory.
        candidates.append((data_path.parent / raw_path).resolve())
        # Some Roboflow exports include "../train/images" even when splits are
        # inside the same dataset/ folder as data.yaml.
        while raw_path.parts and raw_path.parts[0] == "..":
            raw_path = Path(*raw_path.parts[1:])
        if raw_path.parts:
            candidates.append((data_path.parent / raw_path).resolve())

    for p in candidates:
        # Roboflow points split paths to images/.
        split_root = p.parent if p.name == "images" else p
        if split_root.exists():
            return split_root

    # Fallback to first candidate for clearer downstream error paths.
    p = candidates[0]
    return p.parent if p.name == "images" else p


def split_image_label_dirs(split_dir: Path) -> tuple[Path, Path]:
    return split_dir / "images", split_dir / "labels"


def validate_label_file(
    label_file: Path, num_classes: int, rep: Reporter
) -> tuple[int, bool]:
    """Validate one segmentation label file.

    Each non-empty line must be: class id followed by an even number of
    coordinate values (x y pairs), at least 3 pairs to form a polygon.
    Returns (num_polygons, had_error).
    """
    n_polys = 0
    had_error = False
    try:
        lines = label_file.read_text(encoding="utf-8").splitlines()
    except Exception as e:  # noqa: BLE001
        rep.error(f"Cannot read label file {label_file.name}: {e}")
        return 0, True

    for ln, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        # polygon: class + at least 3 (x, y) pairs => 1 + 6 = 7 values minimum
        if len(parts) < 7:
            rep.error(
                f"{label_file.name}:{ln} has fewer than 7 values "
                "(need class + >=3 x/y pairs for a polygon)."
            )
            had_error = True
            continue
        coord_count = len(parts) - 1
        if coord_count % 2 != 0:
            rep.error(
                f"{label_file.name}:{ln} has an odd number of coordinates "
                "(x/y pairs must come in twos)."
            )
            had_error = True
            continue
        try:
            cls = int(float(parts[0]))
            coords = [float(x) for x in parts[1:]]
        except ValueError:
            rep.error(f"{label_file.name}:{ln} has non-numeric values.")
            had_error = True
            continue
        if cls < 0 or cls >= num_classes:
            rep.error(
                f"{label_file.name}:{ln} class id {cls} out of range "
                f"(0..{num_classes - 1})."
            )
            had_error = True
        if any(c < 0 or c > 1 for c in coords):
            rep.warn(
                f"{label_file.name}:{ln} has coords outside 0..1 "
                "(not normalized?)."
            )
        n_polys += 1
    return n_polys, had_error


def check_split(
    split_dir: Path, name: str, num_classes: int, rep: Reporter
) -> Counter:
    """Validate one split folder. Returns class-id counts for the split."""
    print(f"\n[{name}]")
    class_counter: Counter = Counter()

    if split_dir is None or not split_dir.exists():
        rep.warn(f"split '{name}' not found (skipping).")
        return class_counter

    img_dir, lbl_dir = split_image_label_dirs(split_dir)
    if not img_dir.exists():
        rep.error(f"{name}: missing images/ folder at {img_dir}")
        return class_counter
    if not lbl_dir.exists():
        rep.error(f"{name}: missing labels/ folder at {lbl_dir}")
        return class_counter

    images = [p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    if not images:
        rep.error(f"{name}: no images found in {img_dir}")
        return class_counter

    missing_labels = 0
    empty_labels = 0
    label_errors = 0
    for img in images:
        label_file = lbl_dir / (img.stem + ".txt")
        if not label_file.exists():
            missing_labels += 1
            continue
        n_polys, had_error = validate_label_file(label_file, num_classes, rep)
        if had_error:
            label_errors += 1
        if n_polys == 0:
            empty_labels += 1
        else:
            # re-read to tally class ids (cheap; files are small)
            for line in label_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        class_counter[int(float(line.split()[0]))] += 1
                    except (ValueError, IndexError):
                        pass

    # orphan labels (label with no matching image)
    image_stems = {p.stem for p in images}
    orphan_labels = [
        p for p in lbl_dir.glob("*.txt") if p.stem not in image_stems
    ]

    rep.ok(f"{name}: {len(images)} image(s), {len(images) - missing_labels} labelled")
    if missing_labels:
        rep.warn(f"{name}: {missing_labels} image(s) have no label file.")
    if empty_labels:
        rep.warn(
            f"{name}: {empty_labels} label file(s) are empty "
            "(treated as background)."
        )
    if orphan_labels:
        rep.warn(f"{name}: {len(orphan_labels)} label file(s) have no image.")
    if label_errors:
        rep.error(f"{name}: {label_errors} label file(s) have format errors.")

    return class_counter


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a YOLO segmentation dataset."
    )
    parser.add_argument(
        "--data",
        default=str(SCRIPT_DIR / "dataset" / "data.yaml"),
        help="Path to data.yaml (default: <script dir>/dataset/data.yaml)",
    )
    args = parser.parse_args()

    rep = Reporter()
    data_path = Path(args.data)

    print("=== Dataset validation ===")
    cfg = load_data_yaml(data_path, rep)
    if cfg is None:
        _summary(rep)
        return 1

    names = get_class_names(cfg, rep)
    num_classes = len(names)
    if num_classes == 0:
        _summary(rep)
        return 1

    totals: Counter = Counter()
    for key, label in (("train", "train"), ("val", "valid"), ("test", "test")):
        split_dir = resolve_split_dir(data_path, cfg, key)
        # data.yaml sometimes uses 'val' key but folder is 'valid'
        if split_dir is None and key == "val":
            split_dir = resolve_split_dir(data_path, {"val": "valid/images"}, "val")
        counts = check_split(split_dir, label, num_classes, rep)
        totals.update(counts)

    # Per-class coverage summary
    print("\n[class coverage across all splits]")
    for idx, cname in enumerate(names):
        c = totals.get(idx, 0)
        marker = "OK" if c > 0 else "MISSING"
        line = f"  class {idx} ({cname}): {c} polygon(s)  [{marker}]"
        print(line)
        if c == 0:
            rep.warn(f"class '{cname}' has no annotations in any split.")

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
