"""
Run inference with your trained YOLO11 classification model.

Usage:
    .\\.venv\\Scripts\\python.exe predict.py

For each test image it prints the top predicted class and confidence, and saves
annotated images under runs/classify/predict/.
"""
from pathlib import Path
from ultralytics import YOLO

# Anchor paths to this script's folder so it runs from any working directory.
SCRIPT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WEIGHTS = SCRIPT_DIR / "runs" / "classify" / "train" / "weights" / "best.pt"  # trained weights
SOURCE = SCRIPT_DIR / "dataset" / "test"  # folder with class subfolders
IMG_SIZE = 224
# ---------------------------------------------------------------------------


def main():
    weights = Path(WEIGHTS)
    if not weights.exists():
        raise FileNotFoundError(
            f"Could not find trained weights at {weights.resolve()}.\n"
            "Train the model first with train.py."
        )

    model = YOLO(str(weights))

    # Classification test sets have images inside class subfolders (test/SUV/,
    # test/VAN/ etc.). Collect all image paths so YOLO can process them.
    source_dir = Path(SOURCE)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    image_files = sorted(
        p for p in source_dir.rglob("*") if p.suffix.lower() in image_exts
    )
    if not image_files:
        raise FileNotFoundError(
            f"No images found in {source_dir.resolve()} or its subfolders."
        )

    results = model.predict(
        source=[str(p) for p in image_files],
        imgsz=IMG_SIZE,
        save=True,
        project=str(SCRIPT_DIR / "runs" / "classify"),
        name="predict",
    )

    # Print the top class and its confidence per image
    for r in results:
        top_idx = int(r.probs.top1)
        top_conf = float(r.probs.top1conf)
        print(f"{Path(r.path).name} -> {r.names[top_idx]} ({top_conf:.2f})")
    print("\nAnnotated images saved under runs/classify/predict/")


if __name__ == "__main__":
    main()
