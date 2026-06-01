"""
Run inference with your trained YOLO11 instance segmentation model.

Usage:
    .\\.venv\\Scripts\\python.exe predict.py

Annotated images (with masks) are saved under runs/segment/predict/.
Polygon coordinates and confidences are also saved as .txt files.
"""
from pathlib import Path
from ultralytics import YOLO

# Anchor paths to this script's folder so it runs from any working directory.
SCRIPT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WEIGHTS = SCRIPT_DIR / "runs" / "segment" / "train" / "weights" / "best.pt"  # trained weights
SOURCE = SCRIPT_DIR / "dataset" / "test" / "images"  # folder, single image, or video
CONF = 0.25                                     # confidence threshold
IMG_SIZE = 640
# ---------------------------------------------------------------------------


def main():
    weights = Path(WEIGHTS)
    if not weights.exists():
        raise FileNotFoundError(
            f"Could not find trained weights at {weights.resolve()}.\n"
            "Train the model first with train.py."
        )
    source = Path(SOURCE)
    if not source.exists():
        raise FileNotFoundError(
            f"Could not find SOURCE at {source.resolve()}.\n"
            "Update SOURCE to an existing image/folder/video path."
        )

    model = YOLO(str(weights))
    results = model.predict(
        source=str(source),
        conf=CONF,
        imgsz=IMG_SIZE,
        save=True,
        save_txt=True,    # save polygon coordinates
        save_conf=True,   # include confidence values in the .txt files
        project=str(SCRIPT_DIR / "runs" / "segment"),
        name="predict",
    )

    # Print a per-image instance count
    for r in results:
        n = 0 if r.masks is None else len(r.masks)
        print(f"{Path(r.path).name}: {n} instance(s)")
    print("\nAnnotated images saved under runs/segment/predict/")


if __name__ == "__main__":
    main()
