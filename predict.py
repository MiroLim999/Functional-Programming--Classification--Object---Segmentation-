"""
Run inference with your trained YOLO detection model.

Usage:
    .\.venv\Scripts\python.exe predict.py
Annotated images are saved under runs/detect/predict/.
"""
from pathlib import Path
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WEIGHTS = "runs/detect/train/weights/best.pt"  # trained weights
SOURCE = "dataset/test/images"                 # folder, single image, or video
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

    model = YOLO(str(weights))
    results = model.predict(
        source=SOURCE,
        conf=CONF,
        imgsz=IMG_SIZE,
        save=True,
    )

    # Print a per-image detection count
    for r in results:
        n = len(r.boxes)
        print(f"{Path(r.path).name}: {n} detection(s)")
    print("\nAnnotated images saved under runs/detect/predict/")


if __name__ == "__main__":
    main()
