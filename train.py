"""
Local YOLO11 object detection training.

Usage:
    1. Export your dataset from Roboflow as "YOLOv11" format and unzip it into
       this folder (e.g. ./dataset/ containing data.yaml, train/, valid/, test/).
    2. Set DATA_YAML below to point at that data.yaml.
    3. Run:  .venv\Scripts\python.exe train.py
"""
from pathlib import Path
import sys

from ultralytics import YOLO

from check_dataset import main as validate_dataset

# ---------------------------------------------------------------------------
# Config -- edit these to match your setup
# ---------------------------------------------------------------------------
DATA_YAML = "dataset/data.yaml"   # path to the data.yaml from your Roboflow export
MODEL = "yolo11n.pt"              # nano model: best fit for a 6 GB laptop GPU
EPOCHS = 100
IMG_SIZE = 640
BATCH = 8                         # keep small for 6 GB VRAM; lower to 4 if OOM
PATIENCE = 25                     # early-stopping patience
DEVICE = 0                        # 0 = first GPU, or "cpu"
# ---------------------------------------------------------------------------


def main():
    data_path = Path(DATA_YAML)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find {data_path.resolve()}.\n"
            "Export your dataset from Roboflow as 'YOLOv11' format, unzip it here, "
            "and update DATA_YAML to point at its data.yaml."
        )

    # Validate the dataset before spending time training.
    print("Validating dataset...\n")
    sys.argv = ["check_dataset.py", "--data", str(data_path)]
    if validate_dataset() != 0:
        raise SystemExit(
            "\nDataset validation failed. Fix the errors above, then re-run."
        )
    print()

    model = YOLO(MODEL)
    model.train(
        data=str(data_path),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        patience=PATIENCE,
        device=DEVICE,
        plots=True,
    )

    # Validate the best checkpoint on the validation split
    metrics = model.val()
    print("\nValidation summary:")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print("Results saved under runs/detect/train/")


if __name__ == "__main__":
    main()
