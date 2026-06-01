"""
Local YOLO11 instance segmentation training.

Usage:
    1. Export your dataset from Roboflow as "YOLOv11" format (instance
       segmentation) and unzip it into this folder as ./dataset/ containing
       data.yaml, train/, valid/, test/ (each with images/ + labels/).
    2. Run:  .venv\\Scripts\\python.exe train.py
"""
from pathlib import Path
import sys

from ultralytics import YOLO

from check_dataset import main as validate_dataset

# Anchor paths to this script's folder so it runs from any working directory.
SCRIPT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Config -- edit these to match your setup
# ---------------------------------------------------------------------------
DATA_YAML = SCRIPT_DIR / "dataset" / "data.yaml"  # data.yaml from your Roboflow export
MODEL = str(SCRIPT_DIR / "yolo11n-seg.pt")  # nano segmentation model
EPOCHS = 100
IMG_SIZE = 640
BATCH = 8                         # keep small for 6 GB VRAM; lower to 4 if OOM
PATIENCE = 25                     # early-stopping patience
DEVICE = 0                        # 0 = first GPU, or "cpu"
RUN_NAME = "train"                # output folder name under runs/segment/
# ---------------------------------------------------------------------------


def main():
    data_path = Path(DATA_YAML)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find {data_path.resolve()}.\n"
            "Export your dataset from Roboflow as 'YOLOv11' (instance "
            "segmentation), unzip it here, and update DATA_YAML if needed."
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
        project=str(SCRIPT_DIR / "runs" / "segment"),
        name=RUN_NAME,
    )

    # Validate the best checkpoint on the validation split
    metrics = model.val(
        project=str(SCRIPT_DIR / "runs" / "segment"),
        name="val",
    )
    print("\nValidation summary:")
    print(f"  box  mAP50-95: {metrics.box.map:.4f}")
    print(f"  mask mAP50-95: {metrics.seg.map:.4f}")
    print(f"  mask mAP50:    {metrics.seg.map50:.4f}")
    print(f"Results saved under runs/segment/{RUN_NAME}/")


if __name__ == "__main__":
    main()
