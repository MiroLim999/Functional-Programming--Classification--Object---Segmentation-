"""
Local YOLO11 image classification training.

Usage:
    1. Export your dataset from Roboflow as "Folder Structure" (classification)
       and unzip it into this folder as ./dataset/ containing train/, valid/,
       test/ -- each with one subfolder per class.
    2. Run:  .venv\\Scripts\\python.exe train.py

For classification there is no data.yaml; YOLO reads the class names from the
subfolder names and `data` points at the dataset root folder.
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
DATA_DIR = SCRIPT_DIR / "dataset"        # classification dataset root folder
MODEL = str(SCRIPT_DIR / "yolo11n-cls.pt")  # nano classification model
EPOCHS = 100
IMG_SIZE = 224                    # standard size for classification (not 640)
BATCH = 32                        # classification is light; 32 is fine on 6 GB
PATIENCE = 25                     # early-stopping patience
DEVICE = 0                        # 0 = first GPU, or "cpu"
RUN_NAME = "train"                # output folder name under runs/classify/
# ---------------------------------------------------------------------------


def main():
    data_path = Path(DATA_DIR)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find {data_path.resolve()}.\n"
            "Export your dataset from Roboflow as 'Folder Structure' "
            "(classification), unzip it here, and update DATA_DIR if needed."
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
        project=str(SCRIPT_DIR / "runs" / "classify"),
        name=RUN_NAME,
    )

    # Validate the best checkpoint on the validation split
    metrics = model.val(
        project=str(SCRIPT_DIR / "runs" / "classify"),
        name="val",
    )
    print("\nValidation summary:")
    print(f"  top-1 accuracy: {metrics.top1:.4f}")
    print(f"  top-5 accuracy: {metrics.top5:.4f}")
    print(f"Results saved under runs/classify/{RUN_NAME}/")


if __name__ == "__main__":
    main()
