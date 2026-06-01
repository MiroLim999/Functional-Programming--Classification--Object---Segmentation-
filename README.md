# Local YOLO Object Detection Training

Train a YOLO object detection model locally on your GPU (RTX 4050) using an
annotated dataset exported from Roboflow. No API key or internet training needed.

## Environment (already set up)

- Python virtual environment in `.venv/`
- PyTorch 2.6.0 with CUDA 12.4 (GPU-enabled)
- Ultralytics (YOLO)

To verify the GPU is detected:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

Should print `True`.

## Step 1 - Export your dataset from Roboflow

1. In Roboflow, open your project and click **Versions** (create/Generate one if needed).
2. Click **Download Dataset**.
3. Choose format **YOLOv8**.
4. Select **"Download zip to computer"** (not the code/API option).
5. Unzip it into a folder named `dataset` in this project so you have:

```
dataset/
├── data.yaml
├── train/   (images/ + labels/)
├── valid/   (images/ + labels/)
└── test/    (images/ + labels/)
```

## Step 2 - Train

```powershell
.\.venv\Scripts\python.exe train.py
```

Training automatically runs a dataset validation check first (see Step 2a). If
the dataset has blocking errors, training stops before wasting any time.

Adjust the settings at the top of `train.py` if needed:
- `MODEL` - `yolo11n.pt` (nano) is best for a 6 GB laptop GPU. Use `yolo11s.pt` for a bit more accuracy if memory allows.
- `BATCH` - start at 8; lower to 4 if you hit an out-of-memory error.
- `EPOCHS` - 100 is a reasonable starting point.

Results (weights, plots, metrics) are saved to `runs/detect/train/`.
The best checkpoint is `runs/detect/train/weights/best.pt`.

## Step 2a - Validate the dataset (optional, runs automatically)

`train.py` calls this for you, but you can run it on its own any time to check
your Roboflow export is correct:

```powershell
.\.venv\Scripts\python.exe check_dataset.py
```

It verifies:
- `data.yaml` exists and lists class names
- each split (train/valid/test) has matching `images/` and `labels/` folders
- every image has a label file (flags missing labels, empty labels, orphans)
- label files are well-formed (5 columns, valid class ids, normalized coords)
- each class actually appears in the annotations

It exits with code 0 when the dataset is usable, 1 when there are blocking
errors. Warnings (like a few unlabelled images) don't block training.

## Step 3 - Run inference

```powershell
.\.venv\Scripts\python.exe predict.py
```

Annotated predictions are saved to `runs/detect/predict/`.

## Tips

- If you see a CUDA out-of-memory error, lower `BATCH` (8 -> 4 -> 2) or
  reduce `IMG_SIZE` (640 -> 512 -> 416) in `train.py`.
- Review `runs/detect/train/results.png` and `confusion_matrix.png` to judge
  training quality.
- `mAP50-95` is the main accuracy metric (higher is better).
