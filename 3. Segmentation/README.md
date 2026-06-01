# Local YOLO11 Instance Segmentation

Train a YOLO11 instance segmentation model locally on your GPU using a dataset
exported from Roboflow. No API key or internet training needed.

## Environment

This folder needs its own virtual environment with Ultralytics installed.
If you haven't created one yet:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install ultralytics
```

To verify the GPU is detected:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

Should print `True` (if you have a CUDA build of PyTorch). If it prints
`False`, training still works on CPU, just slower.

## Step 1 - Export your dataset from Roboflow

Segmentation uses the same layout as detection (a `data.yaml` plus per-image
label files), but each label is a **polygon** rather than a box.

1. In Roboflow, open your instance-segmentation project and click **Versions**.
2. Click **Download Dataset**.
3. Choose format **YOLOv11** (instance segmentation).
4. Select **"Download zip to computer"**.
5. Unzip it into a folder named `dataset` here so you have:

```
dataset/
├── data.yaml
├── train/   (images/ + labels/)
├── valid/   (images/ + labels/)
└── test/    (images/ + labels/)
```

Each label line looks like: `class x1 y1 x2 y2 x3 y3 ...` (one class id
followed by polygon point pairs, all normalized 0..1).

## Step 2 - Train

```powershell
.\.venv\Scripts\python.exe train.py
```

Training runs a dataset validation check first. If the dataset has blocking
errors, training stops before wasting any time.

Adjust the settings at the top of `train.py` if needed:
- `MODEL` - `yolo11n-seg.pt` (nano) is best for a 6 GB laptop GPU. Use
  `yolo11s-seg.pt` for a bit more accuracy if memory allows.
- `BATCH` - start at 8; lower to 4 if you hit an out-of-memory error.
- `EPOCHS` - 100 is a reasonable starting point.

Results (weights, plots, metrics) are saved to `runs/segment/train/`.
The best checkpoint is `runs/segment/train/weights/best.pt`.

## Step 2a - Validate the dataset (optional, runs automatically)

```powershell
.\.venv\Scripts\python.exe check_dataset.py
```

It verifies `data.yaml`, the splits, matching `images/` and `labels/` folders,
and that each label is a well-formed polygon (class id + >=3 x/y pairs,
valid class ids, normalized coords).

## Step 3 - Run inference

```powershell
.\.venv\Scripts\python.exe predict.py
```

Annotated images (with masks) are saved to `runs/segment/predict/`, along with
polygon coordinate `.txt` files.

## Tips

- If you see a CUDA out-of-memory error, lower `BATCH` (8 -> 4 -> 2) or
  reduce `IMG_SIZE` (640 -> 512 -> 416) in `train.py`.
- Review `runs/segment/train/results.png` and `confusion_matrix.png` to judge
  training quality.
- For segmentation there are two sets of metrics: box (the bounding boxes) and
  mask (the polygon masks). `mask mAP50-95` is the main one to watch.
