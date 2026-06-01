# Local YOLO11 Image Classification

Train a YOLO11 image classification model locally on your GPU using a
folder-structured dataset exported from Roboflow. No API key or internet
training needed.

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

Classification datasets are **folder-based** (no `data.yaml`, no label files).
The class of each image is determined by which folder it sits in.

1. In Roboflow, open your classification project and click **Versions**.
2. Click **Download Dataset**.
3. Choose format **Folder Structure**.
4. Select **"Download zip to computer"**.
5. Unzip it into a folder named `dataset` here so you have:

```
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
```

Every split should contain a subfolder for *each* class. If `valid` or `test`
is missing a class, validation metrics will be misleading.

## Step 2 - Train

```powershell
.\.venv\Scripts\python.exe train.py
```

Training runs a dataset validation check first. If the dataset has blocking
errors, training stops before wasting any time.

Adjust the settings at the top of `train.py` if needed:
- `MODEL` - `yolo11n-cls.pt` (nano) is best for a 6 GB laptop GPU. Use
  `yolo11s-cls.pt` for a bit more accuracy if memory allows.
- `IMG_SIZE` - `224` is the standard classification input size.
- `BATCH` - 32 is fine; lower if you hit an out-of-memory error.
- `EPOCHS` - 100 is a reasonable starting point.

Results (weights, plots, metrics) are saved to `runs/classify/train/`.
The best checkpoint is `runs/classify/train/weights/best.pt`.

## Step 2a - Validate the dataset (optional, runs automatically)

```powershell
.\.venv\Scripts\python.exe check_dataset.py
```

It verifies the dataset root and splits exist, each split has class subfolders,
every class folder has images, and the classes are consistent across splits.

## Step 3 - Run inference

```powershell
.\.venv\Scripts\python.exe predict.py
```

For each test image it prints the predicted class and confidence. Annotated
images are saved to `runs/classify/predict/`.

## Tips

- If you see a CUDA out-of-memory error, lower `BATCH` or reduce `IMG_SIZE`.
- Review `runs/classify/train/results.png` and `confusion_matrix.png` to judge
  training quality.
- top-1 accuracy is the main metric (how often the top guess is correct).
