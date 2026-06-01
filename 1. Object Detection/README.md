# YOLO11 Object Detection - Documentation

This project trains a YOLO11 object detection model for vehicle-part detection.
The results documented here are taken from the current local run artifacts in
`runs/detect/train/`, especially `results.csv` and `args.yaml`.

## What This Model Does

The model predicts bounding boxes for 9 vehicle-part classes:

| Class | Annotation count |
| --- | ---: |
| Fog Lights | 605 |
| Headlights | 1,184 |
| Plate | 456 |
| Rear Window | 123 |
| Side-Mirror | 1,207 |
| Tail Lights | 365 |
| Tire | 2,043 |
| Window | 1,816 |
| Windshield | 957 |

Total labeled objects across all splits: `8,756`.

## Project Layout

```text
1. Object Detection/
|-- check_dataset.py
|-- dataset/
|   |-- data.yaml
|   |-- train/images/ and train/labels/
|   |-- valid/images/ and valid/labels/
|   `-- test/images/ and test/labels/
|-- predict.py
|-- README.md
|-- runs/detect/
|   |-- predict/
|   |   `-- annotated prediction images
|   `-- train/
|       |-- args.yaml
|       |-- results.csv
|       |-- results.png
|       |-- confusion_matrix.png
|       |-- confusion_matrix_normalized.png
|       |-- BoxP_curve.png, BoxR_curve.png, BoxF1_curve.png, BoxPR_curve.png
|       |-- train_batch*.jpg and val_batch*_*.jpg
|       `-- weights/best.pt and weights/last.pt
|-- train.py
`-- yolo11n.pt
```

Note: `dataset/`, `runs/`, `.venv/`, and `*.pt` files are ignored by git in this
folder. The documentation references local artifacts that exist on this machine.
If this project is shared, export the run artifacts separately.

## Dataset Summary

Dataset source metadata in `dataset/data.yaml`:

| Field | Value |
| --- | --- |
| Format | YOLO detection, `data.yaml` plus image/label folders |
| Roboflow project | `objectdetection-7jqlt` |
| Roboflow version | `1` |
| Number of classes | `9` |

Split quality checks from the current local dataset:

| Split | Images | Label files | Objects | Empty labels | Missing labels | Orphan labels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 630 | 630 | 7,667 | 0 | 0 | 0 |
| Valid | 60 | 60 | 737 | 0 | 0 | 0 |
| Test | 30 | 30 | 352 | 0 | 0 | 0 |

The dataset is structurally clean: every image has a matching label file and no
orphan label files were found.

## Environment

This folder uses its own Python virtual environment:

```powershell
cd "1. Object Detection"
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

Expected GPU result: `True` when CUDA-enabled PyTorch is installed. CPU training
also works but is slower.

## Training Configuration

The current run was launched by `train.py` and saved to `runs/detect/train/`.
The full resolved configuration is saved in `runs/detect/train/args.yaml`.

| Setting | Value |
| --- | --- |
| Task | Detection |
| Base model | `yolo11n.pt` |
| Data file | `dataset/data.yaml` |
| Configured epochs | 100 |
| Completed epochs in log | 72 |
| Image size | 640 |
| Batch size | 8 |
| Early stopping patience | 25 |
| Device | `0` GPU |
| Optimizer | `auto` |
| AMP | `true` |
| Seed | `0` |
| Output folder | `runs/detect/train/` |

Run training:

```powershell
cd "1. Object Detection"
.\.venv\Scripts\python.exe train.py
```

`train.py` validates the dataset first by calling `check_dataset.py`. If blocking
errors are found, training stops before any GPU time is spent.

## Training Results

Primary metric: `mAP50-95(B)`. This is stricter than `mAP50` because it averages
box quality over multiple IoU thresholds from 0.50 to 0.95.

| Metric | Best epoch 47 | Final epoch 72 |
| --- | ---: | ---: |
| Precision (B) | 0.8329 | 0.8408 |
| Recall (B) | 0.7587 | 0.7306 |
| mAP50 (B) | 0.8169 | 0.8002 |
| mAP50-95 (B) | 0.6442 | 0.6379 |

The best checkpoint is:

```text
runs/detect/train/weights/best.pt
```

Use `best.pt` for inference unless you specifically need the final checkpoint
`last.pt`. Both checkpoint files are about `5.22 MB`.

## Result Figures

Training curves:

![Training results](runs/detect/train/results.png)

Confusion matrix:

![Confusion matrix](runs/detect/train/confusion_matrix.png)

Normalized confusion matrix:

![Normalized confusion matrix](runs/detect/train/confusion_matrix_normalized.png)

Precision-recall curve:

![Box precision-recall curve](runs/detect/train/BoxPR_curve.png)

Additional curves available in the run folder:

| File | Meaning |
| --- | --- |
| `BoxP_curve.png` | Precision at confidence thresholds |
| `BoxR_curve.png` | Recall at confidence thresholds |
| `BoxF1_curve.png` | F1 score at confidence thresholds |
| `BoxPR_curve.png` | Precision-recall tradeoff |
| `labels.jpg` | Label distribution and bounding-box distribution |
| `train_batch*.jpg` | Augmented training batch samples |
| `val_batch*_labels.jpg` | Validation images with ground-truth labels |
| `val_batch*_pred.jpg` | Validation images with model predictions |

## Inference

The inference script uses:

| Setting | Value |
| --- | --- |
| Weights | `runs/detect/train/weights/best.pt` |
| Source | `dataset/test/images` |
| Confidence threshold | 0.25 |
| Image size | 640 |
| Output folder | `runs/detect/predict/` |

Run prediction:

```powershell
cd "1. Object Detection"
.\.venv\Scripts\python.exe predict.py
```

The current local prediction folder contains:

| Output | Count | Location |
| --- | ---: | --- |
| Annotated images | 30 | `runs/detect/predict/` |
| Prediction text files | 0 | Not generated by the current `predict.py` settings |

The detection script saves annotated images only. If you need YOLO-format
prediction text files, add `save_txt=True` and `save_conf=True` to the
`model.predict(...)` call in `predict.py`.

## How To Read The Metrics

| Metric | Meaning |
| --- | --- |
| Precision | Of the boxes predicted by the model, how many were correct |
| Recall | Of the real labeled objects, how many the model found |
| mAP50 | Mean average precision at IoU 0.50 |
| mAP50-95 | Mean average precision averaged across IoU 0.50 to 0.95 |
| Box loss | Bounding-box localization loss; lower is better |
| Class loss | Classification loss for detected boxes; lower is better |
| DFL loss | Distribution focal loss used for box quality; lower is better |

For this project, `mAP50-95(B)` is the main score to report. `mAP50` is useful
for a looser view of object localization performance.

## Dataset Validation

Run the checker directly:

```powershell
cd "1. Object Detection"
.\.venv\Scripts\python.exe check_dataset.py
```

The checker verifies:

- `data.yaml` exists and defines class names.
- `train`, `valid`, and `test` splits have matching `images/` and `labels/` folders.
- Every image has a matching label file.
- Label files have valid YOLO detection rows: `class x_center y_center width height`.
- Class ids are valid and coordinates are normalized from `0` to `1`.
- Every class appears in the annotations.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| CUDA out of memory | Lower `BATCH` from 8 to 4 or 2 in `train.py`. |
| Training is too slow | Use GPU device `0`; reduce `IMG_SIZE` to 512 if acceptable. |
| Bad or missing predictions | Confirm `runs/detect/train/weights/best.pt` exists before running `predict.py`. |
| Dataset validation fails | Fix the exact file or label path reported by `check_dataset.py`. |
| Metrics look high but examples look wrong | Inspect `val_batch*_pred.jpg` and both confusion matrix files. |

## Reproducible Workflow

```powershell
cd "1. Object Detection"
.\.venv\Scripts\python.exe check_dataset.py
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe predict.py
```

Report the model using the best-validation result: `mAP50-95(B) = 0.6442` at
epoch `47`.
