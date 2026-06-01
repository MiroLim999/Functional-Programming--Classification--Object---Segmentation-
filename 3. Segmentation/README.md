# YOLO11 Instance Segmentation - Documentation

This project trains a YOLO11 instance segmentation model for vehicle-part mask
prediction. The results documented here are taken from the current local run
artifacts in `runs/segment/train/`, especially `results.csv` and `args.yaml`.

## What This Model Does

The model predicts instance masks for 9 vehicle-part classes:

| Class | Annotation count |
| --- | ---: |
| fog lights | 514 |
| headlight | 1,203 |
| plate | 372 |
| rear window | 69 |
| side mirror | 1,119 |
| tail lights | 270 |
| tires | 1,952 |
| window | 2,022 |
| windshield | 962 |

Total labeled instances across all splits: `8,483`.

## Project Layout

```text
3. Segmentation/
|-- check_dataset.py
|-- dataset/
|   |-- data.yaml
|   |-- data.train_resolved.yaml
|   |-- train/images/ and train/labels/
|   |-- valid/images/ and valid/labels/
|   `-- test/images/ and test/labels/
|-- predict.py
|-- README.md
|-- runs/segment/
|   |-- predict/
|   |   |-- labels/
|   |   `-- annotated prediction images
|   |-- train/
|   |   |-- args.yaml
|   |   |-- results.csv
|   |   |-- results.png
|   |   |-- confusion_matrix.png
|   |   |-- confusion_matrix_normalized.png
|   |   |-- Box*.png and Mask*.png curves
|   |   |-- train_batch*.jpg and val_batch*_*.jpg
|   |   `-- weights/best.pt and weights/last.pt
|   `-- val/
|-- train.py
`-- yolo11n-seg.pt
```

Note: `dataset/`, `runs/`, `.venv/`, and `*.pt` files are ignored by git in this
folder. The documentation references local artifacts that exist on this machine.
If this project is shared, export the run artifacts separately.

## Dataset Summary

Dataset source metadata in `dataset/data.yaml`:

| Field | Value |
| --- | --- |
| Format | YOLO instance segmentation, `data.yaml` plus polygon labels |
| Roboflow project | `segmentation-adhao` |
| Roboflow version | `1` |
| Number of classes | `9` |

Split quality checks from the current local dataset:

| Split | Images | Label files | Instances | Empty labels | Missing labels | Orphan labels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 627 | 627 | 7,417 | 0 | 0 | 0 |
| Valid | 60 | 60 | 715 | 0 | 0 | 0 |
| Test | 30 | 30 | 351 | 0 | 0 | 0 |

The dataset is structurally clean: every image has a matching label file and no
orphan label files were found.

A segmentation label row has this format:

```text
class_id x1 y1 x2 y2 x3 y3 ...
```

Each polygon must contain at least 3 `(x, y)` points, and all coordinates should
be normalized from `0` to `1`.

## Environment

This folder uses its own Python virtual environment:

```powershell
cd "3. Segmentation"
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

Expected GPU result: `True` when CUDA-enabled PyTorch is installed. CPU training
also works but is slower.

## Training Configuration

The current run was launched by `train.py` and saved to `runs/segment/train/`.
The full resolved configuration is saved in `runs/segment/train/args.yaml`.

| Setting | Value |
| --- | --- |
| Task | Segmentation |
| Base model | `yolo11n-seg.pt` |
| Data file used for training | `dataset/data.train_resolved.yaml` |
| Original data file | `dataset/data.yaml` |
| Configured epochs | 100 |
| Completed epochs in log | 90 |
| Image size | 640 |
| Batch size | 8 |
| Early stopping patience | 25 |
| Device | `0` GPU |
| Optimizer | `auto` |
| AMP | `true` |
| Seed | `0` |
| Output folder | `runs/segment/train/` |

`train.py` writes `data.train_resolved.yaml` before training so YOLO receives
absolute, existing image paths for `train`, `val`, and `test`.

Run training:

```powershell
cd "3. Segmentation"
.\.venv\Scripts\python.exe train.py
```

`train.py` validates the dataset first by calling `check_dataset.py`. If blocking
errors are found, training stops before any GPU time is spent.

## Training Results

Primary metric: `mAP50-95(M)`. This measures mask quality over multiple IoU
thresholds from 0.50 to 0.95. Box metrics are also logged because the model
learns boxes around each mask.

| Metric | Best mask epoch 65 | Final epoch 90 |
| --- | ---: | ---: |
| Box precision | 0.8101 | 0.8053 |
| Box recall | 0.7488 | 0.7835 |
| Box mAP50 | 0.8086 | 0.8043 |
| Box mAP50-95 | 0.6405 | 0.6346 |
| Mask precision | 0.7803 | 0.7948 |
| Mask recall | 0.7541 | 0.7802 |
| Mask mAP50 | 0.7861 | 0.7916 |
| Mask mAP50-95 | 0.5870 | 0.5776 |

The best checkpoint is:

```text
runs/segment/train/weights/best.pt
```

Use `best.pt` for inference unless you specifically need the final checkpoint
`last.pt`. Both checkpoint files are about `5.74 MB`.

## Result Figures

Training curves:

![Training results](runs/segment/train/results.png)

Confusion matrix:

![Confusion matrix](runs/segment/train/confusion_matrix.png)

Normalized confusion matrix:

![Normalized confusion matrix](runs/segment/train/confusion_matrix_normalized.png)

Mask precision-recall curve:

![Mask precision-recall curve](runs/segment/train/MaskPR_curve.png)

Box precision-recall curve:

![Box precision-recall curve](runs/segment/train/BoxPR_curve.png)

Additional curves available in the run folder:

| File | Meaning |
| --- | --- |
| `BoxP_curve.png` | Box precision at confidence thresholds |
| `BoxR_curve.png` | Box recall at confidence thresholds |
| `BoxF1_curve.png` | Box F1 score at confidence thresholds |
| `BoxPR_curve.png` | Box precision-recall tradeoff |
| `MaskP_curve.png` | Mask precision at confidence thresholds |
| `MaskR_curve.png` | Mask recall at confidence thresholds |
| `MaskF1_curve.png` | Mask F1 score at confidence thresholds |
| `MaskPR_curve.png` | Mask precision-recall tradeoff |
| `labels.jpg` | Label distribution and mask/box distribution |
| `train_batch*.jpg` | Augmented training batch samples |
| `val_batch*_labels.jpg` | Validation images with ground-truth masks |
| `val_batch*_pred.jpg` | Validation images with predicted masks |

## Validation Run Artifacts

`runs/segment/val/` exists and contains a validation artifact set generated from
model validation:

| Artifact type | Files |
| --- | --- |
| Box curves | `BoxP_curve.png`, `BoxR_curve.png`, `BoxF1_curve.png`, `BoxPR_curve.png` |
| Mask curves | `MaskP_curve.png`, `MaskR_curve.png`, `MaskF1_curve.png`, `MaskPR_curve.png` |
| Confusion matrices | `confusion_matrix.png`, `confusion_matrix_normalized.png` |
| Validation examples | `val_batch0_*`, `val_batch1_*`, `val_batch2_*` |

Use the `train/` folder for the logged training history. Use the `val/` folder to
inspect a separate validation visualization pass.

## Inference

The inference script uses:

| Setting | Value |
| --- | --- |
| Weights | `runs/segment/train/weights/best.pt` |
| Source | `dataset/test/images` |
| Confidence threshold | 0.25 |
| Image size | 640 |
| Output folder | `runs/segment/predict/` |
| Text output | `save_txt=True`, `save_conf=True` |

Run prediction:

```powershell
cd "3. Segmentation"
.\.venv\Scripts\python.exe predict.py
```

The current local prediction folder contains:

| Output | Count | Location |
| --- | ---: | --- |
| Annotated images | 30 | `runs/segment/predict/` |
| Prediction text files | 30 | `runs/segment/predict/labels/` |

Prediction text files contain the class id, normalized polygon coordinates, and
confidence value for each predicted instance.

## How To Read The Metrics

| Metric | Meaning |
| --- | --- |
| Box precision | Of predicted boxes, how many were correct |
| Box recall | Of real objects, how many boxes were found |
| Box mAP50-95 | Box quality averaged across IoU 0.50 to 0.95 |
| Mask precision | Of predicted masks, how many were correct |
| Mask recall | Of real masks, how many masks were found |
| Mask mAP50-95 | Mask quality averaged across IoU 0.50 to 0.95 |
| Seg loss | Mask segmentation loss; lower is better |
| Box/Class/DFL losses | Detection-related losses; lower is better |

For this project, report `mAP50-95(M) = 0.5870` from epoch `65` as the best mask
validation result. The final epoch had slightly higher mask recall but lower mask
mAP50-95, so `best.pt` remains the safer inference checkpoint.

## Dataset Validation

Run the checker directly:

```powershell
cd "3. Segmentation"
.\.venv\Scripts\python.exe check_dataset.py
```

The checker verifies:

- `data.yaml` exists and defines class names.
- `train`, `valid`, and `test` splits have matching `images/` and `labels/` folders.
- Every image has a matching label file.
- Label files have valid YOLO polygon rows.
- Each polygon has at least 3 point pairs.
- Class ids are valid and coordinates are normalized from `0` to `1`.
- Every class appears in the annotations.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| CUDA out of memory | Lower `BATCH` from 8 to 4 or 2 in `train.py`. |
| Training is too slow | Use GPU device `0`; reduce `IMG_SIZE` to 512 if acceptable. |
| Dataset validation fails | Fix malformed polygon rows, missing labels, or invalid class ids reported by `check_dataset.py`. |
| Predictions have boxes but poor masks | Prioritize mask curves and `MaskPR_curve.png`, not only box metrics. |
| Prediction text files are missing | Confirm `save_txt=True` and `save_conf=True` in `predict.py`. |

## Reproducible Workflow

```powershell
cd "3. Segmentation"
.\.venv\Scripts\python.exe check_dataset.py
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe predict.py
```

Report the model using the best-validation result: `mAP50-95(M) = 0.5870` at
epoch `65`.
