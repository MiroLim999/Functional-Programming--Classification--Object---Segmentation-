# YOLO11 Image Classification - Documentation

This project trains a YOLO11 image classification model for vehicle type
classification. The results documented here are taken from the current local run
artifacts in `runs/classify/train/`, especially `results.csv` and `args.yaml`.

## What This Model Does

The model predicts one class for each full image:

| Class | Train images | Valid images | Test images |
| --- | ---: | ---: | ---: |
| Pick-Up | 1,236 | 82 | 88 |
| SUV | 1,341 | 93 | 104 |
| VAN | 1,236 | 98 | 81 |
| Total | 3,813 | 273 | 273 |

## Project Layout

```text
2. Classification/
|-- check_dataset.py
|-- dataset/
|   |-- train/Pick-Up/, train/SUV/, train/VAN/
|   |-- valid/Pick-Up/, valid/SUV/, valid/VAN/
|   `-- test/Pick-Up/, test/SUV/, test/VAN/
|-- predict.py
|-- README.md
|-- runs/classify/
|   |-- predict/
|   |-- train/
|   |   |-- args.yaml
|   |   |-- results.csv
|   |   |-- results.png
|   |   |-- confusion_matrix.png
|   |   |-- confusion_matrix_normalized.png
|   |   |-- train_batch*.jpg and val_batch*_*.jpg
|   |   `-- weights/best.pt and weights/last.pt
|   `-- val/
|-- train.py
`-- yolo11n-cls.pt
```

Note: `dataset/`, `runs/`, `.venv/`, and `*.pt` files are ignored by git in this
folder. The documentation references local artifacts that exist on this machine.
If this project is shared, export the run artifacts separately.

## Dataset Summary

Classification uses folder names as labels. There is no `data.yaml` and no label
text file per image.

Expected layout:

```text
dataset/
|-- train/
|   |-- Pick-Up/
|   |-- SUV/
|   `-- VAN/
|-- valid/
|   |-- Pick-Up/
|   |-- SUV/
|   `-- VAN/
`-- test/
    |-- Pick-Up/
    |-- SUV/
    `-- VAN/
```

Split totals:

| Split | Total images | Classes present |
| --- | ---: | --- |
| Train | 3,813 | Pick-Up, SUV, VAN |
| Valid | 273 | Pick-Up, SUV, VAN |
| Test | 273 | Pick-Up, SUV, VAN |

The split structure is consistent: all three classes exist in train, valid, and
test.

## Environment

This folder uses its own Python virtual environment:

```powershell
cd "2. Classification"
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

Expected GPU result: `True` when CUDA-enabled PyTorch is installed. CPU training
also works but is slower.

## Training Configuration

The current run was launched by `train.py` and saved to `runs/classify/train/`.
The full resolved configuration is saved in `runs/classify/train/args.yaml`.

| Setting | Value |
| --- | --- |
| Task | Classification |
| Base model | `yolo11n-cls.pt` |
| Data folder | `dataset/` |
| Configured epochs | 100 |
| Completed epochs in log | 66 |
| Image size | 224 |
| Batch size | 32 |
| Early stopping patience | 25 |
| Device | `0` GPU |
| Optimizer | `auto` |
| AMP | `true` |
| Seed | `0` |
| Output folder | `runs/classify/train/` |

Run training:

```powershell
cd "2. Classification"
.\.venv\Scripts\python.exe train.py
```

`train.py` validates the dataset first by calling `check_dataset.py`. If blocking
errors are found, training stops before any GPU time is spent.

## Training Results

Primary metric: `accuracy_top1`. This is the percentage of validation images
where the model's top prediction is correct.

| Metric | Best epoch 41 | Final epoch 66 |
| --- | ---: | ---: |
| Top-1 accuracy | 0.9927 | 0.9853 |
| Top-5 accuracy | 1.0000 | 1.0000 |
| Train loss | 0.0505 | 0.0271 |
| Validation loss | 0.0400 | 0.0381 |

Top-5 accuracy is not very useful for this project because there are only three
classes. The main score to report is top-1 accuracy.

The best checkpoint is:

```text
runs/classify/train/weights/best.pt
```

Use `best.pt` for inference unless you specifically need the final checkpoint
`last.pt`. Both checkpoint files are about `3.04 MB`.

## Result Figures

Training curves:

![Training results](runs/classify/train/results.png)

Confusion matrix:

![Confusion matrix](runs/classify/train/confusion_matrix.png)

Normalized confusion matrix:

![Normalized confusion matrix](runs/classify/train/confusion_matrix_normalized.png)

Additional training artifacts:

| File | Meaning |
| --- | --- |
| `results.csv` | Per-epoch losses, top-1 accuracy, top-5 accuracy, and learning rates |
| `results.png` | Visual summary of training and validation curves |
| `confusion_matrix.png` | Raw validation confusion matrix |
| `confusion_matrix_normalized.png` | Normalized validation confusion matrix |
| `train_batch*.jpg` | Augmented training batch samples |
| `val_batch*_labels.jpg` | Validation images with ground-truth labels |
| `val_batch*_pred.jpg` | Validation images with predicted labels |
| `weights/best.pt` | Best checkpoint selected from validation performance |
| `weights/last.pt` | Final checkpoint from the last logged epoch |

## Validation Run Artifacts

`runs/classify/val/` exists and contains a validation artifact set generated from
model validation:

| Artifact type | Files |
| --- | --- |
| Confusion matrices | `confusion_matrix.png`, `confusion_matrix_normalized.png` |
| Validation examples | `val_batch0_*`, `val_batch1_*`, `val_batch2_*` |

Use the `train/` folder for the logged training history. Use the `val/` folder to
inspect a separate validation visualization pass.

## Inference

The inference script uses:

| Setting | Value |
| --- | --- |
| Weights | `runs/classify/train/weights/best.pt` |
| Source | `dataset/test/` recursively |
| Image size | 224 |
| Output folder | `runs/classify/predict/` |

Run prediction:

```powershell
cd "2. Classification"
.\.venv\Scripts\python.exe predict.py
```

The current local prediction folder contains `273` annotated image files, matching
the number of images in the test split.

For each image, `predict.py` prints:

```text
image_name.jpg -> predicted_class (confidence)
```

## How To Read The Metrics

| Metric | Meaning |
| --- | --- |
| Top-1 accuracy | The model's highest-confidence class equals the true class |
| Top-5 accuracy | The true class is within the model's top 5 classes; not meaningful here because there are only 3 classes |
| Train loss | Optimization loss on training batches; lower is better |
| Validation loss | Loss on validation data; lower is better |

For this project, report `accuracy_top1 = 0.9927` from epoch `41` as the best
validation result.

## Dataset Validation

Run the checker directly:

```powershell
cd "2. Classification"
.\.venv\Scripts\python.exe check_dataset.py
```

The checker verifies:

- `dataset/` exists.
- `train`, `valid`, and `test` splits exist.
- Each split contains class subfolders.
- Every class folder contains image files.
- Class names are consistent across all splits.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| CUDA out of memory | Lower `BATCH` from 32 to 16 or 8 in `train.py`. |
| Dataset validation fails | Make sure each split contains `Pick-Up`, `SUV`, and `VAN` folders. |
| Model predicts one class too often | Inspect the confusion matrix and check class balance or image leakage. |
| Prediction finds no images | Confirm `SOURCE = dataset/test` in `predict.py` and test images are inside class subfolders. |
| Top-5 looks perfect | Ignore top-5 for reporting; use top-1 because this is a 3-class problem. |

## Reproducible Workflow

```powershell
cd "2. Classification"
.\.venv\Scripts\python.exe check_dataset.py
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe predict.py
```

Report the model using the best-validation result: `accuracy_top1 = 0.9927` at
epoch `41`.
