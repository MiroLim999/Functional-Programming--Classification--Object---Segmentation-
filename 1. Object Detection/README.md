# YOLO11 Object Detection Documentation

## 1. Title Page

**Project Title:** YOLO11 Object Detection for Vehicle-Part Detection  
**Project Type:** Machine Learning / Computer Vision  
**Model Used:** YOLO11 Nano (`yolo11n.pt`)  
**Task:** Object detection  
**Main Output:** Annotated images with bounding boxes around detected vehicle parts  

This project trains and evaluates a YOLO11 object detection model that can identify
important vehicle parts from images. The final trained model is saved as
`runs/detect/train/weights/best.pt` and is used by `predict.py` to generate
annotated prediction images.

## 2. Project Overview

The goal of this project is to detect specific vehicle parts in car images using
a YOLO-based object detection model. Object detection is different from simple
classification because the model does not only predict what object is present,
but also where it is located in the image using bounding boxes.

The model detects 9 vehicle-part classes:

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

Total labeled objects across all dataset splits: `8,756`.

## 3. Dataset Description

The dataset is stored in YOLO detection format. It contains image folders, label
folders, and a `data.yaml` file that defines the dataset paths and class names.
The dataset metadata indicates that it came from a Roboflow project.

| Field | Value |
| --- | --- |
| Dataset format | YOLO detection |
| Dataset config file | `dataset/data.yaml` |
| Roboflow project | `objectdetection-7jqlt` |
| Roboflow version | `1` |
| Number of classes | `9` |

Dataset split summary:

| Split | Images | Label files | Objects | Empty labels | Missing labels | Orphan labels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 630 | 630 | 7,667 | 0 | 0 | 0 |
| Valid | 60 | 60 | 737 | 0 | 0 | 0 |
| Test | 30 | 30 | 352 | 0 | 0 | 0 |

The dataset is structurally clean because every image has a matching label file,
there are no missing labels, and there are no orphan label files.

## 4. Methodology

The project workflow has three main steps: dataset validation, model training,
and model prediction.

### Dataset Validation

Before training, `check_dataset.py` verifies that the dataset is usable. It checks
whether `data.yaml` exists, whether the train, validation, and test folders are
present, and whether each image has a matching label file. It also checks that
YOLO label rows follow the required format:

```text
class_id x_center y_center width height
```

The script also validates that class IDs are within range and bounding-box values
are normalized between `0` and `1`.

### Model Training

Training is handled by `train.py`. The script first runs the dataset checker. If
the dataset has blocking errors, training stops before using GPU time. If the
dataset passes validation, the script trains the YOLO11 model using the dataset
defined in `dataset/data.yaml`.

Run training with:

```powershell
cd "1. Object Detection"
.\.venv\Scripts\python.exe train.py
```

### Prediction

Prediction is handled by `predict.py`. It loads the best trained checkpoint and
runs inference on the test images.

Run prediction with:

```powershell
cd "1. Object Detection"
.\.venv\Scripts\python.exe predict.py
```

Annotated prediction images are saved in:

```text
runs/detect/predict/
```

## 5. Training Configuration

The current training run was saved in `runs/detect/train/`. The full resolved
training configuration is also available in `runs/detect/train/args.yaml`.

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

The best trained model checkpoint is:

```text
runs/detect/train/weights/best.pt
```

The final checkpoint from the last completed epoch is:

```text
runs/detect/train/weights/last.pt
```

For inference and reporting, `best.pt` is the recommended checkpoint because it
represents the best validation performance.

## 6. Results

The main evaluation metric for this object detection model is `mAP50-95(B)`.
This metric is stricter than `mAP50` because it averages detection performance
over multiple IoU thresholds from `0.50` to `0.95`.

| Metric | Best epoch 47 | Final epoch 72 |
| --- | ---: | ---: |
| Precision (B) | 0.8329 | 0.8408 |
| Recall (B) | 0.7587 | 0.7306 |
| mAP50 (B) | 0.8169 | 0.8002 |
| mAP50-95 (B) | 0.6442 | 0.6379 |

The best validation result was achieved at epoch `47`, with:

```text
mAP50-95(B) = 0.6442
```

The current prediction output contains 30 annotated test images in
`runs/detect/predict/`.

## 7. Result Images

The following result images should be included in the Word documentation:

| Result figure | File path | Purpose |
| --- | --- | --- |
| Training curves | `runs/detect/train/results.png` | Shows training and validation loss/metric trends |
| Confusion matrix | `runs/detect/train/confusion_matrix.png` | Shows class-level prediction errors |
| Normalized confusion matrix | `runs/detect/train/confusion_matrix_normalized.png` | Shows class-level errors as normalized values |
| Precision-recall curve | `runs/detect/train/BoxPR_curve.png` | Shows the precision and recall tradeoff |
| Label distribution | `runs/detect/train/labels.jpg` | Shows class and bounding-box distribution |
| Validation predictions | `runs/detect/train/val_batch*_pred.jpg` | Shows validation images with predicted boxes |
| Test predictions | `runs/detect/predict/*.jpg` | Shows final annotated prediction results |

Recommended images to insert in the Word file:

```text
runs/detect/train/results.png
runs/detect/train/confusion_matrix.png
runs/detect/train/confusion_matrix_normalized.png
runs/detect/train/BoxPR_curve.png
runs/detect/train/val_batch0_pred.jpg
runs/detect/predict/03519_Ford-F-150-Regular-Cab-2012_jpg.rf.7196f729cf4ab280a81463d73f2a1293.jpg
```

When writing captions, describe what each image shows. For example:

```text
Figure 1. Training results showing loss and detection metrics across epochs.
Figure 2. Confusion matrix showing correct and incorrect predictions per class.
Figure 3. Sample test image with predicted bounding boxes generated by the model.
```

## 8. Conclusion

This project successfully trained a YOLO11 object detection model for detecting
vehicle parts across 9 classes. The dataset was checked before training and was
found to be structurally clean, with no missing labels or orphan label files.

The best model achieved an `mAP50-95(B)` score of `0.6442` at epoch `47`, with
precision of `0.8329` and recall of `0.7587`. These results show that the model
can detect multiple vehicle parts with good overall performance, especially for
classes with stronger representation in the dataset such as Tire, Window,
Side-Mirror, Headlights, and Windshield.

The trained checkpoint `runs/detect/train/weights/best.pt` was used for
inference, and the prediction results were saved as annotated images in
`runs/detect/predict/`. For future improvement, the model could be trained with
more images for underrepresented classes such as Rear Window, Tail Lights, and
Plate to improve class balance and detection reliability.
