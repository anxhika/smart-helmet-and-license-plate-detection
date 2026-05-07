# YOLO11 Inference Guide

## Overview
This guide explains how to run inference using the trained YOLO11n model on the `dataset_new` configuration.

## Trained Model Location
- **Best weights**: `yolo11/runs/yolo11n_dataset_new/weights/best.pt`
- **Copy to standard location**: `yolo11/weights/yolo11n_dataset_new.pt`

## Class Mapping
The model is trained on 4 classes in strict order:
0. **Helmet** - Rider wearing a helmet
1. **No Helmet** - Rider without a helmet
2. **Rider** - Motorcycle/bike rider detected
3. **LP** - License plate

## Configuration
Config is defined in `yolo11/config.py`:
- Default confidence threshold: 0.25
- Default IoU threshold: 0.45
- Image size: 640x640
- Device: CPU (change to cuda:0 if GPU available)

## Quick Start: Run Inference on a Test Image

### Command 1: Basic inference with visualization saved
```bash
python -m yolo11.detector_image --image "dataset_new/images/test/train__1.jpg"
```

**Output:**
- Annotated image saved to: `yolo11/outputs/images`
- JSON detection summary printed to console

### Command 2: Inference with custom model path
```bash
python -m yolo11.detector_image \
  --image "dataset_new/images/test/train__1.jpg" \
  --model "yolo11/runs/yolo11n_dataset_new/weights/best.pt"
```

### Command 3: Inference with custom confidence threshold
```bash
python -m yolo11.detector_image \
  --image "dataset_new/images/test/train__1.jpg" \
  --conf 0.5 \
  --iou 0.4
```

### Command 4: Inference without saving visualization
```bash
python -m yolo11.detector_image \
  --image "dataset_new/images/test/train__1.jpg" \
  --no-save
```

### Command 5: Save detection results to JSON
```bash
python -m yolo11.detector_image \
  --image "dataset_new/images/test/train__1.jpg" \
  --json "yolo11/outputs/images/result.json"
```

## Python API Usage

### Basic Detection
```python
from yolo11.detector_image import detect_image
import json

result = detect_image(image_path="dataset_new/images/test/train__1.jpg")
print(json.dumps(result, indent=2))
```

### With Custom Parameters
```python
from yolo11.detector_image import detect_image

result = detect_image(
    image_path="dataset_new/images/test/train__1.jpg",
    model_path="yolo11/weights/yolo11n_dataset_new.pt",
    conf=0.5,
    iou=0.4,
    device="cpu",
    save=True
)
```

### Save Detections to JSON Programmatically
```python
from yolo11.detector_image import detect_image, save_detection_json

result = detect_image(image_path="dataset_new/images/test/train__1.jpg")
json_path = save_detection_json(result, output_path="output.json")
print(f"Saved to: {json_path}")
```

## Output Format

The inference returns a JSON-like dictionary with this structure:

```json
{
  "success": true,
  "model": "yolo11/weights/yolo11n_dataset_new.pt",
  "image_path": "C:/Users/tuf gaming/Downloads/Backend-main/Backend-main/dataset_new/images/test/train__1.jpg",
  "image_name": "train__1.jpg",
  "image_size": [640, 480],
  "inference_ms": 42.5,
  "num_detections": 3,
  "class_summary": {
    "Helmet": 1,
    "No Helmet": 0,
    "Rider": 1,
    "LP": 1
  },
  "detections": [
    {
      "class_id": 0,
      "class_name": "Helmet",
      "confidence": 0.9542,
      "bbox_xyxy": [100.25, 50.5, 200.75, 180.0]
    },
    {
      "class_id": 2,
      "class_name": "Rider",
      "confidence": 0.8721,
      "bbox_xyxy": [80.0, 40.0, 220.0, 200.0]
    },
    {
      "class_id": 3,
      "class_name": "LP",
      "confidence": 0.7654,
      "bbox_xyxy": [150.0, 170.0, 200.0, 190.0]
    }
  ],
  "saved_dir": "C:/Users/tuf gaming/Downloads/Backend-main/Backend-main/yolo11/outputs/images"
}
```

## Output Files

When `save=True`, annotated images are saved to:
- **Directory**: `yolo11/outputs/images/`
- **Filename format**: `<date>_<counter>/` subdirectories with predictions overlaid

When using `--json` flag, detection JSON is saved to the specified path.

## Batch Inference Example

```python
from yolo11.detector_image import detect_image
from pathlib import Path
import json

test_dir = Path("dataset_new/images/test")
results = []

for image_path in test_dir.glob("*.jpg"):
    result = detect_image(str(image_path), save=False)
    results.append(result)
    print(f"Processed {image_path.name}: {result['num_detections']} detections")

# Save all results
with open("yolo11/outputs/batch_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

## Troubleshooting

### Model weights not found
Ensure trained weights exist at:
- `yolo11/weights/yolo11n_dataset_new.pt` OR
- `yolo11/runs/yolo11n_dataset_new/weights/best.pt`

### Class mismatch error
Verify `dataset_new/data.yaml` has classes in this exact order:
```yaml
names:
  0: Helmet
  1: No Helmet
  2: Rider
  3: LP
```

### Slow inference (CPU)
For faster inference, use GPU:
```bash
python -m yolo11.detector_image --image "..." --device "cuda:0"
```

### Memory issues with large batches
Reduce batch size or process images one at a time.

## Notes
- YOLOv5 baseline (`models/`, `utils/`, `main.py`, etc.) remains **completely untouched**
- YOLO11 setup is fully isolated in the `yolo11/` folder
- Trained weights (5.2 MB) are lightweight and CPU-compatible
- Dataset structure (`dataset_new/`) mirrors YOLO standard format
