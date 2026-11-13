# Models

All models are ExecuTorch `.pte` files (XNNPACK delegate) that run on a Raspberry Pi CPU.
The files are too large for the repository, so they are attached to the GitHub Release **`v1-models`**.
Download them and place them next to the code (or let `setup/setup.sh` copy them).

| File | What it does | Input | How it was made | Status |
|---|---|---|---|---|
| `head320.pte` | Watch tier: finds heads on every frame | 320 × 320 | YOLOv8n fine-tuned on CrowdHuman head boxes ([notebook 01](../notebooks/01_head_training_kaggle.ipynb)), exported with [notebook 02](../notebooks/02_head_export_colab.ipynb) | ready, 29.4 ms on Pi 5 |
| `head640.pte` | Accurate tier: near-field head detection, run tiled | 640 × 640 | Same as above, 640 px | ready, 141 ms on Pi 5 |
| `head320_int8.pte`, `head640_int8.pte` | **Default.** INT8 versions for Arm speed | as above | Post-training quantisation, XNNPACK quantiser, Detect layer kept FP32 ([notebook 04](../notebooks/04_int8_quantise_colab.ipynb)) | ready, 21 ms / 101 ms on Pi 5, counts within 3% of FP32 |
| `dmcount_qnrf_432x240.pte` | **Used by the hybrid** for far and packed regions, at 1 × 1 or 2 × 2 tiles | 432 × 240 | Pretrained DM-Count (UCF-QNRF), exported with [notebook 03](../notebooks/03_dmcount_export_colab.ipynb) | ready, 720 ms per tile |
| `dmcount_qnrf_256x144.pte` | Alternative density model, 4 × 4 tiles | 256 × 144 | as above | ready, 255 ms per tile |
| `yolov8n_320.pte`, `yolov8n.pte` | Pretrained person (body) detector, kept as fallback | 320 / 640 | Pretrained YOLOv8n (COCO), Ultralytics ExecuTorch export | ready |

## Output formats
- **YOLO models:** `(1, 4 + C, N)`: box centre x, y, width, height, then C class scores (C = 1 for head models,
  80 for COCO). N = 2,100 anchors at 320 px, 8,400 at 640 px.
- **DM-Count:** `(1, 1, H/8, W/8)` density map. The sum of the map is the number of people.

## Credits and licences
- YOLOv8 / Ultralytics: AGPL-3.0.
- DM-Count (Wang et al., NeurIPS 2020): MIT, pretrained weights from the authors.
- CrowdHuman (training data for the head models): CC BY-NC 4.0, non-commercial research. Images are not included here.
