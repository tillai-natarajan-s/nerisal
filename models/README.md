# Models

All models are ExecuTorch `.pte` files (XNNPACK delegate) that run on a Raspberry Pi CPU.
The files are too large for the repository, so they are attached to the GitHub Release **`v1-models`**.
Download them and place them next to the code (or let `setup/setup.sh` copy them).

| File | What it does | Input | How it was made | Status |
|---|---|---|---|---|
| `head320.pte` | Fast watch tier: finds heads on every frame | 320 × 320 | YOLOv8n fine-tuned on CrowdHuman head boxes ([notebook 01](../notebooks/01_head_training_kaggle.ipynb)), exported with [notebook 02](../notebooks/02_head_export_colab.ipynb) | training |
| `head640.pte` | Accurate tier: head detection, run tiled | 640 × 640 | Same as above, 640 px | training |
| `head320_int8.pte`, `head640_int8.pte` | INT8 versions for Arm speed | as above | Post-training quantisation, XNNPACK quantiser | planned |
| `dmcount_qnrf_256x144.pte` | Dense-crowd tier: density map, run as 4 × 4 tiles | 256 × 144 | Pretrained DM-Count (UCF-QNRF), exported with [notebook 03](../notebooks/03_dmcount_export_colab.ipynb) | ready |
| `dmcount_qnrf_432x240.pte` | Faster dense-crowd option, 2 × 2 tiles | 432 × 240 | as above | ready |
| `yolov8n_320.pte`, `yolov8n.pte` | Round-one person (body) detector, kept as fallback | 320 / 640 | Pretrained YOLOv8n (COCO), Ultralytics ExecuTorch export | ready |

## Output formats
- **YOLO models:** `(1, 4 + C, N)`: box centre x, y, width, height, then C class scores (C = 1 for head models,
  80 for COCO). N = 2,100 anchors at 320 px, 8,400 at 640 px.
- **DM-Count:** `(1, 1, H/8, W/8)` density map. The sum of the map is the number of people.

## Credits and licences
- YOLOv8 / Ultralytics: AGPL-3.0.
- DM-Count (Wang et al., NeurIPS 2020): MIT, pretrained weights from the authors.
- CrowdHuman (training data for the head models): CC BY-NC 4.0, non-commercial research. Images are not included here.
