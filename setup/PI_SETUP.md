# Raspberry Pi setup

NERISAL runs on a **Raspberry Pi 4 or 5** with **64-bit Raspberry Pi OS**. Everything runs on the Pi; a laptop on
the same Wi-Fi or hotspot only views the dashboard.

## 1. One-command setup
From the repository folder (or the offline pendrive folder):
```bash
bash setup/setup.sh
```
It will:
1. report the board, OS bit-width (must be `aarch64`), Python version and RAM
2. copy the code and `.pte` models to `~/nerisal`
3. create the Python environment `~/et` and install the packages, **offline** from `wheels3XX/` if present,
   otherwise from the internet
4. time each model on this Pi
5. list the cameras it finds

Then:
```bash
cd ~/nerisal && source ~/et/bin/activate
```

## 2. Fresh SD card (if the Pi has no OS)
1. Raspberry Pi Imager → Device: your Pi → OS: **Raspberry Pi OS (64-bit)** → your SD card.
2. Customisation: hostname `nerisal`, a username and password, Wi-Fi (country IN), **enable SSH**.
3. Boot the Pi, then from the laptop: `ssh <user>@nerisal.local` (or use the Pi's IP from `hostname -I`).

## 3. Cameras
| Camera | Check | Run with |
|---|---|---|
| Pi Camera Module | `rpicam-hello -t 2000` | `--source rpicam` |
| Pi Camera via picamera2 | needs `python3-picamera2` (apt) | `--source picam` |
| USB webcam | `ls /dev/video*` | `--source 0` |
| Video file | – | `--source clip.mp4` |
| Image folder | – | `--source images/` |

`picamera2` is installed by apt, not pip. `setup.sh` creates the venv with `--system-site-packages` so the venv
can see it. For an older venv:
```bash
sed -i 's/include-system-site-packages = false/include-system-site-packages = true/' ~/et/pyvenv.cfg
```

## 4. Offline packages (prepare before the event, on a Pi with internet)
```bash
source ~/et/bin/activate
pip download executorch opencv-python-headless torchvision pillow -d wheels313
pip download executorch opencv-python-headless torchvision pillow -d wheels311 \
  --only-binary=:all: --platform manylinux_2_28_aarch64 --python-version 3.11
```
`wheels313` is for the current Raspberry Pi OS (Python 3.13), `wheels311` for the previous one (Python 3.11).

## 5. Useful checks
```bash
hostname -I               # Pi address for the dashboard
vcgencmd measure_temp     # check for thermal throttling during long runs
sudo shutdown now         # always shut down before unplugging
```

## 6. Run the live monitor
```bash
cd ~/nerisal && source ~/et/bin/activate
python nerisal_live.py --source rpicam --line 0.6
```
Open `http://<pi-ip>:8000`, press **C**, click the four corners of a measured floor square (near left, near right,
far right, far left) and enter its size. INT8 head models are used automatically when present.
Mount the camera about 2–2.5 m high, angled 30–45° down, not straight down.

| Option | Default | Meaning |
|---|---|---|
| `--line 0.6` | off | counting line at 60% of the frame height |
| `--area 25` | none | floor area in m² if you skip calibration (average density only) |
| `--esc-every 4` | 4 s | rest between accurate passes (raise on a Pi 4) |
| `--hot-temp 76` | 76 °C | above this, the accurate tier slows down |
| `--fp32` | off | use FP32 head models even if INT8 ones exist |

## Expected speed (measured, our Pi 5)
| Model | FP32 | INT8 |
|---|---|---|
| head320 | 29.4 ms | 21.2 ms |
| head640 | 141.4 ms | 100.9 ms |
| DM-Count 432×240, per tile | 720 ms | – |
| Live view while the accurate tier runs | – | about 11 frames per second |
| Hybrid accurate count | – | 3.5 s alone, about 8.7 s alongside the live view |

A Pi 4 (Cortex-A72) is expected to be roughly 2–3× slower; `setup.sh` measures the real numbers.
