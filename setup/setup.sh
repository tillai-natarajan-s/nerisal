#!/usr/bin/env bash
# NERISAL one-command setup for an unknown Raspberry Pi (4 or 5).
# Run from the pendrive folder:   bash setup.sh
# Uses offline wheels from ./wheels<pyver>/ if present, otherwise the internet.
set -u
KIT="$(cd "$(dirname "$0")" && pwd)"
WORK="$HOME/nerisal"
VENV="$HOME/et"
ok(){ echo -e "  \e[32mOK\e[0m  $*"; }
bad(){ echo -e "  \e[31m!!\e[0m  $*"; }

echo "== 1. Hardware and OS =="
MODEL=$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo unknown)
ARCH=$(uname -m)
PYV=$(python3 -c 'import sys;print(f"{sys.version_info[0]}{sys.version_info[1]}")')
echo "  Board: $MODEL"; echo "  Arch:  $ARCH"; echo "  Python: 3.${PYV:1}"
[ "$ARCH" = "aarch64" ] && ok "64-bit OS" || bad "NOT 64-bit ($ARCH). ExecuTorch wheels need aarch64. Ask organisers for 64-bit Pi OS or use our SD card."
free -h | awk '/Mem/{print "  RAM: "$2}'

echo "== 2. Copy kit to $WORK =="
mkdir -p "$WORK"
# works from a flat pendrive folder or from the repo (src/, models/, tools/)
for d in "$KIT" "$KIT/src" "$KIT/models" "$KIT/../src" "$KIT/../models"; do
  [ -d "$d" ] && cp -u "$d"/*.py "$d"/dashboard.html "$d"/*.pte "$WORK"/ 2>/dev/null
done
ls "$WORK"/*.pte >/dev/null 2>&1 && ok "models copied" || bad "no .pte models found in kit folder"

echo "== 3. Python environment =="
if [ ! -d "$VENV" ]; then
  python3 -m venv --system-site-packages "$VENV" && ok "venv created at $VENV"
else
  ok "venv exists"
fi
source "$VENV/bin/activate"
WH="$KIT/wheels$PYV"; [ -d "$WH" ] || WH="$KIT/../wheels$PYV"
PKGS="executorch opencv-python-headless torchvision pillow"
if [ -d "$WH" ]; then
  echo "  installing OFFLINE from $WH"
  pip install --no-index --find-links "$WH" $PKGS
else
  echo "  no wheels$PYV folder, installing from internet (slow networks: resumes after drops)"
  pip install --timeout 120 --retries 10 --resume-retries 20 $PKGS || \
    pip install --timeout 120 --retries 10 $PKGS        # older pip without --resume-retries
fi

echo "== 4. Checks =="
python - <<'EOF'
import importlib
for m in ("numpy", "cv2", "torch", "torchvision", "executorch.runtime"):
    try:
        mod = importlib.import_module(m); print(f"  OK  {m} {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"  !!  {m}: {e}")
EOF
cd "$WORK"
if ls head*.pte >/dev/null 2>&1; then
  python - <<'EOF'
import os, time, numpy as np, torch
from executorch.runtime import Runtime
for name, s in (("head320_int8.pte", 320), ("head640_int8.pte", 640), ("head320.pte", 320), ("head640.pte", 640)):
    if not os.path.exists(name):
        continue
    try:
        m = Runtime.get().load_program(name).load_method("forward")
        x = torch.rand(1, 3, s, s); m.execute([x])
        t = time.perf_counter()
        for _ in range(5): m.execute([x])
        print(f"  OK  {name}: {(time.perf_counter()-t)/5*1000:.0f} ms/frame on this Pi")
    except Exception as e:
        print(f"  !!  {name}: {e}")
EOF
fi

echo "== 5. Cameras =="
ls /dev/video* 2>/dev/null | head -3 | sed 's/^/  video device: /' || true
if command -v rpicam-hello >/dev/null; then
  rpicam-hello --list-cameras 2>/dev/null | grep -E "^[0-9]+ :" | sed 's/^/  Pi camera: /' || echo "  no Pi camera listed"
fi
echo
echo "Next:  cd $WORK && source $VENV/bin/activate"
echo "       python nerisal_live.py --source rpicam --line 0.6                      (Pi camera)"
echo "       python nerisal_live.py --source 0 --line 0.6                           (USB camera)"
echo "       python nerisal_live.py --source http://PHONE-IP:8080/video --line 0.6  (phone, IP Webcam app)"
echo "       then open http://$(hostname -I | awk '{print $1}'):8000 and press C to calibrate the floor"
