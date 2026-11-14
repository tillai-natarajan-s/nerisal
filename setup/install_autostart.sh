#!/usr/bin/env bash
# NERISAL: start by itself when the Pi powers on.
#   - the engine runs as a system service (restarts itself if it ever stops)
#   - the desktop logs in automatically and opens the dashboard full screen on the monitor
#   - the spoken alarm is on from the start
# Run once:   bash ~/nerisal/install_autostart.sh            (needs internet only for the voice packages)
# Undo:       bash ~/nerisal/install_autostart.sh remove
set -u
U="$(whoami)"; H="$HOME"; N="$H/nerisal"; PY="$H/et/bin/python"
OPTS="${NERISAL_OPTS:---source 0 --area 15 --stream-w 800}"
LANGS="${NERISAL_LANG:-en,ta}"

if [ "${1:-}" = "remove" ]; then
  sudo systemctl disable --now nerisal 2>/dev/null
  sudo rm -f /etc/systemd/system/nerisal.service
  rm -f "$H/.config/autostart/nerisal-kiosk.desktop"
  sudo systemctl daemon-reload
  echo "Auto-start removed. Start by hand again with: cd ~/nerisal && source ~/et/bin/activate && python nerisal_live.py --source 0"
  exit 0
fi

[ -x "$PY" ] || { echo "!! $PY not found: run setup.sh first"; exit 1; }
[ -f "$N/nerisal_live.py" ] || { echo "!! $N/nerisal_live.py not found"; exit 1; }

echo "== 1. Engine as a service =="
sudo tee /etc/systemd/system/nerisal.service >/dev/null <<EOF
[Unit]
Description=NERISAL crowd safety engine
After=network.target

[Service]
User=$U
WorkingDirectory=$N
ExecStart=$PY $N/nerisal_live.py $OPTS
Restart=always
RestartSec=5
Nice=-2

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now nerisal && echo "  OK  engine service started (logs: journalctl -u nerisal -f)"

echo "== 2. Desktop logs in by itself =="
sudo raspi-config nonint do_boot_behaviour B4 && echo "  OK  desktop autologin"

echo "== 3. Dashboard opens full screen on the monitor =="
cat > "$H/nerisal_kiosk.sh" <<EOF
#!/bin/bash
# wait until the engine answers, then open the dashboard full screen
for i in \$(seq 1 90); do curl -s -o /dev/null http://localhost:8000/stats && break; sleep 2; done
B=\$(command -v chromium || command -v chromium-browser)
exec \$B --kiosk --noerrdialogs --disable-infobars --password-store=basic --no-first-run \\
  --autoplay-policy=no-user-gesture-required --enable-speech-dispatcher \\
  "http://localhost:8000/?sound=1&lang=$LANGS"
EOF
chmod +x "$H/nerisal_kiosk.sh"
mkdir -p "$H/.config/autostart"
cat > "$H/.config/autostart/nerisal-kiosk.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=NERISAL dashboard
Exec=$H/nerisal_kiosk.sh
X-GNOME-Autostart-enabled=true
EOF
echo "  OK  kiosk browser on login"

echo "== 4. Voices for the spoken alarm (English, Hindi, Tamil) =="
if sudo apt-get install -y -q speech-dispatcher espeak-ng >/dev/null 2>&1; then
  echo "  OK  speech-dispatcher + espeak-ng installed"
else
  echo "  --  could not install voices (no internet?): the beep still works, speech may be silent"
fi

echo
echo "Done. Reboot to test:   sudo reboot"
echo "  engine status:  systemctl status nerisal       live log:  journalctl -u nerisal -f"
echo "  stop for now:   sudo systemctl stop nerisal ; pkill -f nerisal_kiosk ; pkill chromium"
echo "  change options: NERISAL_OPTS='--source crowd.mp4 --area 50' bash install_autostart.sh"
