# Deploying NERISAL at a site

A guide for installing one NERISAL unit (camera + Raspberry Pi) at a venue such as a temple entrance, station
platform, stadium gate or event walkway.

## What one unit needs

| Item | Notes |
|---|---|
| Raspberry Pi 5 (4 GB or more) | A Pi 4 works, about 2–3 times slower |
| Cooling | Official Active Cooler or a small fan; sustained use heats the Pi to about 70 °C |
| Power | Official 27 W USB-C supply |
| Camera | Pi Camera Module, USB webcam, or an IP camera stream |
| Display (optional) | Any HDMI monitor, or view from laptops and phones on the local network |
| SD card | 32 GB, prepared with `setup/setup.sh` |

Internet is needed only once, to install packages. After that the unit runs with no network.

## Mounting the camera

- **Height:** 2.5 m or more. Higher is better.
- **Angle:** tilted down 20–35°, like CCTV. Not straight down: the models were trained on angled views.
- **Framing:** mostly floor and people. Avoid ceilings, bright lights and large printed text in view.
- **Fixed:** the camera must not move. Self-calibration adapts within about 30 s if it does.

## Choosing the settings for the site

| Site | Suggested options | Why |
|---|---|---|
| Walkway, platform, plaza (default) | `--source 0` | Hybrid counter, 3 m hotspot window, severe grades from about 10 people |
| Staircase, narrow corridor | `--peak-win 2` | Smaller window: severe grades from about 5 people |
| Close indoor room | `--live-heads-only` | No distant crowd, so no density model; avoids clutter being counted |
| Doorway or gate | `--line 0.5` | Counts people crossing in each direction |
| No camera tilt possible | `--no-auto-calib --area <m²>` | Average density from a known floor area |
| Exact square metres needed | press **C** on the dashboard | Manual calibration from four measured floor corners |
| Hot environment | `--esc-every 4` | Accurate count less often, lower temperature |

## Starting automatically

```bash
bash ~/nerisal/install_autostart.sh          # engine as a service, dashboard full screen on boot, alarm on
NERISAL_OPTS="--source 0 --peak-win 2" bash ~/nerisal/install_autostart.sh   # with site options
bash ~/nerisal/install_autostart.sh remove   # undo
```

After a power cut the unit comes back on its own in about 90 seconds.

## Viewing without any network

- **Monitor on the Pi:** the dashboard opens at `http://localhost:8000`, no network involved.
- **Phones with no router:** the Pi can broadcast its own local Wi-Fi (no internet) using a USB Wi-Fi adapter:
  `sudo nmcli device wifi hotspot ifname wlan1 ssid NERISAL password <password>`. Phones join it and scan the QR
  code on the dashboard.

## Daily checks

| Check | How |
|---|---|
| Running | dashboard shows *Live* and a moving clock |
| Temperature | bottom strip, keep below about 76 °C |
| Calibration | panel reads *measured automatically* or shows the manual calibration time |
| Incidents | *Results* → *Open the incident report* |
| Engine log | `journalctl -u nerisal -f` (auto-start) or the terminal window |
