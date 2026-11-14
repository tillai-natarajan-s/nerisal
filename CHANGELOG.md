# Changelog

## Finale version (14 November 2026)

**Speed**
- DM-Count density model quantised to INT8: 768 → 157 ms per tile on the Pi 5 (4.9× faster), 86 → 22 MB,
  1.1% mean count difference from FP32 on 23 photos ([notebook 05](notebooks/05_dmcount_int8_colab.ipynb)).
- All three models now INT8; the live accurate count takes about 1.1 s instead of 2.4 s.
- The accurate tier runs in its own process: live view 3 → 17–24 frames per second.

**Measurement**
- Self-calibration from head sizes: people per m² with no clicks and no tape.
- Minimum-crowd rule: severe grades only when enough people fill the hotspot window at grade E.
- Heads-only live mode for close indoor cameras (`--live-heads-only`).
- Close-up scenes skip the density model, so walls and posters are not counted.
- Count and grade steadied: time-based smoothing, grade hysteresis, results dropped when the view changes.

**For staff**
- Spoken alarm at grade E and F, in English, Hindi or Tamil.
- Automatic incident record with blurred snapshots and a printable report (`/report`).
- *Test a file*: count any photo or video from the dashboard.
- Phone access by QR code; runs on a monitor attached to the Pi with no laptop.
- Start on power-up (`setup/install_autostart.sh`).

**Fixes**
- Crashes when switching between uploaded files and the camera.
- False grade F when the view could not be measured; the dashboard now says why instead.

## Day before the finale (13 November 2026)

- Hybrid counter: head detector near, density model far, per region, using a perspective fit.
- Benchmark on 18 hand-counted Indian photos: median error 6% (tiled detection alone 24%).
- head320 and head640 quantised to INT8 (28% faster, counts within 3%).
- Live engine v2 with floor calibration, hotspot and heat map; new dashboard.

## Earlier

- head320 and head640 trained on CrowdHuman head boxes (Kaggle).
- DM-Count exported to ExecuTorch at 256×144 and 432×240.
- First live engine, batch tool, setup script.
