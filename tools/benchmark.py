#!/usr/bin/env python3
"""
NERISAL accuracy benchmark: run several counting methods on images with known true counts.

True counts come from either
  --truth counts.csv            CSV with columns: image,count   (our hand-counted set)
  --truth path/to/ground_truth  ShanghaiTech folder of GT_IMG_*.mat files

Methods (repeat --method as often as you like):
  yolo:<model.pte>:<imgsz>:<target>[:<grid>]     target = person | head ; grid 0 = single pass
  dm:<model.pte>:<W>x<H>:<grid>                   DM-Count, grid x grid tiles

Example (ShanghaiTech B test set, three methods):
  python tools/benchmark.py --images SHB/test_data/images --truth SHB/test_data/ground_truth \\
     --method yolo:yolov8n.pte:640:person:0 \\
     --method yolo:head640.pte:640:head:3 \\
     --method dm:dmcount_qnrf_256x144.pte:256x144:4 \\
     --out results/benchmark/shb

Writes <out>_per_image.csv and <out>_summary.csv and prints the summary table.
Run from the repository root (it imports src/nerisal.py).
"""
import argparse, csv, glob, os, statistics, sys, time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

BANDS = [(0, 25, "light (<25)"), (25, 100, "medium (25-99)"), (100, 10 ** 9, "dense (100+)")]
EXTS = ("jpg", "jpeg", "png", "webp", "JPG", "JPEG", "PNG")


# ------------------------------------------------------------------ truth ---
def load_truth(path, images):
    """Return {image_basename: true_count}."""
    truth = {}
    if os.path.isfile(path) and path.lower().endswith(".csv"):
        with open(path) as f:
            for row in csv.DictReader(f):
                truth[os.path.basename(row["image"].strip())] = float(row["count"])
        return truth
    from scipy.io import loadmat                       # ShanghaiTech ground truth
    for img in images:
        b = os.path.splitext(os.path.basename(img))[0]   # IMG_12
        mat = os.path.join(path, f"GT_{b}.mat")
        if os.path.exists(mat):
            m = loadmat(mat)
            pts = m["image_info"][0, 0][0, 0][0]          # N x 2 head points
            truth[os.path.basename(img)] = float(len(pts))
    return truth


# ---------------------------------------------------------------- methods ---
class YoloMethod:
    def __init__(self, spec):
        from nerisal import Detector
        parts = spec.split(":")
        model, imgsz, target = parts[1], int(parts[2]), parts[3]
        self.grid = int(parts[4]) if len(parts) > 4 else 0
        self.det = Detector(model, imgsz, args.conf, target)
        self.det.run(np.zeros((480, 640, 3), np.uint8))   # warm-up
        self.name = f"yolo {os.path.basename(model)} @{imgsz} " + (f"tiled {self.grid}x{self.grid}" if self.grid
                                                                     else "single")

    def __call__(self, img):
        t = time.perf_counter()
        b, s, c, _ = self.det.detect(img, tiles=self.grid > 0, grid=max(self.grid, 2))
        return float(len(b)), (time.perf_counter() - t) * 1000


class DMMethod:
    def __init__(self, spec):
        from nerisal import DMCounter
        _, model, size, grid = spec.split(":")
        w, h = map(int, size.lower().split("x"))
        self.dm = DMCounter(model, (w, h), int(grid))
        self.dm.count(np.zeros((480, 640, 3), np.uint8))   # warm-up
        self.name = f"dmcount {os.path.basename(model)} tiles {grid}x{grid}"

    def __call__(self, img):
        c, _, ms = self.dm.count(img)
        return c, ms


def make_method(spec):
    kind = spec.split(":")[0]
    if kind == "yolo":
        return YoloMethod(spec)
    if kind == "dm":
        return DMMethod(spec)
    raise SystemExit(f"unknown method '{spec}'")


# ---------------------------------------------------------------- metrics ---
def metrics(pairs):
    """pairs: list of (true, predicted)."""
    if not pairs:
        return {}
    err = [p - t for t, p in pairs]
    ae = [abs(e) for e in err]
    pct = [abs(p - t) / max(t, 1) * 100 for t, p in pairs]
    return {"n": len(pairs), "MAE": statistics.mean(ae), "RMSE": (statistics.mean([e * e for e in err])) ** 0.5,
            "mean_pct_err": statistics.mean(pct), "median_pct_err": statistics.median(pct),
            "bias": statistics.mean(err)}           # negative = undercounts


def main():
    global args
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--method", action="append", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N images (quick test)")
    ap.add_argument("--out", default="results/benchmark/run")
    args = ap.parse_args()

    images = sorted(f for e in EXTS for f in glob.glob(os.path.join(args.images, f"*.{e}")))
    truth = load_truth(args.truth, images)
    images = [i for i in images if os.path.basename(i) in truth]
    if args.limit:
        images = images[:args.limit]
    if not images:
        raise SystemExit("No images with a true count found. Check --images and --truth.")
    print(f"{len(images)} images with true counts "
          f"(mean {statistics.mean(truth[os.path.basename(i)] for i in images):.1f} people per image)\n")

    methods = [make_method(s) for s in args.method]
    per_image, results = [], {m.name: [] for m in methods}
    times = {m.name: [] for m in methods}
    for k, path in enumerate(images, 1):
        img = cv2.imread(path)
        if img is None:
            print("skip unreadable", path); continue
        t = truth[os.path.basename(path)]
        row = {"image": os.path.basename(path), "true": t}
        for m in methods:
            pred, ms = m(img)
            results[m.name].append((t, pred)); times[m.name].append(ms)
            row[m.name] = round(pred, 1); row[m.name + " ms"] = round(ms, 1)
        per_image.append(row)
        if k % 20 == 0 or k == len(images):
            print(f"  {k}/{len(images)} images done", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + "_per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_image[0].keys())); w.writeheader(); w.writerows(per_image)

    summary = []
    for m in methods:
        for lo, hi, band in [(0, 10 ** 9, "ALL")] + BANDS:
            pairs = [(t, p) for t, p in results[m.name] if lo <= t < hi]
            r = metrics(pairs)
            if r:
                summary.append({"method": m.name, "band": band, **{k: round(v, 2) if isinstance(v, float) else v
                                                                    for k, v in r.items()},
                                "avg_ms": round(statistics.mean(times[m.name]), 1)})
    with open(args.out + "_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)

    print(f"\n{'method':48s}{'band':16s}{'n':>5s}{'MAE':>8s}{'RMSE':>8s}{'mean%':>8s}{'bias':>8s}{'ms':>9s}")
    for s in summary:
        print(f"{s['method'][:47]:48s}{s['band']:16s}{s['n']:5d}{s['MAE']:8.1f}{s['RMSE']:8.1f}"
              f"{s['mean_pct_err']:8.1f}{s['bias']:8.1f}{s['avg_ms']:9.1f}")
    print(f"\nSaved {args.out}_per_image.csv and {args.out}_summary.csv")
    print("Lower MAE / mean% is better. Negative bias = the method undercounts.")


if __name__ == "__main__":
    main()
