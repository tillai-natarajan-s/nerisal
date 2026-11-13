#!/usr/bin/env python3
"""
Dump the raw outputs of every counting model for a folder of images, so the
hybrid (head + density) rule can be tuned offline without re-running models.

  python dump_raw.py --images india_set --out raw_india.npz

Saves per image: head640 boxes+scores (single and tiled 3x3, conf >= 0.10) and
DM-Count density maps at several scales (block-summed to 1/8 resolution).
Takes ~15 s per image on a Pi 5.
"""
import argparse, glob, os, sys, time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "..", "src")):
    sys.path.insert(0, p)
from nerisal import Detector, DMCounter, load_image  # noqa: E402

EXTS = ("jpg", "jpeg", "png", "webp")


def small_map(dmap, f=8):
    """Area-preserving downsample: each output cell = sum of an f x f block."""
    h, w = dmap.shape
    H, W = max(1, h // f), max(1, w // f)
    s = cv2.resize(dmap, (W, H), interpolation=cv2.INTER_AREA)
    if s.sum() > 0:
        s *= dmap.sum() / s.sum()
    return s.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="raw.npz")
    ap.add_argument("--head", default="head640.pte")
    ap.add_argument("--dm432", default="dmcount_qnrf_432x240.pte")
    ap.add_argument("--dm256", default="dmcount_qnrf_256x144.pte")
    ap.add_argument("--quick", action="store_true", help="skip the slow 3x3 DM variant")
    a = ap.parse_args()

    files = sorted(f for e in EXTS for f in glob.glob(os.path.join(a.images, f"*.{e}")))
    det = Detector(a.head, 640, 0.10, "head")
    det.run(np.zeros((480, 640, 3), np.uint8))
    dm432 = {g: DMCounter(a.dm432, (432, 240), g) for g in (1, 2, 3)}
    shared = dm432[1].method                       # one loaded program for all grids
    for g in (2, 3):
        dm432[g] = DMCounter(None, (432, 240), g, method=shared)
    dm256 = DMCounter(a.dm256, (256, 144), 4)
    dm256.count(np.zeros((480, 640, 3), np.uint8))
    dm432[1].count(np.zeros((480, 640, 3), np.uint8))

    out = {}
    for k, f in enumerate(files, 1):
        img = load_image(f)
        if img is None:
            continue
        name = os.path.splitext(os.path.basename(f))[0]
        t0 = time.perf_counter()
        h, w = img.shape[:2]
        out[f"{name}__size"] = np.array([w, h])
        b, s, _, _ = det.detect(img, tiles=False)
        out[f"{name}__single"] = np.hstack([b.numpy(), s.numpy()[:, None]]).astype(np.float32)
        b, s, _, _ = det.detect(img, tiles=True, grid=3)
        out[f"{name}__tiled"] = np.hstack([b.numpy(), s.numpy()[:, None]]).astype(np.float32)
        grids = (1, 2) if a.quick else (1, 2, 3)
        for g in grids:
            c, dmap, _ = dm432[g].count(img)
            out[f"{name}__dm432g{g}"] = small_map(dmap)
        c, dmap, _ = dm256.count(img)
        out[f"{name}__dm256g4"] = small_map(dmap)
        print(f"  {k}/{len(files)} {name}  {time.perf_counter() - t0:.1f} s", flush=True)
    np.savez_compressed(a.out, **out)
    print(f"saved {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
