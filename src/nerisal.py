#!/usr/bin/env python3
"""
NERISAL - Edge crowd-pressure meter (Team Iconic)
ExecuTorch (XNNPACK) + YOLOv8n on Raspberry Pi 5

Counts people (or vehicles) in images / a video, converts the count to crowd
density and a Fruin Level-of-Service grade (A-F), blurs heads for privacy,
and writes annotated images + results.csv + report.html.

Examples:
  python nerisal.py --input images/                       # folder of images
  python nerisal.py --input clip.mp4 --every 10           # every 10th frame
  python nerisal.py --input images/ --area 60             # 60 m^2 visible floor -> grade
  python nerisal.py --input images/ --tiles               # tiled mode for dense crowds
  python nerisal.py --input images/ --target vehicle      # vehicle counting
"""
import argparse, csv, glob, html, os, statistics, time

import cv2
import numpy as np
import torch
import torchvision
from executorch.runtime import Runtime

# COCO class ids
TARGETS = {
    "person": {0: "person"},
    "vehicle": {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"},
}

# Fruin walkway Level of Service, by density (persons per m^2): upper bounds
FRUIN = [
    (0.31, "A", "Free flow"),
    (0.43, "B", "Minor conflicts"),
    (0.72, "C", "Restricted speed"),
    (1.08, "D", "Crowded - monitor closely"),
    (2.17, "E", "Severe - hold entry / open extra exit"),
    (float("inf"), "F", "Critical - crush risk, intervene now"),
]
GRADE_COLOR = {"A": (80, 180, 60), "B": (80, 180, 60), "C": (0, 200, 230),
               "D": (0, 140, 255), "E": (0, 60, 240), "F": (0, 0, 200)}


def fruin(density):
    for limit, grade, action in FRUIN:
        if density <= limit:
            return grade, action


def letterbox(img, size):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))
    return canvas, r, left, top


def merge_boxes(b, s, c, iou_thr=0.5, contain_thr=0.8):
    """Greedy merge, highest score first. A box is dropped if it overlaps a kept
    box of the same class by IoU > iou_thr, OR if > contain_thr of its own area
    lies inside a kept box (catches head-only / partial duplicates that plain
    NMS misses because their IoU with the full-body box is small)."""
    if len(b) == 0:
        return torch.zeros(0, dtype=torch.long)
    order = s.argsort(descending=True)
    area = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    keep = []
    for i in order.tolist():
        if keep:
            k = torch.tensor(keep)
            same = c[k] == c[i]
            lt = torch.maximum(b[k, :2], b[i, :2])
            rb = torch.minimum(b[k, 2:], b[i, 2:])
            inter = (rb - lt).clamp(min=0).prod(1)
            iou = inter / (area[k] + area[i] - inter + 1e-9)
            inside_kept = inter / (area[i] + 1e-9)        # i mostly inside a kept box
            kept_inside = inter / (area[k] + 1e-9)        # a kept box mostly inside i
            dup = same & ((iou > iou_thr) | (inside_kept > contain_thr) | (kept_inside > contain_thr))
            if dup.any():
                continue
        keep.append(i)
    return torch.tensor(keep, dtype=torch.long)


class Detector:
    def __init__(self, model_path, size, conf, target):
        self.method = Runtime.get().load_program(model_path).load_method("forward")
        self.size, self.conf = size, conf
        self.ids = list(TARGETS[target].keys())
        self.names = TARGETS[target]
        self.weak_conf, self.measure_weak, self.weak = 0.05, False, 0

    def run(self, img):
        """Returns boxes (N,4 xyxy in img coords), scores (N), class ids (N), ms."""
        lb, r, dx, dy = letterbox(img, self.size)
        x = torch.from_numpy(np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1)))
        x = x.float().unsqueeze(0) / 255.0
        t0 = time.perf_counter()
        out = self.method.execute([x])[0]
        ms = (time.perf_counter() - t0) * 1000
        p = out[0].T                                   # (anchors, 4 + 80)
        cls = p[:, 4:][:, self.ids]
        if self.measure_weak:
            # Weak evidence: candidates above a very low score, de-duplicated.
            # In packed crowds YOLO emits many sub-threshold "maybe person"
            # signals even when few pass the main threshold.
            ws, _ = cls.max(1)
            wk = ws > self.weak_conf
            if wk.sum() > 0:
                wb, wsc = p[wk, :4], ws[wk]
                top = wsc.argsort(descending=True)[:3000]
                wb, wsc = wb[top], wsc[top]
                wxy = torch.cat([wb[:, :2] - wb[:, 2:] / 2, wb[:, :2] + wb[:, 2:] / 2], 1)
                self.weak = int(len(torchvision.ops.nms(wxy, wsc, 0.5)))
            else:
                self.weak = 0
        score, idx = cls.max(1)
        keep = score > self.conf
        b, score, idx = p[keep, :4], score[keep], idx[keep]
        xyxy = torch.cat([b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2], 1)
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - dx) / r
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - dy) / r
        cid = torch.tensor(self.ids)[idx] if len(idx) else idx
        return xyxy, score, cid, ms

    def detect(self, img, tiles=False, grid=2, iou=0.5, contain=0.8, seam=4):
        if not tiles:
            self.measure_weak = True
            b, s, c, ms = self.run(img)
            self.measure_weak = False
            if len(b):
                k = torchvision.ops.batched_nms(b, s, c, iou)
                b, s, c = b[k], s[k], c[k]
            return b, s, c, ms

        # Seam-aware tiling: full frame + grid x grid overlapping tiles.
        h, w = img.shape[:2]
        ov = 0.25                                       # tile overlap fraction
        tw = int(w / (grid - (grid - 1) * ov))
        th = int(h / (grid - (grid - 1) * ov))
        xs = [round(i * (w - tw) / (grid - 1)) for i in range(grid)]
        ys = [round(i * (h - th) / (grid - 1)) for i in range(grid)]
        crops = [(0, 0, w, h)] + [(x0, y0, x0 + tw, y0 + th) for y0 in ys for x0 in xs]

        B, S, C, ms = [], [], [], 0.0
        for x0, y0, x1, y1 in crops:
            self.measure_weak = (x1 - x0, y1 - y0) == (w, h)
            b, s, c, t = self.run(img[y0:y1, x0:x1])
            self.measure_weak = False
            ms += t
            if len(b) and (x1 - x0, y1 - y0) != (w, h):
                # Drop boxes cut by an INNER tile edge (partial bodies / heads).
                # Edges that are also image edges are kept.
                cut = torch.zeros(len(b), dtype=torch.bool)
                if x0 > 0:  cut |= b[:, 0] <= seam
                if y0 > 0:  cut |= b[:, 1] <= seam
                if x1 < w:  cut |= b[:, 2] >= (x1 - x0) - seam
                if y1 < h:  cut |= b[:, 3] >= (y1 - y0) - seam
                b, s, c = b[~cut], s[~cut], c[~cut]
            b = b.clone()
            b[:, [0, 2]] += x0
            b[:, [1, 3]] += y0
            B.append(b); S.append(s); C.append(c)
        b, s, c = torch.cat(B), torch.cat(S), torch.cat(C)
        k = merge_boxes(b, s, c, iou, contain)
        return b[k], s[k], c[k], ms


def annotate(img, boxes, classes, names, label, color, blur_heads):
    h, w = img.shape[:2]
    for (x1, y1, x2, y2), c in zip(boxes.int().tolist(), classes.tolist()):
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if blur_heads and names[c] == "person" and x2 > x1 and y2 > y1:
            hy2 = y1 + max(1, (y2 - y1) // 5)          # top fifth of the body box
            roi = img[y1:hy2, x1:x2]
            if roi.size:
                img[y1:hy2, x1:x2] = cv2.GaussianBlur(roi, (31, 31), 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), (60, 220, 60), 2)
    cv2.rectangle(img, (0, 0), (w, 44), (0, 0, 0), -1)
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return img


IMG_EXT = ("jpg", "jpeg", "png", "webp", "avif", "JPG", "JPEG", "PNG", "WEBP", "AVIF")


def load_image(path):
    """OpenCV first; Pillow fallback for formats OpenCV can't read (e.g. AVIF)."""
    img = cv2.imread(path)
    if img is not None:
        return img
    try:
        from PIL import Image
        with Image.open(path) as im:
            return cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"  could not read {os.path.basename(path)}: {e}")
        return None


def texture(img):
    """Edge density of the frame (0..1). Packed crowds are uniformly 'busy'."""
    h, w = img.shape[:2]
    g = cv2.cvtColor(cv2.resize(img, (640, max(1, int(640 * h / w)))), cv2.COLOR_BGR2GRAY)
    return float(cv2.Canny(g, 80, 160).mean() / 255)


def frames(inp, every):
    vid_ext = (".mp4", ".avi", ".mov", ".mkv")
    if os.path.isdir(inp):
        files = sorted(f for e in IMG_EXT for f in glob.glob(os.path.join(inp, f"*.{e}")))
        for f in files:
            yield os.path.basename(f), load_image(f)
    elif inp.lower().endswith(vid_ext):
        cap, i = cv2.VideoCapture(inp), 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i % every == 0:
                yield f"frame_{i:05d}.jpg", fr
            i += 1
    else:
        yield os.path.basename(inp), load_image(inp)


def write_report(path, rows, summary):
    trs = "".join(
        f"<tr><td>{html.escape(r['name'])}</td><td>{r['count']}</td><td>{r['breakdown']}</td>"
        f"<td>{r['density']}</td><td>{r['grade']}</td><td>{r['saturated']}</td><td>{r['ms']}</td></tr>" for r in rows)
    imgs = "".join(f"<figure><img src='{html.escape(r['out'])}'><figcaption>{html.escape(r['name'])}"
                   f"</figcaption></figure>" for r in rows[:6])
    srows = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in summary.items())
    with open(path, "w") as f:
        f.write(f"""<!doctype html><meta charset=utf-8><title>NERISAL report</title>
<style>body{{font-family:sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#222}}
table{{border-collapse:collapse;width:100%;margin:12px 0}}td,th{{border:1px solid #ccc;padding:6px 8px;text-align:left}}
th{{background:#f3f3f3}}figure{{display:inline-block;width:48%;margin:1%}}img{{width:100%}}</style>
<h1>NERISAL &mdash; Edge crowd-pressure report</h1><p>Team Iconic &middot; ExecuTorch on Raspberry Pi 5</p>
<h2>Summary</h2><table>{srows}</table>
<h2>Per-frame results</h2><table><tr><th>Frame</th><th>Count</th><th>Breakdown</th>
<th>Density (p/m&sup2;)</th><th>Fruin grade</th><th>Saturated</th><th>Inference (ms)</th></tr>{trs}</table>
<h2>Sample detections</h2>{imgs}""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pte")
    ap.add_argument("--input", required=True, help="image, folder of images, or video")
    ap.add_argument("--target", choices=list(TARGETS), default="person")
    ap.add_argument("--area", type=float, default=None, help="visible floor area in m^2 (enables Fruin grade)")
    ap.add_argument("--imgsz", type=int, default=640, help="must match the exported model")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--tiles", action="store_true", help="seam-aware tiled detection for dense/distant crowds")
    ap.add_argument("--grid", type=int, default=2, help="tiles per side in tiled mode (2 = 2x2, 3 = 3x3)")
    ap.add_argument("--every", type=int, default=5, help="video: process every Nth frame")
    ap.add_argument("--no-blur", action="store_true")
    ap.add_argument("--tex-thr", type=float, default=0.22, help="saturated if frame edge density >= this...")
    ap.add_argument("--sat-count", type=int, default=15, help="...and fewer than this many people detected")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    det = Detector(a.model, a.imgsz, a.conf, a.target)
    det.run(np.zeros((480, 640, 3), np.uint8))      # warm-up, not timed

    rows, times = [], []
    for name, img in frames(a.input, a.every):
        if img is None:
            print(f"skip {name} (unreadable)")
            continue
        boxes, scores, cls, ms = det.detect(img, tiles=a.tiles, grid=a.grid)
        n = len(boxes)
        counts = {}
        for c in cls.tolist():
            counts[det.names[c]] = counts.get(det.names[c], 0) + 1
        breakdown = ", ".join(f"{k}:{v}" for k, v in counts.items()) or "-"
        weak = det.weak
        tex = texture(img)
        # Fail-safe: the scene looks packed (busy texture everywhere) but the
        # detector sees few people -> it is overwhelmed, not the scene empty.
        sat = a.target == "person" and tex >= a.tex_thr and n < a.sat_count
        density, grade, action, color = "-", "-", "", (255, 255, 255)
        if a.area and a.target == "person":
            d = n / a.area
            grade, action = fruin(d)
            density, color = f"{d:.2f}", GRADE_COLOR[grade]
        if sat:
            # Fail-safe: the detector is overwhelmed, so the count is only a lower
            # bound. Never report a calm grade for a scene we cannot count.
            if grade in ("-", "A", "B", "C", "D"):
                grade = "E*"
            action, color = "SATURATED - too dense to count, treat as E or worse", GRADE_COLOR["F"]
        cnt = f">={n}" if sat else f"{n}"
        label = f"{a.target.upper()}: {cnt}" + (f" | LOS {grade} - {action}" if action else "") + f" | {ms:.0f} ms"
        out_name = "out_" + os.path.splitext(name)[0] + ".jpg"
        cv2.imwrite(os.path.join(a.out, out_name),
                    annotate(img, boxes, cls, det.names, label, color, not a.no_blur))
        times.append(ms)
        rows.append({"name": name, "count": n, "breakdown": breakdown, "density": density,
                     "grade": grade, "ms": f"{ms:.1f}", "weak": weak, "texture": f"{tex:.3f}", "saturated": "YES" if sat else "",
                     "out": out_name})
        print(f"{name:28s} count={n:4d}  tex={tex:.3f}  {'SATURATED ' if sat else '          '}"
              f"density={density}  LOS={grade:2s}  {ms:7.1f} ms")

    if not times:
        print("No frames processed.")
        return
    avg = statistics.mean(times)
    summary = {
        "Model": f"{os.path.basename(a.model)} (ExecuTorch, XNNPACK), input {a.imgsz}px",
        "Mode": f"seam-aware tiled (full + {a.grid}x{a.grid})" if a.tiles else "single pass",
        "Target": a.target, "Frames processed": len(times),
        "Total counted": sum(r["count"] for r in rows),
        "Frames flagged saturated": sum(1 for r in rows if r["saturated"]),
        "Avg inference / frame": f"{avg:.1f} ms",
        "Min / max": f"{min(times):.1f} / {max(times):.1f} ms",
        "Approx throughput": f"{1000 / avg:.2f} FPS",
    }
    with open(os.path.join(a.out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "count", "breakdown", "density", "grade", "ms", "weak", "texture", "saturated", "out"])
        w.writeheader(); w.writerows(rows)
    write_report(os.path.join(a.out, "report.html"), rows, summary)
    print("\n" + "\n".join(f"{k:24s} {v}" for k, v in summary.items()))
    print(f"\nSaved: {a.out}/report.html, results.csv, annotated images")


if __name__ == "__main__":
    main()
