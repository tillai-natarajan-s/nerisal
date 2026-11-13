#!/usr/bin/env python3
"""
NERISAL Live - real-time edge crowd-pressure monitor (Team Iconic)

Camera -> watch tier every frame: head320 (ExecuTorch, XNNPACK) -> smooth live view,
          line crossings, privacy blur
       -> accurate tier in a background thread: perspective-aware hybrid
          (head640 tiled for near people + DM-Count density for far / packed regions)
       -> floor calibration (homography) -> average AND peak density in people per m^2,
          hotspot, Fruin grade A-F, time-to-grade-E, saturation fail-safe
       -> browser dashboard served by the Pi itself (no internet, no extra packages)

Runs on Raspberry Pi 4 or 5. Needs only: opencv, numpy, executorch (+ nerisal.py
in the same folder).

Examples
  python nerisal_live.py --source rpicam                       # Pi Camera (rpicam-vid), our models
  python nerisal_live.py --source 0                            # USB webcam
  python nerisal_live.py --source http://PHONE-IP:8080/video   # phone as camera (IP Webcam app)
  python nerisal_live.py --source clip.mp4 --line 0.6          # video file, looped, counting line
  python nerisal_live.py --source images/ --hold 6             # image folder slideshow
  python nerisal_live.py --source clip.mp4 --mock              # laptop test, no ExecuTorch
Then open  http://<pi-ip>:8000  on the laptop and press C to calibrate the floor.
"""
import argparse, collections, csv, glob, json, multiprocessing as mp, os, socket, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

# ---------------------------------------------------------------- grading ----
FRUIN = [  # walkway Level of Service, persons per m^2 (upper bounds)
    (0.31, "A", "Free flow"),
    (0.43, "B", "Minor conflicts"),
    (0.72, "C", "Restricted speed - watch the trend"),
    (1.08, "D", "Crowded - monitor closely, prepare staff"),
    (2.17, "E", "Severe - hold entry, open an extra exit"),
    (float("inf"), "F", "Critical - crush risk, intervene now"),
]
E_LIMIT = 1.08
GRADE_HEX = {"A": "#2e9e5b", "B": "#5aa83c", "C": "#d9a400", "D": "#e07a10",
             "E": "#d64528", "E*": "#c0182b", "F": "#9b0d1f", "-": "#6b7280"}


def fruin(d):
    for lim, g, act in FRUIN:
        if d <= lim:
            return g, act


def texture(img):
    """Edge density (0..1). Packed crowds are uniformly busy."""
    h, w = img.shape[:2]
    g = cv2.cvtColor(cv2.resize(img, (640, max(1, int(640 * h / w)))), cv2.COLOR_BGR2GRAY)
    return float(cv2.Canny(g, 80, 160).mean() / 255)


# ---------------------------------------------------------------- sources ----
class Source:
    """Background grabber that always holds the newest frame."""

    def __init__(self, spec, width, height, hold):
        self.spec, self.w, self.h, self.hold = spec, width, height, hold
        self.frame, self.seq, self.lock, self.alive = None, 0, threading.Lock(), True
        self.scene = 0                         # changes only when the slideshow moves to a new image
        self.desc = spec
        self._open()
        threading.Thread(target=self._loop, daemon=True).start()

    def _open(self):
        s = self.spec
        self.kind = None
        if s == "picam":
            from picamera2 import Picamera2          # apt: python3-picamera2
            self.cam = Picamera2()
            cfg = self.cam.create_video_configuration(main={"size": (self.w, self.h), "format": "RGB888"})
            self.cam.configure(cfg)
            self.cam.start()
            self.kind, self.desc = "picam", f"Pi Camera {self.w}x{self.h}"
        elif s == "rpicam":
            # No Python camera library needed: stream MJPEG from the rpicam-vid CLI.
            import subprocess
            exe = "rpicam-vid" if subprocess.call(["which", "rpicam-vid"], stdout=subprocess.DEVNULL) == 0 \
                else "libcamera-vid"
            self.proc = subprocess.Popen([exe, "-t", "0", "-n", "--codec", "mjpeg", "--width", str(self.w),
                                          "--height", str(self.h), "--framerate", "15", "-o", "-"],
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
            self.buf = b""
            self.kind, self.desc = "rpicam", f"Pi Camera via {exe} {self.w}x{self.h}"
        elif s.isdigit():
            self.cap = cv2.VideoCapture(int(s))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
            if not self.cap.isOpened():
                raise SystemExit(f"Cannot open camera {s}. Try another index, or --source picam.")
            self.kind, self.desc = "usb", f"USB camera {s}"
        elif os.path.isdir(s):
            exts = ("jpg", "jpeg", "png", "webp", "JPG", "JPEG", "PNG")
            self.files = sorted(f for e in exts for f in glob.glob(os.path.join(s, f"*.{e}")))
            if not self.files:
                raise SystemExit(f"No images in {s}")
            self.kind, self.desc = "images", f"{len(self.files)} images from {s}"
        elif s.startswith(("http://", "https://", "rtsp://")):
            self.cap = cv2.VideoCapture(s)
            if not self.cap.isOpened():
                raise SystemExit(f"Cannot open stream {s}. Check the phone app is running and on the same Wi-Fi.")
            self.kind, self.desc = "stream", "network camera"
        else:
            self.cap = cv2.VideoCapture(s)
            if not self.cap.isOpened():
                raise SystemExit(f"Cannot open video {s}")
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
            self.kind, self.desc = "video", f"video {os.path.basename(s)} (looped)"

    def _put(self, f):
        if f is None:
            return
        with self.lock:
            self.frame, self.seq = f, self.seq + 1

    def _loop(self):
        i = 0
        while self.alive:
            if self.kind == "picam":
                self._put(self.cam.capture_array())
            elif self.kind == "rpicam":
                chunk = self.proc.stdout.read(65536)
                if not chunk:
                    time.sleep(0.05)
                    continue
                self.buf += chunk
                while True:
                    a = self.buf.find(b"\xff\xd8")
                    b = self.buf.find(b"\xff\xd9", a + 2)
                    if a < 0 or b < 0:
                        break
                    jpg, self.buf = self.buf[a:b + 2], self.buf[b + 2:]
                    self._put(cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR))
                if len(self.buf) > 4_000_000:
                    self.buf = b""
            elif self.kind in ("usb", "stream"):
                ok, f = self.cap.read()
                self._put(f if ok else None)
                if not ok:
                    time.sleep(0.05)
            elif self.kind == "images":
                f = cv2.imread(self.files[i % len(self.files)])
                with self.lock:
                    self.scene += 1
                t_end = time.time() + self.hold
                while time.time() < t_end and self.alive:   # keep feeding it like a still camera
                    self._put(f)
                    time.sleep(1 / 15)
                i += 1
            else:  # video, paced to real time, looped
                ok, f = self.cap.read()
                if not ok:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self._put(f)
                time.sleep(1.0 / self.fps)

    def latest(self):
        with self.lock:
            return self.frame, self.seq


# -------------------------------------------------------------- detectors ----
class ETDetector:
    """ExecuTorch YOLO via nerisal.py (same folder)."""

    def __init__(self, model, imgsz, conf, target, tiles, grid):
        from nerisal import Detector
        self.d = Detector(model, imgsz, conf, target)
        self.tiles, self.grid = tiles, grid
        self.names = self.d.names
        self.label = os.path.splitext(os.path.basename(model))[0] + (f" tiled {grid}x{grid}" if tiles else "")

    def __call__(self, img):
        b, s, c, ms = self.d.detect(img, tiles=self.tiles, grid=self.grid)
        return b.numpy(), s.numpy(), c.numpy().astype(int), ms


class HOGDetector:
    """Stand-in for testing without ExecuTorch (OpenCV's classic people detector)."""

    def __init__(self, scale=1.0, label="mock HOG"):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.scale, self.names, self.label = scale, {0: "person"}, label

    def __call__(self, img):
        t = time.perf_counter()
        small = cv2.resize(img, None, fx=self.scale, fy=self.scale)
        rects, w = self.hog.detectMultiScale(small, winStride=(8, 8), padding=(8, 8), scale=1.05)
        ms = (time.perf_counter() - t) * 1000
        if len(rects) == 0:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), ms
        r = np.array(rects, float) / self.scale
        b = np.c_[r[:, 0], r[:, 1], r[:, 0] + r[:, 2], r[:, 1] + r[:, 3]]
        return b, np.array(w).ravel(), np.zeros(len(b), int), ms


# ---------------------------------------------------------------- tracking ---
class LineCounter:
    """Greedy centroid tracker + virtual counting line (horizontal) with hysteresis.
    A crossing counts only when a track moves from clearly above the line
    (beyond a dead band) to clearly below it, or the reverse, so detector jitter
    near the line is not counted."""

    def __init__(self, line_frac, max_jump=0.08, max_age=8, band=0.04, min_hits=3):
        self.line_frac, self.max_jump, self.max_age = line_frac, max_jump, max_age
        self.band, self.min_hits = band, min_hits
        self.tracks, self.next_id = {}, 1        # id -> dict(x, y, age, side, hits)
        self.down = self.up = 0

    def _side(self, y, ly, m):
        return -1 if y < ly - m else (1 if y > ly + m else 0)

    def update(self, boxes, shape):
        if self.line_frac <= 0:
            return
        h, w = shape[:2]
        ly, m, lim = self.line_frac * h, self.band * h, self.max_jump * max(h, w)
        cents = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
        pairs = sorted((np.hypot(cx - t["x"], cy - t["y"]), tid, j)
                       for tid, t in self.tracks.items() for j, (cx, cy) in enumerate(cents))
        used_t, used_c = set(), set()
        for d, tid, j in pairs:                     # globally nearest pairs first
            if d > lim or tid in used_t or j in used_c:
                continue
            used_t.add(tid); used_c.add(j)
            t = self.tracks[tid]
            t["x"], t["y"], t["age"], t["hits"] = cents[j][0], cents[j][1], 0, t["hits"] + 1
            s = self._side(t["y"], ly, m)
            if s != 0:
                if t["side"] == -1 and s == 1 and t["hits"] >= self.min_hits:
                    self.down += 1
                elif t["side"] == 1 and s == -1 and t["hits"] >= self.min_hits:
                    self.up += 1
                t["side"] = s
        for tid in list(self.tracks):
            if tid not in used_t:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    del self.tracks[tid]
        for j, (cx, cy) in enumerate(cents):
            if j not in used_c:
                self.tracks[self.next_id] = {"x": cx, "y": cy, "age": 0, "hits": 1,
                                             "side": self._side(cy, ly, m)}
                self.next_id += 1

# --------------------------------------------------------- accurate tier ----
class HybridAcc:
    """Accurate tier: perspective-aware hybrid (head640 near field + DM-Count far field)."""

    def __init__(self, head_model, dm_model, frame_w, frame_h):
        from nerisal import Detector, HybridCounter
        det = Detector(head_model, 640, 0.10, "head")
        det.run(np.zeros((480, 640, 3), np.uint8))
        grid = 2 if max(frame_w, frame_h) <= 1400 else 3        # tiles ~640 px: no upscaling artefacts
        self.h = HybridCounter(det, dm_model, det_grid=grid)
        self.h.count(np.zeros((frame_h, frame_w, 3), np.uint8))
        self.label = f"hybrid: {os.path.basename(head_model)} {grid}x{grid} + {os.path.basename(dm_model)}"

    def __call__(self, img):
        total, b, dmap, info, ms = self.h.count(img)
        return {"count": float(total), "boxes": b.numpy() if len(b) else np.zeros((0, 4)),
                "dmap": dmap, "info": info, "ms": ms}


class TiledAcc:
    """Accurate tier without DM-Count: tiled head detector only."""

    def __init__(self, model, imgsz, conf, target, grid):
        self.d = ETDetector(model, imgsz, conf, target, True, grid)
        self.label = self.d.label

    def __call__(self, img):
        b, s, c, ms = self.d(img)
        return {"count": float(len(b)), "boxes": b, "dmap": None,
                "info": {"regime": "detector", "dm_rects": [], "head_h": None}, "ms": ms}


class MockAcc:
    """Laptop test of the accurate tier: HOG boxes, fake density in the top third."""

    def __init__(self):
        self.hog = HOGDetector(1.0, "mock HOG (accurate)")
        self.label = "mock hybrid (HOG + fake density)"

    def __call__(self, img):
        b, s, c, ms = self.hog(img)
        h, w = img.shape[:2]
        dmap = np.zeros((h, w), np.float32)
        for x1, y1, x2, y2 in b:
            dmap[int(min(h - 1, (y1 + y2) / 2)), int(min(w - 1, (x1 + x2) / 2))] += 1
        far = 0.0
        if texture(img) > 0.12:                                 # pretend the far third is a packed crowd
            dmap[: h // 3] += 6.0 / max(1, (h // 3) * w)
            far = 6.0
        rects = [(0.0, 0.0, 1.0, 1 / 3)] if far else []
        return {"count": float(len(b)) + far, "boxes": b, "dmap": dmap,
                "info": {"regime": "hybrid" if far else "detector", "dm_rects": rects, "head_h": None}, "ms": ms}


# ------------------------------------------- accurate tier in its own process ---
def _block_map(d, f=8):
    """Sum a full-frame density map into f x f blocks (small enough to send between processes)."""
    h, w = d.shape
    H, W = h // f, w // f
    return d[:H * f, :W * f].reshape(H, f, W, f).sum(axis=(1, 3)).astype(np.float32)


def _acc_worker(conn, kind, kw):
    """Child process: owns the accurate models. Python runs model code one thread at a time,
    so a separate process is what keeps the live view smooth while this works."""
    try:
        acc = {"hybrid": HybridAcc, "tiled": TiledAcc, "mock": MockAcc}[kind](**kw)
        conn.send(("ready", acc.label))
    except Exception as e:
        conn.send(("error", repr(e)))
        return
    while True:
        frame = conn.recv()
        if frame is None:
            return
        try:
            r = acc(frame)
            r["blk"] = None if r["dmap"] is None else _block_map(r["dmap"])
            r["dmap"] = None
            r["info"] = {k: v for k, v in r["info"].items() if k != "head_h"}   # functions can't be sent
            r["boxes"] = np.asarray(r["boxes"], np.float32).reshape(-1, 4)
            conn.send(r)
        except Exception as e:
            conn.send({"error": repr(e)})


class AccProcess:
    def __init__(self, kind, **kw):
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_acc_worker, args=(child, kind, kw), daemon=True)
        self.proc.start()
        status, self.label = self.conn.recv()
        if status != "ready":
            raise SystemExit(f"accurate tier failed to start: {self.label}")

    def __call__(self, frame):
        self.conn.send(frame)
        r = self.conn.recv()                    # waiting here does not block the live view
        if "error" in r:
            raise RuntimeError(r["error"])
        return r


# ------------------------------------------------------------ floor model ---
class Floor:
    """Maps image points to floor metres with a homography from 4 clicked corners
    (near left, near right, far right, far left) of a measured rectangle."""

    def __init__(self, path):
        self.path, self.c, self.H = path, None, None
        try:
            with open(path) as f:
                self.set(json.load(f), save=False)
        except (OSError, ValueError, KeyError, AssertionError):
            pass

    def set(self, c, save=True):
        pts, size = c["points"], c["size_m"]
        assert len(pts) == 4 and all(0 <= u <= 1 and 0 <= v <= 1 for u, v in pts)
        assert len(size) == 2 and size[0] > 0 and size[1] > 0
        self.c = {"calibrated": True, "points": [[float(u), float(v)] for u, v in pts],
                  "size_m": [float(size[0]), float(size[1])], "time": c.get("time") or time.strftime("%H:%M")}
        self.H = None                                          # rebuilt for the frame size on first use
        if save:
            with open(self.path, "w") as f:
                json.dump(self.c, f)

    @property
    def ok(self):
        return self.c is not None

    def matrix(self, w, h):
        if self.H is None or self.H[0] != (w, h):
            src = np.float32([[u * w, v * h] for u, v in self.c["points"]])
            W, D = self.c["size_m"]
            dst = np.float32([[0, 0], [W, 0], [W, D], [0, D]])   # near edge at y=0, far edge at y=D
            M = cv2.getPerspectiveTransform(src, dst)
            self.H = ((w, h), M, np.linalg.inv(M))
        return self.H[1], self.H[2]


def cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return round(int(f.read()) / 1000, 1)
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------------ engine ---
class Engine:
    HEAT_W, HEAT_H = 32, 18

    def __init__(self, a, src, fast, acc):
        self.a, self.src, self.fast, self.acc = a, src, fast, acc
        self.lock = threading.Lock()
        self.jpg, self.jseq = b"", 0
        self.counter = LineCounter(a.line)
        self.hist = collections.deque(maxlen=1200)      # (t, count, density)
        self.events = collections.deque(maxlen=40)
        self.ema = None
        self.acc_res, self.acc_t, self.acc_ms = None, 0.0, 0.0
        self.need_acc = threading.Event()
        self.loop_t = collections.deque(maxlen=30)
        self.state = {}
        self.prev_grade = self.prev_peak = None
        self.prev_sat, self.prev_by, self.by_t = False, None, 0.0
        self.t0 = time.time()
        self.last_seq = -1
        self.floor = Floor(a.calib)
        self.temp, self.temp_t = None, 0.0
        os.makedirs(a.logdir, exist_ok=True)
        self.logf = open(os.path.join(a.logdir, time.strftime("session_%Y%m%d_%H%M%S.csv")), "w", newline="")
        self.log = csv.writer(self.logf)
        self.log.writerow(["time", "count", "avg_density", "peak_density", "grade", "peak_grade", "saturated",
                           "counted_by", "fast_ms", "acc_ms", "crossed_down", "crossed_up", "ttc_s", "cpu_temp"])
        self.last_log = 0.0

    def event(self, msg, level="info"):
        self.events.appendleft({"t": time.strftime("%H:%M:%S"), "msg": msg, "level": level})

    # ----------------------------------------------------- accurate worker --
    def acc_loop(self):
        """Runs the accurate tier in the background so the video never freezes."""
        a = self.a
        last_scene, hot = -1, False
        while True:
            idle = time.time() - self.acc_t
            new_scene = self.src.scene != last_scene
            t = self.temp or 0
            if t >= a.hot_temp and not hot:
                hot = True
                self.event(f"Pi at {t:.0f} °C: accurate count slowed down to cool it", "warn")
            elif hot and t < a.hot_temp - 5:
                hot = False
                self.event(f"Pi cooled to {t:.0f} °C: normal accurate count rate")
            gap = a.esc_every * (3 if hot else 1)
            if not new_scene and (not (self.need_acc.is_set() or idle >= a.idle_every) or idle < gap):
                time.sleep(0.05)
                continue
            frame, _ = self.src.latest()
            if frame is None:
                time.sleep(0.1)
                continue
            frame, scene = frame.copy(), self.src.scene
            last_scene = scene
            try:
                r = self.acc(frame)
            except Exception as e:                          # keep the demo alive
                self.event(f"accurate tier error: {e}", "alert")
                print("accurate tier error:", e)
                time.sleep(1)
                continue
            r["t"], r["shape"], r["scene"] = time.time(), frame.shape[:2], scene
            with self.lock:
                self.acc_res, self.acc_t, self.acc_ms = r, r["t"], r["ms"]
            self.need_acc.clear()

    # ------------------------------------------------------- crowd points --
    def people_points(self, boxes, shape, res, use_acc):
        """(x, y, weight) image points standing for people, plus a head-height function."""
        h, w = shape
        if use_acc and res.get("blk") is not None:
            blk, f = res["blk"], 8                          # 8x8 px blocks -> weighted points
            ys, xs = np.nonzero(blk > 1e-3)
            pts = np.c_[(xs + 0.5) * f, (ys + 0.5) * f, blk[ys, xs]]
            hh = None                                       # refitted from the boxes below
            src_boxes = res["boxes"]
        else:
            b = np.asarray(boxes, float).reshape(-1, 4)
            pts = np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, np.ones(len(b))]
            hh, src_boxes = None, b
        if self.a.target != "head":                     # body boxes: feet = box bottom
            bb = np.asarray(src_boxes, float).reshape(-1, 4)
            half = float(np.median(bb[:, 3] - bb[:, 1])) / 2 if len(bb) else 0.0
            return pts, (lambda y: half)
        if hh is None:
            from_boxes = np.asarray(src_boxes, float).reshape(-1, 4)
            try:
                from nerisal import HybridCounter
                hh = HybridCounter.perspective(from_boxes)
            except Exception:                               # mock mode: no ExecuTorch import
                med = float(np.median(from_boxes[:, 3] - from_boxes[:, 1])) if len(from_boxes) else 0.0
                hh = lambda y: med
        return pts, hh

    def spatial(self, pts, hh, shape, total):
        """Average and peak density, hotspot and the heat grid for the dashboard."""
        a, (h, w) = self.a, shape
        out = {"avg": None, "peak": None, "hotspot": None, "heat": None}
        if len(pts) and pts[:, 2].sum() > 0:
            pts = pts.copy()
            pts[:, 2] *= total / pts[:, 2].sum()            # weights add up to the fused count
        if self.floor.ok:
            M, Minv = self.floor.matrix(w, h)
            Wm, Dm = self.floor.c["size_m"]
            if len(pts):
                k_foot = a.foot if a.target == "head" else 1.0
                foot = pts[:, 1] + k_foot * np.array([hh(y) for y in pts[:, 1]])   # head -> feet on the floor
                fl = cv2.perspectiveTransform(np.float32(np.c_[pts[:, 0], foot])[None], M)[0]
                inside = (fl[:, 0] >= 0) & (fl[:, 0] <= Wm) & (fl[:, 1] >= 0) & (fl[:, 1] <= Dm)
                fl, wt = fl[inside], pts[inside, 2]
            else:
                fl, wt = np.zeros((0, 2)), np.zeros(0)
            out["avg"] = float(wt.sum()) / (Wm * Dm)
            cs = 0.25                                       # floor grid, 25 cm cells
            gx, gy = max(1, int(np.ceil(Wm / cs))), max(1, int(np.ceil(Dm / cs)))
            grid = np.zeros((gy, gx), np.float32)
            if len(fl):
                ix = np.clip((fl[:, 0] / cs).astype(int), 0, gx - 1)
                iy = np.clip((fl[:, 1] / cs).astype(int), 0, gy - 1)
                np.add.at(grid, (iy, ix), wt)
            k = max(1, int(round(a.peak_win / cs)))
            local = cv2.boxFilter(grid, -1, (k, k), normalize=False, borderType=cv2.BORDER_CONSTANT)
            local /= (k * cs) ** 2                          # people per m^2 in a peak_win x peak_win window
            if a.peak_win * a.peak_win > Wm * Dm:           # zone smaller than the window: peak = average
                local[:] = out["avg"]
            j, i = np.unravel_index(int(np.argmax(local)), local.shape)
            out["peak"] = max(float(local[j, i]), out["avg"])
            hp = cv2.perspectiveTransform(np.float32([[[(i + 0.5) * cs, (j + 0.5) * cs]]]), Minv)[0, 0]
            out["hotspot"] = {"x": round(float(np.clip(hp[0] / w, 0, 1)), 3),
                              "y": round(float(np.clip(hp[1] / h, 0, 1)), 3), "density": round(out["peak"], 2)}
            # heat grid over the image: each cell shows the local density of the floor under it
            cx = (np.arange(self.HEAT_W) + 0.5) * w / self.HEAT_W
            cy = (np.arange(self.HEAT_H) + 0.5) * h / self.HEAT_H
            X, Y = np.meshgrid(cx, cy)
            fc = cv2.perspectiveTransform(np.float32(np.c_[X.ravel(), Y.ravel()])[None], M)[0]
            ok = (fc[:, 0] >= 0) & (fc[:, 0] <= Wm) & (fc[:, 1] >= 0) & (fc[:, 1] <= Dm)
            vals = np.zeros(len(fc), np.float32)
            ii = np.clip((fc[ok, 0] / cs).astype(int), 0, gx - 1)
            jj = np.clip((fc[ok, 1] / cs).astype(int), 0, gy - 1)
            vals[ok] = local[jj, ii]
            out["heat"] = vals
        elif a.area:
            out["avg"] = total / a.area     # average only: a peak needs real floor geometry (calibration)
        if out["heat"] is not None:
            v = np.round(out["heat"], 3)
            out["heat"] = {"w": self.HEAT_W, "h": self.HEAT_H, "max": round(float(v.max()), 2),
                           "cells": v.tolist()}
        return out

    def ttc(self, now, d):
        """Seconds until average density reaches Grade E at the current trend, or None."""
        pts = [(t, dd) for t, _, dd in self.hist if now - t <= self.a.trend]
        if len(pts) < 8 or pts[-1][0] - pts[0][0] < 5 or d >= E_LIMIT:
            return None
        t = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
        slope = np.polyfit(t - t[0], y, 1)[0]
        if slope <= 1e-4:
            return None
        s = (E_LIMIT - d) / slope
        return s if s < 1800 else None

    # ---------------------------------------------------------------- step --
    def step(self):
        frame, seq = self.src.latest()
        if frame is None or seq == self.last_seq:     # wait for a new frame
            time.sleep(0.005)
            return
        self.last_seq = seq
        a, now = self.a, time.time()
        h, w = frame.shape[:2]
        boxes, scores, cls, fms = self.fast(frame)
        self.counter.update(boxes, frame.shape)
        tex = texture(frame)
        n_fast = len(boxes)
        busy = tex >= a.tex_thr
        dens_fast = n_fast / a.area if a.area else 0.0
        if self.acc and (busy or n_fast >= a.esc_count or dens_fast >= a.esc_density):
            self.need_acc.set()

        with self.lock:
            res = self.acc_res
        fresh = (res is not None and now - res["t"] < a.hold_acc and res["shape"] == (h, w)
                 and res["scene"] == self.src.scene)
        use_acc = fresh and res["count"] >= n_fast
        if use_acc:
            n, reg = res["count"], res["info"].get("regime", "detector")
            by = {"detector": "head640 tiled", "hybrid": "heads near, DM-Count far",
                  "density": "DM-Count density"}.get(reg, self.acc.label)
        else:
            n, by = float(n_fast), self.fast.label

        if self.src.scene != getattr(self, "ema_scene", None):
            self.ema, self.ema_scene = None, self.src.scene
        self.ema = n if self.ema is None else self.ema + a.smooth * (n - self.ema)
        cnt = int(round(self.ema))
        sp = self.spatial(*self.people_points(boxes, (h, w), res if use_acc else None, use_acc), (h, w), self.ema)
        avg, peak = sp["avg"], sp["peak"]
        sat = a.target in ("person", "head") and busy and cnt < a.sat_count
        if avg is not None:
            grade, action = fruin(avg)
            peak_grade = fruin(peak)[0] if peak is not None else "-"
        else:
            grade, action, peak_grade = "-", "Calibrate the floor (C) or set --area to enable grades", "-"
        if sat:
            grade = "E*" if grade in ("-", "A", "B", "C", "D") else grade
            action = "Too dense to count. Treat as grade E or worse."
            by = "saturation guard"
        elif peak_grade in ("E", "F") and grade not in ("E", "F"):
            action = f"{action}. Hotspot at grade {peak_grade}: thin it out locally."
        regime = ("dense" if sat or (use_acc and res["info"].get("regime") in ("hybrid", "density"))
                  else "medium" if use_acc else "light")
        density = avg or 0.0
        self.hist.append((now, cnt, density))
        ttc = (self.ttc(now, density) if avg is not None and not sat and self.src.kind != "images"
               else None)

        if self.prev_grade is not None and grade != self.prev_grade:
            lvl = "alert" if grade in ("E", "E*", "F") else ("warn" if grade in ("C", "D") else "info")
            self.event(f"Grade changed from {self.prev_grade} to {grade}", lvl)
        if self.prev_peak not in (None, "E", "F") and peak_grade in ("E", "F"):
            self.event(f"Hotspot reached grade {peak_grade} ({peak:.1f} people per m²)", "alert")
        if sat and not self.prev_sat:
            self.event("Saturation guard: crowd too dense to count", "alert")
        if self.prev_by is not None and by != self.prev_by and not sat and now - self.by_t > 20:
            self.event(f"Now counted by {by}")
            self.by_t = now
        self.prev_grade, self.prev_peak, self.prev_sat, self.prev_by = grade, peak_grade, sat, by

        img = self.annotate(frame.copy(), boxes, cls, res if fresh else None)
        if max(h, w) > a.stream_w:                       # lighter stream for the browser
            s = a.stream_w / max(h, w)
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])

        if now - self.temp_t > 2:
            self.temp, self.temp_t = cpu_temp(), now
        self.loop_t.append(now)
        fps = (len(self.loop_t) - 1) / (self.loop_t[-1] - self.loop_t[0]) if len(self.loop_t) > 1 else 0.0
        st = {
            "count": cnt, "raw": round(n, 1), "saturated": sat,
            "density": round(density, 3), "avg_density": None if avg is None else round(avg, 3),
            "peak_density": None if peak is None else round(peak, 2),
            "grade": grade, "peak_grade": peak_grade, "action": action,
            "mode": "escalated" if use_acc else "watch", "regime": regime, "counted_by": by,
            "fast_label": self.fast.label, "acc_label": self.acc.label if self.acc else "off",
            "fps": round(fps, 1), "fast_ms": round(fms, 1), "acc_ms": round(self.acc_ms, 0),
            "acc_age": None if res is None else round(now - res["t"], 1),
            "ttc": None if ttc is None else int(ttc), "down": self.counter.down, "up": self.counter.up,
            "line": a.line > 0, "area": (self.floor.c["size_m"][0] * self.floor.c["size_m"][1]
                                         if self.floor.ok else a.area),
            "calibrated": self.floor.ok, "calib_time": self.floor.c["time"] if self.floor.ok else None,
            "hotspot": sp["hotspot"], "heat": sp["heat"], "texture": round(tex, 3), "cpu_temp": self.temp,
            "source": self.src.desc, "uptime": int(now - self.t0),
            "hist": [[round(t - now, 1), c, round(d, 3)] for t, c, d in self.hist if now - t <= 240][::2],
            "events": list(self.events)[:15],
        }
        with self.lock:
            if ok:
                self.jpg, self.jseq = jpg.tobytes(), self.jseq + 1
            self.state = st
        if now - self.last_log >= 1.0:
            self.last_log = now
            self.log.writerow([time.strftime("%H:%M:%S"), cnt, f"{density:.3f}",
                               "" if peak is None else f"{peak:.2f}", grade, peak_grade, int(sat), by,
                               f"{fms:.1f}", f"{self.acc_ms:.0f}", self.counter.down, self.counter.up,
                               "" if ttc is None else int(ttc), "" if self.temp is None else self.temp])
            self.logf.flush()

    def annotate(self, img, boxes, cls, res):
        """Privacy first: blur every detected head and pixelate density-counted regions.
        No text is burned in; the dashboard shows the numbers."""
        h, w = img.shape[:2]
        t = max(2, int(round(max(h, w) / 640)))           # box line width in pixels
        if not self.a.no_blur and res is not None:
            for x0, y0, x1, y1 in res["info"].get("dm_rects", []):
                X0, Y0, X1, Y1 = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)
                roi = img[Y0:Y1, X0:X1]
                if roi.size:
                    sm = cv2.resize(roi, (max(1, (X1 - X0) // 12), max(1, (Y1 - Y0) // 12)),
                                    interpolation=cv2.INTER_AREA)
                    img[Y0:Y1, X0:X1] = cv2.resize(sm, (X1 - X0, Y1 - Y0), interpolation=cv2.INTER_NEAREST)
        for b in boxes:
            x1, y1, x2, y2 = [int(v) for v in b]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            if not self.a.no_blur:
                hy = y2 if self.a.target == "head" else y1 + max(1, (y2 - y1) // 5)
                roi = img[y1:hy, x1:x2]
                if roi.size:
                    img[y1:hy, x1:x2] = cv2.GaussianBlur(roi, (31, 31), 0)
            # dark outline under a bright green line: visible on light and dark backgrounds,
            # thickness scaled so it survives the downscaled browser stream
            cv2.rectangle(img, (x1, y1), (x2, y2), (20, 20, 20), t + 2, cv2.LINE_AA)
            cv2.rectangle(img, (x1, y1), (x2, y2), (60, 255, 60), t, cv2.LINE_AA)
        if self.a.line > 0:
            ly = int(self.a.line * h)
            cv2.line(img, (0, ly), (w, ly), (230, 230, 230), 1, cv2.LINE_AA)
        return img

    def run(self):
        while True:
            try:
                self.step()
            except Exception as e:                        # keep the demo alive
                self.event(f"error: {e}", "alert")
                print("engine error:", e)
                time.sleep(0.5)

    def snapshot(self):
        with self.lock:
            jpg, st = self.jpg, dict(self.state)
        os.makedirs(self.a.snapdir, exist_ok=True)
        name = time.strftime("snap_%H%M%S.jpg")
        with open(os.path.join(self.a.snapdir, name), "wb") as f:
            f.write(jpg)
        self.event(f"Snapshot saved: {name} ({st.get('count')} people, grade {st.get('grade')})")
        return name


# --------------------------------------------------------------- web server --
HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "dashboard.html")


def make_handler(engine):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, "application/json", json.dumps(obj).encode())

        def do_GET(self):
            p = self.path.split("?")[0]
            if p == "/":
                with open(PAGE, "rb") as f:
                    self._send(200, "text/html; charset=utf-8", f.read())
            elif p == "/stats":
                with engine.lock:
                    body = json.dumps(engine.state).encode()
                self._send(200, "application/json", body)
            elif p == "/snapshot":
                self._json({"saved": engine.snapshot()})
            elif p == "/calibrate":
                self._json(engine.floor.c or {"calibrated": False})
            elif p == "/proof.json":
                try:
                    with open(engine.a.proof, "rb") as f:
                        self._send(200, "application/json", f.read())
                except OSError:
                    self._send(404, "application/json", b"{}")
            elif p == "/frame.jpg":
                with engine.lock:
                    jpg = engine.jpg
                self._send(200, "image/jpeg", jpg)
            elif p == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                last = -1
                try:
                    while True:
                        with engine.lock:
                            jpg, seq = engine.jpg, engine.jseq
                        if seq != last and jpg:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                            last = seq
                        time.sleep(0.03)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
            else:
                self._send(404, "text/plain", b"not found")

        def do_POST(self):
            if self.path.split("?")[0] != "/calibrate":
                return self._send(404, "text/plain", b"not found")
            try:
                n = int(self.headers.get("Content-Length", 0))
                engine.floor.set(json.loads(self.rfile.read(n) or b"{}"))
            except Exception:
                return self._json({"ok": False, "error": "Send 4 points between 0 and 1 and a positive size."}, 400)
            W, D = engine.floor.c["size_m"]
            engine.event(f"Floor calibrated: {W:g} × {D:g} m")
            self._json({"ok": True})
    return H


def local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def main():
    ap = argparse.ArgumentParser(description="NERISAL live crowd-pressure monitor")
    ap.add_argument("--source", default="0", help="rpicam | picam | camera index (0) | video file | URL | image folder")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--hold", type=float, default=4.0, help="seconds per image in folder mode")
    ap.add_argument("--target", choices=["person", "head", "vehicle"], default="head")
    ap.add_argument("--fast-model", default="head320.pte", help="watch tier, every frame")
    ap.add_argument("--fast-imgsz", type=int, default=320)
    ap.add_argument("--acc-model", default="head640.pte", help="accurate tier detector ('' = off)")
    ap.add_argument("--acc-imgsz", type=int, default=640)
    ap.add_argument("--dm", default="dmcount_qnrf_432x240.pte", help="DM-Count for the hybrid ('' = heads only)")
    ap.add_argument("--grid", type=int, default=2, help="tiles per side when --dm is off")
    ap.add_argument("--conf", type=float, default=0.15, help="watch tier threshold (0.15 for head models)")
    ap.add_argument("--area", type=float, default=0, help="visible floor m^2 if the floor is not calibrated")
    ap.add_argument("--calib", default="calib.json", help="floor calibration file (written from the dashboard)")
    ap.add_argument("--foot", type=float, default=5.0, help="feet are this many head-heights below the head")
    ap.add_argument("--peak-win", type=float, default=2.0, help="peak density window side in metres")
    ap.add_argument("--line", type=float, default=0.0, help="counting line at this fraction of height, 0=off")
    ap.add_argument("--esc-count", type=int, default=8, help="escalate when the watch tier sees this many")
    ap.add_argument("--esc-density", type=float, default=0.43, help="...or this density (p/m^2)")
    ap.add_argument("--esc-every", type=float, default=4.0,
                    help="rest (s) after each accurate pass, so the live view stays smooth")
    ap.add_argument("--hot-temp", type=float, default=76, help="above this CPU temp (C) rest 3x longer")
    ap.add_argument("--idle-every", type=float, default=15.0, help="accurate pass at least this often anyway")
    ap.add_argument("--hold-acc", type=float, default=10.0, help="seconds an accurate result stays valid")
    ap.add_argument("--smooth", type=float, default=0.35, help="count smoothing (0..1, 1=no smoothing)")
    ap.add_argument("--tex-thr", type=float, default=0.22)
    ap.add_argument("--sat-count", type=int, default=15)
    ap.add_argument("--trend", type=float, default=60, help="seconds of history for time-to-critical")
    ap.add_argument("--stream-w", type=int, default=960, help="longest side of the browser stream")
    ap.add_argument("--no-blur", action="store_true")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--proof", default=os.path.join(HERE, "proof.json"), help="results file for the Results drawer")
    ap.add_argument("--logdir", default="live_logs")
    ap.add_argument("--snapdir", default="live_snaps")
    ap.add_argument("--fp32", action="store_true", help="use FP32 head models even if INT8 ones exist")
    ap.add_argument("--mock", action="store_true", help="test without ExecuTorch (OpenCV HOG)")
    a = ap.parse_args()

    for k in ("fast_model", "acc_model"):              # prefer the INT8 build when it is here
        m = getattr(a, k)
        q = m.replace(".pte", "_int8.pte")
        if m and not m.endswith("_int8.pte") and os.path.exists(q) and not a.fp32:
            setattr(a, k, q)
    src = Source(a.source, a.width, a.height, a.hold)
    if a.mock:
        fast = HOGDetector(0.6, "mock HOG (fast)")
        acc = AccProcess("mock") if a.acc_model else None
    else:
        fast = ETDetector(a.fast_model, a.fast_imgsz, a.conf, a.target, False, 1)
        acc = None
        if a.acc_model and a.dm and os.path.exists(a.dm) and a.target == "head":
            for _ in range(100):                          # frame size decides the tile grid
                f, _s = src.latest()
                if f is not None:
                    break
                time.sleep(0.05)
            fh, fw = f.shape[:2] if f is not None else (a.height, a.width)
            acc = AccProcess("hybrid", head_model=a.acc_model, dm_model=a.dm, frame_w=fw, frame_h=fh)
        elif a.acc_model:
            acc = AccProcess("tiled", model=a.acc_model, imgsz=a.acc_imgsz,
                             conf=0.25 if a.target == "head" else a.conf, target=a.target, grid=a.grid)
    eng = Engine(a, src, fast, acc)
    eng.event(f"Started: {src.desc}")
    eng.event(f"Watch: {fast.label}. Accurate: {acc.label if acc else 'off'}")
    if not eng.floor.ok:
        eng.event("Floor not calibrated: press C on the dashboard", "warn")
    threading.Thread(target=eng.run, daemon=True).start()
    if acc:
        threading.Thread(target=eng.acc_loop, daemon=True).start()

    srv = ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(eng))
    print(f"\n  NERISAL live is running.  Open  http://{local_ip()}:{a.port}  on the laptop.\n"
          f"  Logs: {a.logdir}/   Snapshots: {a.snapdir}/   Ctrl+C to stop.\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("stopping")
    finally:
        src.alive = False
        eng.logf.close()


if __name__ == "__main__":
    main()
