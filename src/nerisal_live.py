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
import argparse, collections, csv, glob, itertools, json, multiprocessing as mp, os, socket, threading, time
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
SCENES = itertools.count(1)        # unique scene ids across all sources: an accurate result never leaks
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
class Source:
    """Background grabber that always holds the newest frame."""

    def __init__(self, spec, width, height, hold):
        self.spec, self.w, self.h, self.hold = spec, width, height, hold
        self.frame, self.seq, self.lock, self.alive = None, 0, threading.Lock(), True
        self.scene = next(SCENES)              # changes only when the slideshow moves to a new image
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
        elif os.path.isfile(s) and s.lower().endswith(IMG_EXT):
            self.files, self.hold = [s], 10 ** 9            # one photo: hold it until the source changes
            self.kind, self.desc = "images", f"photo {os.path.basename(s)}"
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
                f = read_image(self.files[i % len(self.files)])
                with self.lock:
                    self.scene = next(SCENES)
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

    def stop(self):
        self.alive = False
        for attr in ("cap", "cam"):
            try:
                getattr(self, attr).release() if attr == "cap" else getattr(self, attr).stop()
            except Exception:
                pass
        if getattr(self, "proc", None):
            self.proc.terminate()


def read_image(path):
    """OpenCV first, Pillow for formats OpenCV can't read (AVIF, some WebP)."""
    img = cv2.imread(path)
    if img is None:
        try:
            from PIL import Image
            with Image.open(path) as im:
                img = cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception:
            return None
    return img


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


class AutoFloor:
    """Self-calibration from the people in view: no clicks, no tape.

    In an angled camera view, the pixel height of a head grows in a straight line with image row
    (heads near the horizon are tiny, heads at the bottom are large). Fitting that line gives the
    horizon row and, because an adult head is about --head-m tall, the camera's height above the
    plane of the heads. With an assumed lens field of view, that fixes a homography from image to
    that plane, so people per square metre can be measured on it. Refitted continuously from the
    last --auto-window seconds of detections, so it adapts if the camera moves."""

    def __init__(self, head_m, fov_deg, window_s):
        self.head_m, self.fov, self.window = head_m, np.radians(fov_deg), window_s
        self.samples = collections.deque(maxlen=4000)        # (t, row, box height)
        self.model, self.fit_t, self.why = None, 0.0, "waiting for people in view"

    def add(self, boxes, now):
        b = np.asarray(boxes, float).reshape(-1, 4)
        for x1, y1, x2, y2 in b:
            if y2 - y1 >= 4:
                self.samples.append((now, (y1 + y2) / 2, y2 - y1))

    def fit(self, w, h, now):
        if now - self.fit_t < 1.0 and self.model and self.model["shape"] == (w, h):
            return self.model
        self.fit_t = now
        prev, self.model = self.model, None
        m = self._fit(w, h, now)
        if m is None:                                   # keep a recent good fit rather than flicker
            if prev and prev["shape"] == (w, h) and now - prev["t"] <= 2 * self.window:
                self.model = prev
            return self.model
        if prev and prev["shape"] == (w, h):            # smooth the camera estimate over time
            for k in ("y_h", "Hp"):
                m[k] = prev[k] + 0.2 * (m[k] - prev[k])
            m["th"] = np.arctan2(h / 2 - m["y_h"], m["f"])
        self.model = m
        return m

    def _fit(self, w, h, now):
        S = np.array([s for s in self.samples if now - s[0] <= self.window])
        if len(S) < 20:
            self.why = f"needs more people in view ({len(S)} of 20 head samples)"
            return None
        y, hh = S[:, 1], S[:, 2]
        if np.percentile(y, 90) - np.percentile(y, 10) < 0.12 * h:
            self.why = "needs people at different distances (mount the camera higher, angled down)"
            return None
        A = np.vstack([np.ones_like(y), y]).T
        co = np.linalg.lstsq(A, hh, rcond=None)[0]
        r = np.abs(A @ co - hh)
        k = r <= np.percentile(r, 80)
        a0, b = np.linalg.lstsq(A[k], hh[k], rcond=None)[0]
        if b <= 1e-4:
            self.why = "head size does not grow towards the bottom: camera too level"
            return None
        y_h = -a0 / b                                        # horizon row: head size would be 0 there
        Hp = self.head_m / b                                 # camera height above the plane of the heads
        if not (0.2 <= Hp <= 30) or y_h > np.percentile(y, 5):
            self.why = "head sizes inconsistent; check the camera is steady and angled down"
            return None
        f = (w / 2) / np.tan(self.fov / 2)
        th = np.arctan2(h / 2 - y_h, f)                      # camera tilt below horizontal
        self.why = ""
        return {"shape": (w, h), "y_h": y_h, "Hp": Hp, "f": f, "th": th, "n": len(S), "t": now}

    @staticmethod
    def to_plane(m, x, y):
        w, h = m["shape"]
        dy = np.asarray(y, float) - h / 2
        c, s_ = np.cos(m["th"]), np.sin(m["th"])
        dn = dy * c + m["f"] * s_
        ok = dn > 1e-3 * m["f"]
        t = np.where(ok, m["Hp"] / np.where(ok, dn, 1), np.nan)
        D = t * (m["f"] * c - dy * s_)
        X = t * (np.asarray(x, float) - w / 2)
        return X, D, ok

    @staticmethod
    def to_image(m, X, D):
        w, h = m["shape"]
        c, s_ = np.cos(m["th"]), np.sin(m["th"])
        zc = D * c + m["Hp"] * s_
        yc = -D * s_ + m["Hp"] * c
        return w / 2 + m["f"] * X / zc, h / 2 + m["f"] * yc / zc


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
        self.auto = AutoFloor(a.head_m, a.fov, a.auto_window) if a.auto_calib else None
        self.auto_sample_t, self.auto_res_t = 0.0, 0.0
        self.last_fast_n = 0
        self.live_src, self.upload = src, None
        self.incident, self.calm_since = None, None
        try:
            with open(os.path.join(a.incident_dir, "incidents.json")) as f:
                self.incidents = json.load(f)
        except (OSError, ValueError):
            self.incidents = []
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
            frame, scene, fast_n = frame.copy(), self.src.scene, self.last_fast_n
            last_scene = scene
            try:
                r = self.acc(frame)
            except Exception as e:                          # keep the demo alive
                self.event(f"accurate tier error: {e}", "alert")
                print("accurate tier error:", e)
                time.sleep(1)
                continue
            r["t"], r["shape"], r["scene"], r["fast_n"] = time.time(), frame.shape[:2], scene, fast_n
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
        elif self.auto is not None and self.auto.fit(w, h, time.time()) is not None:
            out.update(self.auto_spatial(pts, (h, w)))
            if out["avg"] is None and a.area:
                out["avg"] = total / a.area
        elif a.area:
            out["avg"] = total / a.area     # average only: a peak needs real floor geometry (calibration)
        if out["heat"] is not None:
            v = np.round(out["heat"], 3)
            out["heat"] = {"w": self.HEAT_W, "h": self.HEAT_H, "max": round(float(v.max()), 2),
                           "cells": v.tolist()}
        return out

    def auto_spatial(self, pts, shape):
        """Density on the plane of the heads, from the self-calibrated camera model."""
        a, (h, w), m, cs = self.a, shape, self.auto.model, 0.25
        out = {}
        if len(pts):
            X, D, ok = AutoFloor.to_plane(m, pts[:, 0], pts[:, 1])
            X, D, wt = X[ok], D[ok], pts[ok, 2]
        else:
            X = D = wt = np.zeros(0)
        # the floor we can see: from the bottom of the image to just beyond the farthest person
        _, Dn, _ = AutoFloor.to_plane(m, np.array([w / 2]), np.array([h - 1]))
        d_near = float(Dn[0])
        d_far = float(np.percentile(D, 98)) + 0.5 if len(D) else d_near + 6
        d_far = float(np.clip(d_far, d_near + 2, d_near + a.auto_depth))
        prev = getattr(self, "auto_dfar", None)                  # smooth the far edge too
        d_far = d_far if prev is None else prev + 0.015 * (d_far - prev)   # about 5 s at 15 fps
        self.auto_dfar = d_far
        half = (d_far * w / 2) / m["f"] * 1.2
        gx, gy = int(np.ceil(2 * half / cs)), int(np.ceil((d_far - d_near) / cs))
        gx, gy = min(gx, 400), min(gy, 400)
        Xc = -half + (np.arange(gx) + 0.5) * (2 * half / gx)
        Dc = d_near + (np.arange(gy) + 0.5) * ((d_far - d_near) / gy)
        XX, DD = np.meshgrid(Xc, Dc)
        ix_, iy_ = AutoFloor.to_image(m, XX, DD)
        mask = ((ix_ >= 0) & (ix_ < w) & (iy_ >= 0) & (iy_ < h)).astype(np.float32)   # floor in view
        cx_, cy_ = 2 * half / gx, (d_far - d_near) / gy
        area = float(mask.sum() * cx_ * cy_)
        grid = np.zeros((gy, gx), np.float32)
        if len(D):
            ix = ((X + half) / cx_).astype(int)
            iy = ((D - d_near) / cy_).astype(int)
            keep = (ix >= 0) & (ix < gx) & (iy >= 0) & (iy < gy)
            np.add.at(grid, (iy[keep], ix[keep]), wt[keep])
        if area < self.a.auto_min_area:                          # camera too close or too level to measure
            self.auto.why = (f"floor in view looks only {area:.1f} m²: mount the camera higher and angled down, "
                             "or start with --area")
            self.auto_bad = True
            return {"avg": None, "peak": None, "hotspot": None, "heat": None}
        self.auto_bad = False
        inside = float(grid[mask > 0].sum())
        out["avg"] = inside / max(area, 1e-6)
        kx = max(1, int(round(a.peak_win / cx_))); ky = max(1, int(round(a.peak_win / cy_)))
        mass = cv2.boxFilter(grid, -1, (kx, ky), normalize=False, borderType=cv2.BORDER_CONSTANT)
        cover = cv2.boxFilter(mask, -1, (kx, ky), normalize=False, borderType=cv2.BORDER_CONSTANT)
        full = kx * ky
        local = np.where(cover >= 0.5 * full, mass / np.maximum(cover, 1) / (cx_ * cy_), 0) * (mask > 0)
        j, i = np.unravel_index(int(np.argmax(local)), local.shape)
        out["peak"] = max(float(local[j, i]), out["avg"])
        hx, hy = AutoFloor.to_image(m, np.array([Xc[i]]), np.array([Dc[j]]))
        out["hotspot"] = {"x": round(float(np.clip(hx[0] / w, 0, 1)), 3),
                          "y": round(float(np.clip(hy[0] / h, 0, 1)), 3), "density": round(out["peak"], 2)}
        cxs = (np.arange(self.HEAT_W) + 0.5) * w / self.HEAT_W
        cys = (np.arange(self.HEAT_H) + 0.5) * h / self.HEAT_H
        PX, PY = np.meshgrid(cxs, cys)
        hX, hD, ok = AutoFloor.to_plane(m, PX.ravel(), PY.ravel())
        vals = np.zeros(PX.size, np.float32)
        ii = ((np.nan_to_num(hX) + half) / cx_).astype(int)
        jj = ((np.nan_to_num(hD) - d_near) / cy_).astype(int)
        good = ok & (ii >= 0) & (ii < gx) & (jj >= 0) & (jj < gy)
        vals[good] = local[jj[good], ii[good]]
        out["heat"] = vals
        self.auto_area = area
        return out

    ORDER = {"-": -1, "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}

    def steady(self, key, raw, now):
        """Grade with hysteresis: a worse grade must hold for --grade-up seconds, a better one for
        --grade-down seconds, so people walking past don't make the grade flicker."""
        st = self.__dict__.setdefault("_steady", {})
        cur = st.get(key)
        if cur is None or cur["scene"] != self.src.scene:
            st[key] = {"shown": raw, "cand": raw, "since": now, "scene": self.src.scene}
            return raw
        if raw == cur["shown"]:
            cur["cand"] = raw
            return raw
        if raw != cur["cand"]:
            cur["cand"], cur["since"] = raw, now
        wait = self.a.grade_up if self.ORDER[raw] > self.ORDER[cur["shown"]] else self.a.grade_down
        if now - cur["since"] >= wait:
            cur["shown"] = raw
        return cur["shown"]

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
        self.last_fast_n = n_fast
        busy = tex >= a.tex_thr
        dens_fast = n_fast / a.area if a.area else 0.0
        if self.acc and (busy or n_fast >= a.esc_count or dens_fast >= a.esc_density):
            self.need_acc.set()

        with self.lock:
            res = self.acc_res
        if self.auto is not None:
            if res is not None and res["t"] != self.auto_res_t and res.get("scene") == self.src.scene:
                self.auto.add(res["boxes"], now); self.auto_res_t = res["t"]
            elif now - self.auto_sample_t >= 0.5:
                self.auto.add(boxes, now); self.auto_sample_t = now
        fresh = (res is not None and now - res["t"] < a.hold_acc and res["shape"] == (h, w)
                 and res["scene"] == self.src.scene
                 and abs(n_fast - res.get("fast_n", n_fast)) <= max(2, 0.4 * res.get("fast_n", 0)))
        if res is not None and not fresh and now - res["t"] < a.hold_acc and self.acc:
            self.need_acc.set()                       # the view changed: get a new accurate count soon
        use_acc = fresh and res["count"] >= n_fast
        if use_acc:
            n, reg = res["count"], res["info"].get("regime", "detector")
            by = {"detector": "head640 tiled", "hybrid": "heads near, DM-Count far",
                  "density": "DM-Count density"}.get(reg, self.acc.label)
        else:
            n, by = float(n_fast), self.fast.label

        if self.src.scene != getattr(self, "ema_scene", None):
            self.ema, self.ema_scene = None, self.src.scene
        # time-based smoothing: about a.smooth_s seconds to follow a change, whatever the frame rate
        dt = now - getattr(self, "ema_t", now)
        self.ema_t = now
        k = 1.0 if self.ema is None else 1 - np.exp(-dt / max(a.smooth_s, 0.01))
        self.ema = n if self.ema is None else self.ema + k * (n - self.ema)
        cnt = int(round(self.ema))
        sp = self.spatial(*self.people_points(boxes, (h, w), res if use_acc else None, use_acc), (h, w), self.ema)
        avg, peak = sp["avg"], sp["peak"]
        auto_ok = ((not self.floor.ok) and self.auto is not None and self.auto.model is not None
                   and not getattr(self, "auto_bad", False))
        sat = a.target in ("person", "head") and busy and cnt < a.sat_count
        if avg is not None:
            grade = self.steady("avg", fruin(avg)[0], now)
            action = next(act for _, g, act in FRUIN if g == grade)
            peak_grade = self.steady("peak", fruin(peak)[0], now) if peak is not None else "-"
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

        if ok:
            self.track_incident(now, grade, peak_grade, sat, cnt, avg, peak, by, jpg)
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
            "line": a.line > 0, "area": (self.floor.c["size_m"][0] * self.floor.c["size_m"][1] if self.floor.ok
                                         else round(getattr(self, "auto_area", 0), 1) if auto_ok else a.area),
            "calibrated": self.floor.ok or auto_ok,
            "calib_mode": "manual" if self.floor.ok else ("auto" if auto_ok else None),
            "calib_status": None if (self.floor.ok or auto_ok or self.auto is None) else self.auto.why,
            "calib_time": self.floor.c["time"] if self.floor.ok else None,
            "hotspot": sp["hotspot"], "heat": sp["heat"], "texture": round(tex, 3), "cpu_temp": self.temp,
            "source": self.src.desc, "upload": self.upload, "uptime": int(now - self.t0),
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

    # ------------------------------------------------------------ incidents --
    def track_incident(self, now, grade, peak_grade, sat, cnt, avg, peak, by, jpg):
        """Record every period at grade E/F (or saturated): start, end, worst values and blurred snapshots."""
        rank = self.ORDER
        worst = max([grade if grade != "E*" else "E", peak_grade], key=lambda g: rank.get(g, -1))
        severe = sat or rank.get(worst, -1) >= 4
        inc = getattr(self, "incident", None)
        if severe:
            self.calm_since = None
            if inc is None:
                os.makedirs(self.a.incident_dir, exist_ok=True)
                n = len(self.incidents) + 1
                inc = self.incident = {"id": n, "start": time.strftime("%Y-%m-%d %H:%M:%S"), "start_t": now,
                                       "end": None, "seconds": 0, "worst_grade": None, "max_avg": 0.0,
                                       "max_peak": 0.0, "max_count": 0, "saturated": False,
                                       "source": self.src.desc, "counted_by": by, "snaps": []}
                self.event(f"Incident {n} started: grade {worst}" + (" (too dense to count)" if sat else ""), "alert")
            g_now = "E*" if sat and rank.get(worst, -1) < 4 else worst
            new_worst = inc["worst_grade"] is None or rank.get(g_now.rstrip("*"), -1) > rank.get(inc["worst_grade"].rstrip("*"), -1)
            if new_worst:
                inc["worst_grade"] = g_now
            inc["max_avg"] = round(max(inc["max_avg"], avg or 0), 2)
            inc["max_peak"] = round(max(inc["max_peak"], peak or 0), 2)
            inc["max_count"] = max(inc["max_count"], cnt)
            inc["saturated"] = inc["saturated"] or sat
            inc["seconds"] = int(now - inc["start_t"])
            if (new_worst or not inc["snaps"]) and len(inc["snaps"]) < 4:
                name = f"incident{inc['id']}_{time.strftime('%H%M%S')}.jpg"
                with open(os.path.join(self.a.incident_dir, name), "wb") as f:
                    f.write(jpg)                     # already blurred / pixelated
                inc["snaps"].append(name)
        elif inc is not None:
            self.calm_since = self.calm_since or now
            if now - self.calm_since >= 10:          # closed after 10 calm seconds
                inc["end"] = time.strftime("%Y-%m-%d %H:%M:%S")
                inc["seconds"] = int(self.calm_since - inc["start_t"])
                self.incidents.append({k: v for k, v in inc.items() if k != "start_t"})
                self.incident = None
                self.save_incidents()
                self.event(f"Incident {inc['id']} ended after {inc['seconds']} s (worst grade {inc['worst_grade']})")

    def save_incidents(self):
        os.makedirs(self.a.incident_dir, exist_ok=True)
        with open(os.path.join(self.a.incident_dir, "incidents.json"), "w") as f:
            json.dump(self.incidents, f, indent=1)

    def incident_list(self):
        cur = getattr(self, "incident", None)
        live = [] if cur is None else [{**{k: v for k, v in cur.items() if k != "start_t"}, "end": "ongoing"}]
        return self.incidents + live

    def use_file(self, path, name):
        """Count an uploaded photo or video instead of the camera."""
        new = Source(path, self.a.width, self.a.height, self.a.hold)
        new.desc = f"Uploaded {'photo' if new.kind == 'images' else 'video'}: {name}"
        old, self.src, self.upload = self.src, new, name
        if old is not self.live_src:
            old.stop()
        self.ema = None
        self.need_acc.set()
        self.event(f"Testing uploaded file: {name}")

    def use_live(self):
        if self.src is not self.live_src:
            old, self.src, self.upload = self.src, self.live_src, None
            old.stop()
            self.ema = None
            self.event("Back to the live camera")

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
            elif p == "/incidents.json":
                self._json(engine.incident_list())
            elif p.startswith("/incident/"):
                name = os.path.basename(p)
                path = os.path.join(engine.a.incident_dir, name)
                if name.endswith(".jpg") and os.path.isfile(path):
                    with open(path, "rb") as f:
                        self._send(200, "image/jpeg", f.read())
                else:
                    self._send(404, "text/plain", b"not found")
            elif p == "/report":
                self._send(200, "text/html; charset=utf-8", incident_report(engine).encode())
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
            p = self.path.split("?")[0]
            if p == "/upload":
                return self.upload()
            if p == "/source":
                try:
                    n = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(n) or b"{}")
                except Exception:
                    body = {}
                if body.get("live"):
                    engine.use_live()
                return self._json({"ok": True})
            if p != "/calibrate":
                return self._send(404, "text/plain", b"not found")
            try:
                n = int(self.headers.get("Content-Length", 0))
                engine.floor.set(json.loads(self.rfile.read(n) or b"{}"))
            except Exception:
                return self._json({"ok": False, "error": "Send 4 points between 0 and 1 and a positive size."}, 400)
            W, D = engine.floor.c["size_m"]
            engine.event(f"Floor calibrated: {W:g} × {D:g} m")
            self._json({"ok": True})
        def upload(self):
            from urllib.parse import unquote
            name = os.path.basename(unquote(self.headers.get("X-Filename", "upload")))[:80] or "upload"
            ext = os.path.splitext(name)[1].lower()
            if ext not in IMG_EXT + (".mp4", ".avi", ".mov", ".mkv", ".webm"):
                return self._json({"ok": False, "error": "Choose a photo (jpg, png, webp) or a video (mp4, mov, avi, mkv)."}, 400)
            n = int(self.headers.get("Content-Length", 0))
            if not 0 < n <= 300 * 1024 * 1024:
                return self._json({"ok": False, "error": "The file is empty or over 300 MB."}, 400)
            os.makedirs(engine.a.upload_dir, exist_ok=True)
            for old in sorted(glob.glob(os.path.join(engine.a.upload_dir, "*")), key=os.path.getmtime)[:-4]:
                try:
                    os.remove(old)                          # keep only the last few uploads
                except OSError:
                    pass
            safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
            path = os.path.join(engine.a.upload_dir, time.strftime("%H%M%S_") + safe)
            with open(path, "wb") as f:
                left = n
                while left > 0:
                    chunk = self.rfile.read(min(1 << 20, left))
                    if not chunk:
                        break
                    f.write(chunk)
                    left -= len(chunk)
            if ext in IMG_EXT and read_image(path) is None:
                return self._json({"ok": False, "error": f"Could not read {name} as a photo."}, 400)
            try:
                engine.use_file(path, name)
            except BaseException as e:                      # Source raises SystemExit on unreadable files
                return self._json({"ok": False, "error": f"Could not open {name}. Try another photo or an mp4 video."}, 400)
            self._json({"ok": True, "file": name})
    return H


def incident_report(engine):
    """Printable incident report: one section per incident, snapshots blurred at capture."""
    import html as H
    rows = engine.incident_list()
    secs = []
    for r in reversed(rows):
        imgs = "".join(f'<img src="/incident/{H.escape(n)}" alt="Blurred snapshot of incident {r["id"]}">'
                       for n in r.get("snaps", []))
        secs.append(f"""<section><h2>Incident {r['id']} <span class="g g{H.escape(str(r['worst_grade'])[0])}">
grade {H.escape(str(r['worst_grade']))}</span></h2>
<table><tr><th>Started</th><td>{H.escape(r['start'])}</td><th>Ended</th><td>{H.escape(str(r['end']))}</td></tr>
<tr><th>Duration</th><td>{r['seconds']} s</td><th>Most people in view</th><td>{r['max_count']}</td></tr>
<tr><th>Highest average density</th><td>{r['max_avg']} per m²</td><th>Densest spot</th><td>{(str(r['max_peak']) + ' per m²') if r['max_peak'] else 'not measured'}</td></tr>
<tr><th>Too dense to count</th><td>{'yes' if r.get('saturated') else 'no'}</td><th>Camera</th><td>{H.escape(r['source'])}</td></tr></table>
<div class="imgs">{imgs}</div></section>""")
    body = "".join(secs) or "<p>No incidents recorded yet. One is recorded each time the grade reaches E or F.</p>"
    return f"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NERISAL incident report</title>
<style>body{{font:15px/1.45 system-ui,Segoe UI,Arial,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;color:#14232c;background:#f6f7f5}}
h1{{font-size:26px;margin:0 0 4px}} .sub{{color:#4a5a62;margin:0 0 20px}} section{{background:#fff;border:1px solid #d5dcd9;border-radius:8px;padding:16px;margin:0 0 16px;break-inside:avoid}}
h2{{font-size:19px;margin:0 0 10px}} .g{{font-size:14px;padding:2px 8px;border-radius:4px;color:#fff;margin-left:8px}} .gE{{background:#d64528}} .gF{{background:#9b0d1f}}
table{{border-collapse:collapse;width:100%}} th,td{{text-align:left;padding:5px 8px;border-top:1px solid #e3e8e5}} th{{color:#4a5a62;font-weight:600;width:22%}}
.imgs{{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}} .imgs img{{width:calc(50% - 4px);border-radius:4px}}
@media print{{body{{background:#fff}} .noprint{{display:none}}}}</style>
<h1>NERISAL incident report</h1>
<p class="sub">{H.escape(engine.src.desc)} · generated {time.strftime('%d %b %Y, %H:%M')} on the device · faces blurred at capture ·
<a class="noprint" href="javascript:print()">Print or save as PDF</a></p>{body}"""


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
    ap.add_argument("--no-auto-calib", dest="auto_calib", action="store_false",
                    help="turn off self-calibration from head sizes (it is used when no floor is clicked)")
    ap.add_argument("--head-m", type=float, default=0.24, help="real height of a head box in metres")
    ap.add_argument("--fov", type=float, default=65, help="camera horizontal field of view in degrees")
    ap.add_argument("--auto-window", type=float, default=30, help="seconds of detections used to self-calibrate")
    ap.add_argument("--auto-depth", type=float, default=25, help="max depth of floor (m) used for the average")
    ap.add_argument("--auto-min-area", type=float, default=6, help="refuse self-calibration below this floor area (m²)")
    ap.add_argument("--peak-win", type=float, default=3.0, help="peak density window side in metres")
    ap.add_argument("--line", type=float, default=0.0, help="counting line at this fraction of height, 0=off")
    ap.add_argument("--esc-count", type=int, default=8, help="escalate when the watch tier sees this many")
    ap.add_argument("--esc-density", type=float, default=0.43, help="...or this density (p/m^2)")
    ap.add_argument("--esc-every", type=float, default=2.0,
                    help="rest (s) after each accurate pass, so the live view stays smooth")
    ap.add_argument("--hot-temp", type=float, default=76, help="above this CPU temp (C) rest 3x longer")
    ap.add_argument("--idle-every", type=float, default=15.0, help="accurate pass at least this often anyway")
    ap.add_argument("--hold-acc", type=float, default=5.0, help="seconds an accurate result stays valid")
    ap.add_argument("--smooth-s", type=float, default=1.5, help="count smoothing time in seconds (0.3 = fast)")
    ap.add_argument("--grade-up", type=float, default=2.0, help="seconds a worse grade must hold before it shows")
    ap.add_argument("--grade-down", type=float, default=5.0, help="seconds a better grade must hold before it shows")
    ap.add_argument("--tex-thr", type=float, default=0.22)
    ap.add_argument("--sat-count", type=int, default=15)
    ap.add_argument("--trend", type=float, default=60, help="seconds of history for time-to-critical")
    ap.add_argument("--stream-w", type=int, default=960, help="longest side of the browser stream")
    ap.add_argument("--no-blur", action="store_true")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--proof", default=os.path.join(HERE, "proof.json"), help="results file for the Results drawer")
    ap.add_argument("--logdir", default="live_logs")
    ap.add_argument("--snapdir", default="live_snaps")
    ap.add_argument("--incident-dir", default="incidents", help="incident records and snapshots")
    ap.add_argument("--upload-dir", default="uploads", help="photos and videos tested from the dashboard")
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
